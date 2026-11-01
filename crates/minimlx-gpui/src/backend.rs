use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use minimlx_protocol::{Request, ResponsePayload};
use minimlx_supervisor::PythonWorker;
use serde_json::Value;

const DEFAULT_API_PORT: u16 = 18765;
const DEFAULT_API_MAX_TOKENS: u32 = 32768;
const DEFAULT_API_MAX_CONTEXT: u32 = 32768;
const MODEL_FILE_REQUEST_RETRIES: usize = 5;
const GIB: u64 = 1024 * 1024 * 1024;
// Match oMLX's aggressive memory tier: 2% of RAM, bounded to 1.5–4 GiB.
const MINIMUM_MEMORY_HEADROOM: u64 = GIB + GIB / 2;
const MAXIMUM_MEMORY_HEADROOM: u64 = 4 * GIB;
const MEMORY_HEADROOM_PERCENT: u64 = 2;
const MODEL_WEIGHT_OVERHEAD_NUMERATOR: u64 = 5;
const MODEL_WEIGHT_OVERHEAD_DENOMINATOR: u64 = 4;

// Keep the machine awake only while a download is active. `-i` prevents
// idle system sleep without keeping the display on; `-w` also releases the
// assertion if MiniMLX terminates unexpectedly.
struct DownloadSleepGuard {
    #[cfg(target_os = "macos")]
    child: Child,
}

impl DownloadSleepGuard {
    fn acquire() -> Result<Self, String> {
        #[cfg(target_os = "macos")]
        {
            let child = Command::new("/usr/bin/caffeinate")
                .args(["-i", "-w", &std::process::id().to_string()])
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .map_err(|error| {
                    format!("Could not prevent idle sleep during download: {error}")
                })?;
            Ok(Self { child })
        }
        #[cfg(not(target_os = "macos"))]
        {
            Ok(Self {})
        }
    }
}

impl Drop for DownloadSleepGuard {
    fn drop(&mut self) {
        #[cfg(target_os = "macos")]
        {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

#[cfg(target_os = "macos")]
fn system_physical_memory_bytes() -> u64 {
    let pages = unsafe { libc::sysconf(libc::_SC_PHYS_PAGES) };
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if pages > 0 && page_size > 0 {
        (pages as u64).saturating_mul(page_size as u64)
    } else {
        0
    }
}

#[cfg(not(target_os = "macos"))]
fn system_physical_memory_bytes() -> u64 {
    0
}

pub struct ModelEntry {
    pub name: String,
    pub path: String,
    pub size_bytes: u64,
    pub quantization: Option<String>,
    pub model_type: Option<String>,
    pub architecture: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct ModelInferenceOptions {
    pub alias: Option<String>,
    pub max_tokens: Option<u32>,
    pub max_context_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub min_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub force_sampling: Option<bool>,
    pub expert_cache_gb: Option<f32>,
    pub chat_template_kwargs: Option<serde_json::Map<String, Value>>,
    /// Optional Qwen MTP drafter checkpoint attached to this target model.
    pub mtp_draft_model: Option<String>,
    /// Optional DFlash draft checkpoint attached to this target model.
    pub dflash_draft_model: Option<String>,
}

#[derive(Clone, Debug)]
pub struct ContextBenchmarkOptions {
    pub corpus_profile: String,
    pub contexts: Vec<usize>,
    pub warmup: usize,
    pub repeats: usize,
    pub max_tokens: usize,
    pub prefill_step: usize,
    pub ignore_eos: bool,
}

#[derive(Clone)]
pub struct IncompleteDownload {
    pub repo_id: String,
    pub size_bytes: Option<u64>,
    pub source: String,
}

#[derive(Clone, serde::Deserialize)]
pub struct ModelSearchResult {
    pub repo_id: String,
    pub downloads: Option<u64>,
    pub likes: Option<u64>,
    pub parameters: Option<u64>,
    pub size_bytes: Option<u64>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ModelSearchSort {
    Name,
    Size,
    #[default]
    Downloads,
    Likes,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ModelSearchSortDirection {
    Ascending,
    #[default]
    Descending,
}

pub fn sort_model_search_results(
    results: &mut [ModelSearchResult],
    sort: ModelSearchSort,
    direction: ModelSearchSortDirection,
) {
    results.sort_by(|left, right| {
        let compare_optional = |left: Option<u64>, right: Option<u64>| match (left, right) {
            (Some(left), Some(right)) => match direction {
                ModelSearchSortDirection::Ascending => left.cmp(&right),
                ModelSearchSortDirection::Descending => right.cmp(&left),
            },
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => std::cmp::Ordering::Equal,
        };
        let order = match sort {
            ModelSearchSort::Name => {
                let order = left
                    .repo_id
                    .to_lowercase()
                    .cmp(&right.repo_id.to_lowercase());
                match direction {
                    ModelSearchSortDirection::Ascending => order,
                    ModelSearchSortDirection::Descending => order.reverse(),
                }
            }
            ModelSearchSort::Size => compare_optional(left.size_bytes, right.size_bytes),
            ModelSearchSort::Downloads => compare_optional(left.downloads, right.downloads),
            ModelSearchSort::Likes => compare_optional(left.likes, right.likes),
        };
        order.then_with(|| left.repo_id.cmp(&right.repo_id))
    });
}

#[derive(Clone)]
pub struct DownloadContext {
    download_directory: PathBuf,
    huggingface_endpoint: String,
    modelscope_endpoint: String,
    http_proxy: String,
    https_proxy: String,
    no_proxy: String,
    ca_bundle: String,
}

pub struct Backend {
    root: PathBuf,
    python: PathBuf,
    model_directory: PathBuf,
    model_directories: Vec<PathBuf>,
    worker: Option<PythonWorker>,
    socket_path: PathBuf,
    pub loaded_model: Option<String>,
    api_process: Option<Child>,
    api_port: Option<u16>,
    last_api_model_path: Option<String>,
    last_api_model_options: Option<ModelInferenceOptions>,
    last_worker_model_options: Option<ModelInferenceOptions>,
    configured_api_port: u16,
    configured_api_max_tokens: u32,
    configured_host: String,
    configured_max_context: u32,
    configured_temperature: f32,
    configured_top_p: f32,
    configured_top_k: u32,
    configured_repetition_penalty: f32,
    configured_log_level: String,
    configured_history: bool,
    configured_history_db: PathBuf,
    configured_auth_required: bool,
    configured_api_key: Option<String>,
    configured_huggingface_endpoint: String,
    configured_modelscope_endpoint: String,
    configured_http_proxy: String,
    configured_https_proxy: String,
    configured_no_proxy: String,
    configured_ca_bundle: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct RemoteModelFile {
    path: String,
    size: Option<u64>,
    revision: String,
}

fn build_download_client(context: &DownloadContext) -> Result<reqwest::blocking::Client, String> {
    let no_proxy = || reqwest::NoProxy::from_string(&context.no_proxy);
    let mut builder = reqwest::blocking::Client::builder()
        .connect_timeout(Duration::from_secs(20))
        .no_proxy();
    if !context.http_proxy.trim().is_empty() {
        let proxy = reqwest::Proxy::http(context.http_proxy.trim())
            .map_err(|error| format!("invalid HTTP proxy: {error}"))?
            .no_proxy(no_proxy());
        builder = builder.proxy(proxy);
    }
    if !context.https_proxy.trim().is_empty() {
        let proxy = reqwest::Proxy::https(context.https_proxy.trim())
            .map_err(|error| format!("invalid HTTPS proxy: {error}"))?
            .no_proxy(no_proxy());
        builder = builder.proxy(proxy);
    }
    if !context.ca_bundle.trim().is_empty() {
        let pem = fs::read(context.ca_bundle.trim())
            .map_err(|error| format!("could not read CA bundle: {error}"))?;
        let certificates = reqwest::Certificate::from_pem_bundle(&pem)
            .map_err(|error| format!("invalid CA bundle: {error}"))?;
        for certificate in certificates {
            builder = builder.add_root_certificate(certificate);
        }
    }
    builder
        .build()
        .map_err(|error| format!("could not configure model hub connection: {error}"))
}

fn huggingface_model_search_params(query: &str) -> Vec<(&'static str, &str)> {
    let mut params = vec![("search", query)];
    // DFlash draft checkpoints are supported by the MLX DFlash runtime but
    // are published as Transformers repositories, so Hugging Face's `mlx`
    // library filter hides the official checkpoint from the in-app search.
    if !query.to_ascii_lowercase().contains("dflash") {
        params.push(("filter", "mlx"));
    }
    params.extend([
        ("sort", "downloads"),
        ("limit", "50"),
        ("expand", "downloads"),
        ("expand", "likes"),
        ("expand", "safetensors"),
    ]);
    params
}

fn huggingface_model_search_url(context: &DownloadContext) -> String {
    format!(
        "{}/api/models",
        context.huggingface_endpoint.trim_end_matches('/')
    )
}

fn modelscope_model_search_params(query: &str) -> Vec<(&'static str, &str)> {
    vec![
        ("search", query),
        ("filter.library", "mlx"),
        ("sort", "downloads"),
        ("page_size", "50"),
    ]
}

fn modelscope_model_search_url(context: &DownloadContext) -> String {
    format!(
        "{}/openapi/v1/models",
        context.modelscope_endpoint.trim_end_matches('/')
    )
}

fn huggingface_repo_size(
    client: &reqwest::blocking::Client,
    context: &DownloadContext,
    repo_id: &str,
) -> Option<u64> {
    let url = format!(
        "{}/api/models/{repo_id}",
        context.huggingface_endpoint.trim_end_matches('/')
    );
    let payload: Value = client
        .get(url)
        .timeout(Duration::from_secs(20))
        .query(&[("blobs", "true")])
        .send()
        .ok()?
        .error_for_status()
        .ok()?
        .json()
        .ok()?;
    parse_huggingface_repo_size(&payload)
}

fn parse_huggingface_repo_size(payload: &Value) -> Option<u64> {
    let total = payload
        .get("siblings")?
        .as_array()?
        .iter()
        .filter_map(|file| {
            file.get("size")
                .and_then(Value::as_u64)
                .or_else(|| file.get("lfs")?.get("size")?.as_u64())
        })
        .fold(0_u64, u64::saturating_add);
    (total > 0).then_some(total)
}

fn fetch_model_card(
    client: &reqwest::blocking::Client,
    context: &DownloadContext,
    repo_id: &str,
    source: &str,
) -> Result<String, String> {
    let files = match source {
        "huggingface" => list_huggingface_files(client, context, repo_id)?,
        "modelscope" => list_modelscope_files(client, context, repo_id)?,
        _ => return Err("unsupported model hub".into()),
    };
    let readme = files
        .iter()
        .find(|file| file.path.eq_ignore_ascii_case("README.md"))
        .ok_or_else(|| "model repository does not contain README.md".to_string())?;
    let url = match source {
        "huggingface" => huggingface_file_url(context, repo_id, &readme.revision, &readme.path)?,
        "modelscope" => modelscope_file_url(context, repo_id, &readme.revision, &readme.path)?,
        _ => unreachable!(),
    };
    let response = client
        .get(url)
        .timeout(Duration::from_secs(30))
        .send()
        .and_then(reqwest::blocking::Response::error_for_status)
        .map_err(|error| format!("could not fetch model card: {error}"))?;
    const MAX_MODEL_CARD_BYTES: u64 = 1_048_576;
    let mut card = Vec::new();
    response
        .take(MAX_MODEL_CARD_BYTES)
        .read_to_end(&mut card)
        .map_err(|error| format!("could not read model card: {error}"))?;
    Ok(String::from_utf8_lossy(&card).into_owned())
}

fn validate_repo_id(repo_id: &str) -> Result<&str, String> {
    let repo_id = repo_id.trim();
    let parts: Vec<_> = repo_id.split('/').collect();
    if repo_id.is_empty()
        || repo_id.len() > 256
        || parts.is_empty()
        || parts.len() > 2
        || parts.iter().any(|part| {
            part.is_empty()
                || *part == "."
                || *part == ".."
                || part.chars().any(|character| {
                    character.is_control()
                        || character.is_whitespace()
                        || matches!(character, '\\' | '?' | '#' | '%')
                })
        })
    {
        return Err("enter a valid model repository ID".into());
    }
    Ok(repo_id)
}

fn validate_remote_path(path: &str) -> Result<(), String> {
    let relative = Path::new(path);
    if path.is_empty()
        || path.len() > 1024
        || relative.is_absolute()
        || path.contains('\\')
        || relative
            .components()
            .any(|part| !matches!(part, std::path::Component::Normal(_)))
    {
        return Err(format!("model hub returned an unsafe file path: {path:?}"));
    }
    Ok(())
}

fn list_huggingface_files(
    client: &reqwest::blocking::Client,
    context: &DownloadContext,
    repo_id: &str,
) -> Result<Vec<RemoteModelFile>, String> {
    let url = format!(
        "{}/api/models/{}",
        context.huggingface_endpoint.trim_end_matches('/'),
        repo_id
    );
    let payload: Value = client
        .get(url)
        .timeout(Duration::from_secs(60))
        .query(&[("blobs", "true")])
        .send()
        .and_then(reqwest::blocking::Response::error_for_status)
        .map_err(|error| format!("could not read Hugging Face repository: {error}"))?
        .json()
        .map_err(|error| format!("invalid Hugging Face repository response: {error}"))?;
    let revision = payload
        .get("sha")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .unwrap_or("main")
        .to_string();
    let siblings = payload
        .get("siblings")
        .and_then(Value::as_array)
        .ok_or_else(|| "Hugging Face repository response has no file list".to_string())?;
    let mut files = Vec::with_capacity(siblings.len());
    for sibling in siblings {
        let Some(path) = sibling.get("rfilename").and_then(Value::as_str) else {
            continue;
        };
        validate_remote_path(path)?;
        // Git's repository attributes do not affect model loading. Avoid an
        // extra request to mirrors which may rate-limit each resolved file.
        if is_unneeded_huggingface_metadata(path) {
            continue;
        }
        let size = sibling.get("size").and_then(Value::as_u64).or_else(|| {
            sibling
                .get("lfs")
                .and_then(|lfs| lfs.get("size"))
                .and_then(Value::as_u64)
        });
        files.push(RemoteModelFile {
            path: path.to_string(),
            size,
            revision: revision.clone(),
        });
    }
    Ok(files)
}

fn list_modelscope_files(
    client: &reqwest::blocking::Client,
    context: &DownloadContext,
    repo_id: &str,
) -> Result<Vec<RemoteModelFile>, String> {
    let mut url = reqwest::Url::parse(&format!(
        "{}/api/v1/models/{repo_id}/repo/files",
        context.modelscope_endpoint.trim_end_matches('/')
    ))
    .map_err(|error| format!("invalid ModelScope API URL: {error}"))?;
    url.query_pairs_mut()
        .append_pair("Revision", "master")
        .append_pair("Recursive", "True");
    let payload: Value = client
        .get(url)
        .timeout(Duration::from_secs(60))
        .send()
        .and_then(reqwest::blocking::Response::error_for_status)
        .map_err(|error| format!("could not read ModelScope repository: {error}"))?
        .json()
        .map_err(|error| format!("invalid ModelScope repository response: {error}"))?;
    let data = payload.get("Data").or_else(|| payload.get("data"));
    let entries = data
        .and_then(|value| value.get("Files").or_else(|| value.get("files")))
        .and_then(Value::as_array)
        .ok_or_else(|| {
            payload
                .get("Message")
                .or_else(|| payload.get("message"))
                .and_then(Value::as_str)
                .map(str::to_string)
                .unwrap_or_else(|| "ModelScope repository response has no file list".into())
        })?;
    let mut files = Vec::with_capacity(entries.len());
    for entry in entries {
        let kind = entry
            .get("Type")
            .or_else(|| entry.get("type"))
            .and_then(Value::as_str);
        if kind.is_some_and(|kind| !kind.eq_ignore_ascii_case("blob")) {
            continue;
        }
        let Some(path) = entry
            .get("Path")
            .or_else(|| entry.get("path"))
            .and_then(Value::as_str)
        else {
            continue;
        };
        validate_remote_path(path)?;
        files.push(RemoteModelFile {
            path: path.to_string(),
            size: entry
                .get("Size")
                .or_else(|| entry.get("size"))
                .and_then(Value::as_u64),
            revision: "master".into(),
        });
    }
    Ok(files)
}

fn huggingface_file_url(
    context: &DownloadContext,
    repo_id: &str,
    revision: &str,
    file_path: &str,
) -> Result<reqwest::Url, String> {
    // Use a URL path assembled from individually escaped segments, preserving the
    // configured endpoint's optional path prefix (e.g. a self-hosted Hub).
    let base = context.huggingface_endpoint.trim_end_matches('/');
    let mut url = reqwest::Url::parse(&format!("{base}/"))
        .map_err(|error| format!("invalid Hugging Face endpoint: {error}"))?;
    {
        let mut segments = url
            .path_segments_mut()
            .map_err(|_| "Hugging Face endpoint cannot accept file paths".to_string())?;
        segments.pop_if_empty();
        segments.extend(repo_id.split('/'));
        segments.push("resolve");
        segments.push(revision);
        for segment in file_path.split('/') {
            segments.push(segment);
        }
    }
    Ok(url)
}

fn modelscope_file_url(
    context: &DownloadContext,
    repo_id: &str,
    revision: &str,
    file_path: &str,
) -> Result<reqwest::Url, String> {
    let url = reqwest::Url::parse(&format!(
        "{}/api/v1/models/{repo_id}/repo",
        context.modelscope_endpoint.trim_end_matches('/')
    ))
    .map_err(|error| format!("invalid ModelScope download URL: {error}"))?;
    let mut url = url;
    url.query_pairs_mut()
        .append_pair("Revision", revision)
        .append_pair("FilePath", file_path);
    Ok(url)
}

fn download_one_file(
    client: &reqwest::blocking::Client,
    url: &reqwest::Url,
    destination: &Path,
    file: &RemoteModelFile,
    on_bytes: &mut impl FnMut(i64) -> bool,
) -> Result<bool, String> {
    let relative = Path::new(&file.path);
    let final_path = destination.join(relative);
    let partial_path = destination
        .join(".cache/huggingface/download")
        .join(format!("{}.incomplete", file.path));
    if file
        .size
        .is_some_and(|size| fs::metadata(&final_path).is_ok_and(|metadata| metadata.len() == size))
    {
        return Ok(true);
    }
    let canonical_root = destination
        .canonicalize()
        .map_err(|error| format!("could not resolve model download directory: {error}"))?;
    ensure_safe_parent(
        &canonical_root,
        Path::new(".cache/huggingface/download")
            .join(format!("{}.incomplete", file.path))
            .as_path(),
    )?;
    ensure_safe_parent(&canonical_root, relative)?;

    let mut offset = fs::metadata(&partial_path)
        .map(|metadata| metadata.len())
        .unwrap_or(0);
    let partial_bytes = offset;
    if fs::symlink_metadata(&partial_path).is_ok_and(|metadata| metadata.file_type().is_symlink()) {
        return Err(format!(
            "partial download path must not be a symlink: {}",
            partial_path.display()
        ));
    }
    if file.size.is_some_and(|size| offset > size) {
        offset = 0;
    }
    if file.size.is_some_and(|size| offset == size && size > 0) {
        fs::rename(&partial_path, &final_path).map_err(|error| {
            format!(
                "could not finalize resumed model file {}: {error}",
                file.path
            )
        })?;
        return Ok(true);
    }
    let mut response = send_file_request(client, url, offset)?;
    if response.status() == reqwest::StatusCode::RANGE_NOT_SATISFIABLE && offset > 0 {
        offset = 0;
        response = send_file_request(client, url, 0)?;
    }
    let range_start = response
        .headers()
        .get(reqwest::header::CONTENT_RANGE)
        .and_then(|value| value.to_str().ok())
        .and_then(parse_content_range_start);
    if offset > 0
        && matches!(
            response.status(),
            reqwest::StatusCode::OK | reqwest::StatusCode::PARTIAL_CONTENT
        )
        && range_start.is_some()
        && range_start != Some(offset)
    {
        offset = 0;
        response = send_file_request(client, url, 0)?;
    }
    let status = response.status();
    if !status.is_success() {
        return Err(format!("file download failed (HTTP {status}): {}", url));
    }
    let response_offset = response
        .headers()
        .get(reqwest::header::CONTENT_RANGE)
        .and_then(|value| value.to_str().ok())
        .and_then(parse_content_range_start);
    let append = offset > 0
        && matches!(
            response.status(),
            reqwest::StatusCode::OK | reqwest::StatusCode::PARTIAL_CONTENT
        )
        && response_offset == Some(offset);
    let mut output = fs::OpenOptions::new()
        .create(true)
        .write(true)
        .append(append)
        .truncate(!append)
        .open(&partial_path)
        .map_err(|error| format!("could not open partial model file: {error}"))?;
    if !append && partial_bytes > 0 && !on_bytes(0) {
        return Ok(false);
    }
    let mut buffer = [0u8; 256 * 1024];
    loop {
        let count = response.read(&mut buffer).map_err(|error| {
            format!("model file transfer interrupted; partial data was kept: {error}")
        })?;
        if count == 0 {
            break;
        }
        output
            .write_all(&buffer[..count])
            .map_err(|error| format!("could not write partial model file: {error}"))?;
        if !on_bytes(count as i64) {
            output
                .flush()
                .map_err(|error| format!("could not flush paused model file: {error}"))?;
            return Ok(false);
        }
    }
    output
        .flush()
        .map_err(|error| format!("could not flush downloaded model file: {error}"))?;
    let actual_size = output
        .metadata()
        .map_err(|error| format!("could not inspect downloaded model file: {error}"))?
        .len();
    if file.size.is_some_and(|expected| expected != actual_size) {
        return Err(format!(
            "incomplete model file {}: expected {} bytes, received {} bytes; resume is available",
            file.path,
            file.size.unwrap_or_default(),
            actual_size
        ));
    }
    drop(output);
    fs::rename(&partial_path, &final_path)
        .map_err(|error| format!("could not finalize model file {}: {error}", file.path))?;
    Ok(true)
}

fn ensure_safe_parent(root: &Path, file_path: &Path) -> Result<(), String> {
    let mut parent = root.to_path_buf();
    let components: Vec<_> = file_path.components().collect();
    for component in components.iter().take(components.len().saturating_sub(1)) {
        let std::path::Component::Normal(segment) = component else {
            return Err("model hub returned an unsafe file path".into());
        };
        parent.push(segment);
        match fs::symlink_metadata(&parent) {
            Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_dir() => {
                return Err(format!(
                    "model path contains a non-directory or symlink: {}",
                    parent.display()
                ));
            }
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                fs::create_dir(&parent)
                    .map_err(|error| format!("could not create model directory: {error}"))?;
            }
            Err(error) => return Err(format!("could not inspect model directory: {error}")),
        }
        if !parent.starts_with(root) {
            return Err("model file path escaped its destination directory".into());
        }
    }
    Ok(())
}

fn send_file_request(
    client: &reqwest::blocking::Client,
    url: &reqwest::Url,
    offset: u64,
) -> Result<reqwest::blocking::Response, String> {
    for attempt in 0..=MODEL_FILE_REQUEST_RETRIES {
        let mut request = client.get(url.clone());
        if offset > 0 {
            request = request.header(reqwest::header::RANGE, format!("bytes={offset}-"));
        }
        match request.send() {
            Ok(response)
                if attempt < MODEL_FILE_REQUEST_RETRIES
                    && should_retry_download_status(response.status()) =>
            {
                let retry_after = response
                    .headers()
                    .get(reqwest::header::RETRY_AFTER)
                    .and_then(|value| value.to_str().ok())
                    .map(str::to_string);
                std::thread::sleep(download_retry_delay(retry_after.as_deref(), attempt));
            }
            Ok(response) => return Ok(response),
            Err(error)
                if attempt < MODEL_FILE_REQUEST_RETRIES
                    && (error.is_timeout() || error.is_connect()) =>
            {
                std::thread::sleep(download_retry_delay(None, attempt));
            }
            Err(error) => return Err(format!("could not fetch model file: {error}")),
        }
    }
    unreachable!("the retry loop always returns the final response or error")
}

fn is_unneeded_huggingface_metadata(path: &str) -> bool {
    path == ".gitattributes"
}

fn should_retry_download_status(status: reqwest::StatusCode) -> bool {
    status == reqwest::StatusCode::TOO_MANY_REQUESTS || status.is_server_error()
}

fn download_retry_delay(retry_after: Option<&str>, attempt: usize) -> Duration {
    retry_after
        .and_then(|value| value.parse::<u64>().ok())
        .map(|seconds| Duration::from_secs(seconds.min(60)))
        .unwrap_or_else(|| Duration::from_secs(1 << attempt.min(5)))
}

fn parse_content_range_start(value: &str) -> Option<u64> {
    value
        .strip_prefix("bytes ")?
        .split_once('-')?
        .0
        .parse()
        .ok()
}

impl Backend {
    pub(crate) fn running_model_api_id(&self, path: &str) -> Option<String> {
        if self.api_process.is_none() || !self.last_api_model_path.as_deref()
            .is_some_and(|running| same_filesystem_path(running, path)) {
            return None;
        }
        Some(model_api_id(Path::new(path), self.last_api_model_options.as_ref()
            .and_then(|options| options.alias.as_deref())))
    }

    pub fn prepare_ssd_model(&self, model_path: &str) -> Result<(), String> {
        if !self.is_listed_model_path(model_path) {
            return Err("Model is not in the model library".into());
        }
        let output = Command::new(&self.python)
            .current_dir(&self.root)
            .env("PYTHONPATH", self.root.join("python"))
            .args(["-m", "minimlx_runtime.prepare_offload", model_path])
            .output()
            .map_err(|e| format!("Could not start SSD preparation: {e}"))?;
        if !output.status.success() {
            return Err(format!(
                "SSD preparation failed: {} {}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        Ok(())
    }

    fn execution_model_path(
        &self,
        model_path: &str,
        options: Option<&ModelInferenceOptions>,
        context: Option<u32>,
        replaces_loaded_model: bool,
    ) -> Result<String, String> {
        let source = Path::new(model_path);
        let full_memory_error =
            match self.ensure_memory_budget(model_path, options, context, replaces_loaded_model) {
                Ok(()) => return Ok(model_path.to_string()),
                Err(error) => error,
            };
        let config = fs::read_to_string(source.join("config.json"))
            .ok()
            .and_then(|s| serde_json::from_str::<Value>(&s).ok());
        let supports_paging = config.as_ref().is_some_and(supports_ssd_paging_config);
        if !supports_paging {
            return Err(full_memory_error);
        }
        let copy = source.join(".minimlx-ssd");
        if fs::symlink_metadata(&copy).is_ok_and(|m| m.file_type().is_symlink()) {
            return Err("SSD copy must not be a symbolic link".into());
        }
        if !copy.exists() {
            self.prepare_ssd_model(model_path).map_err(|error| format!("Full model cannot fit in memory: {full_memory_error}\nSSD paging preparation failed: {error}"))?;
        }
        if !is_model_dir(&copy) || !copy.join(".minimlx-ssd-manifest.json").is_file() {
            return Err("SSD paging copy is incomplete; it cannot be used for loading.".into());
        }
        let copy_path = copy.to_string_lossy().into_owned();
        self.ensure_memory_budget(&copy_path, options, context, replaces_loaded_model)
            .map_err(|error| format!("SSD paging also cannot fit in available memory: {error}"))?;
        Ok(copy_path)
    }
    pub fn fetch_model_card(
        context: &DownloadContext,
        repo_id: &str,
        source: &str,
    ) -> Result<String, String> {
        validate_repo_id(repo_id)?;
        let client = build_download_client(context)?;
        fetch_model_card(&client, context, repo_id, source)
    }

    pub fn new(root: PathBuf, python: PathBuf, model_directory: PathBuf) -> Self {
        let socket_path =
            std::env::temp_dir().join(format!("minimlx-gui-{}.sock", std::process::id()));
        Self {
            root,
            python,
            model_directory,
            model_directories: Vec::new(),
            worker: None,
            socket_path,
            loaded_model: None,
            api_process: None,
            api_port: None,
            last_api_model_path: None,
            last_api_model_options: None,
            last_worker_model_options: None,
            configured_api_port: DEFAULT_API_PORT,
            configured_api_max_tokens: DEFAULT_API_MAX_TOKENS,
            configured_host: "127.0.0.1".into(),
            configured_max_context: DEFAULT_API_MAX_CONTEXT,
            configured_temperature: 1.0,
            configured_top_p: 0.95,
            configured_top_k: 0,
            configured_repetition_penalty: 1.0,
            configured_log_level: "INFO".into(),
            configured_history: true,
            configured_history_db: std::env::var_os("HOME")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".minimlx/usage.sqlite3"),
            configured_auth_required: false,
            configured_api_key: None,
            configured_huggingface_endpoint: "https://huggingface.co".into(),
            configured_modelscope_endpoint: "https://modelscope.cn".into(),
            configured_http_proxy: String::new(),
            configured_https_proxy: String::new(),
            configured_no_proxy: String::new(),
            configured_ca_bundle: String::new(),
        }
    }

    pub fn models(&self) -> Vec<ModelEntry> {
        discover_model_directories(&self.model_directories)
            .into_iter()
            .map(|path| {
                let name = model_display_name(&path);
                let size_bytes = directory_size(&path);
                let quantization = fs::read_to_string(path.join("config.json"))
                    .ok()
                    .and_then(|text| serde_json::from_str::<Value>(&text).ok())
                    .and_then(|config| config.get("quantization_config").cloned())
                    .and_then(|config| {
                        let bits = config.get("bits").and_then(Value::as_u64)?;
                        let group_size = config.get("group_size").and_then(Value::as_u64);
                        Some(match group_size {
                            Some(group_size) => format!("{bits}-bit · group {group_size}"),
                            None => format!("{bits}-bit"),
                        })
                    });
                let config = fs::read_to_string(path.join("config.json"))
                    .ok()
                    .and_then(|text| serde_json::from_str::<Value>(&text).ok());
                let model_type = config
                    .as_ref()
                    .and_then(|config| config.get("model_type"))
                    .and_then(Value::as_str)
                    .map(str::to_string);
                let architecture = config
                    .as_ref()
                    .and_then(|config| config.get("architectures"))
                    .and_then(Value::as_array)
                    .map(|architectures| {
                        architectures
                            .iter()
                            .filter_map(Value::as_str)
                            .collect::<Vec<_>>()
                            .join(", ")
                    })
                    .filter(|value| !value.is_empty());
                ModelEntry {
                    name,
                    path: path.display().to_string(),
                    size_bytes,
                    quantization,
                    model_type,
                    architecture,
                }
            })
            .collect()
    }

    fn is_listed_model_path(&self, model_path: &str) -> bool {
        is_model_dir(Path::new(model_path))
            && discover_model_directories(&self.model_directories)
                .iter()
                .any(|listed| same_filesystem_path(&listed.display().to_string(), model_path))
    }

    pub fn incomplete_downloads(&self) -> Vec<IncompleteDownload> {
        let Some(root) = self.model_directories.first() else {
            return Vec::new();
        };
        let Ok(entries) = fs::read_dir(root) else {
            return Vec::new();
        };
        let mut downloads: Vec<_> = entries
            .flatten()
            .map(|entry| entry.path())
            .filter_map(|path| {
                if !path.is_dir()
                    || is_managed_model_dir(&path)
                    || !is_incomplete_download_dir(&path)
                {
                    return None;
                }
                let repo_id = path
                    .file_name()?
                    .to_str()?
                    .split_once("--")
                    .map(|(owner, repo)| format!("{owner}/{repo}"))?;
                let source = fs::read_to_string(path.join(".minimlx-download-source"))
                    .ok()
                    .map(|source| source.trim().to_string())
                    .filter(|source| matches!(source.as_str(), "huggingface" | "modelscope"))
                    .unwrap_or_else(|| "huggingface".into());
                let files = load_download_manifest(&path);
                Some(IncompleteDownload {
                    repo_id,
                    size_bytes: files.map(|files| downloaded_bytes_for_files(&path, &files)),
                    source,
                })
            })
            .collect();
        downloads.sort_by(|left, right| left.repo_id.cmp(&right.repo_id));
        downloads
    }

    pub fn delete_incomplete_download(&self, repo_id: &str) -> Result<(), String> {
        let repo_id = validate_repo_id(repo_id)?;
        let root = self
            .model_directories
            .first()
            .ok_or_else(|| "model library directory is not configured".to_string())?
            .canonicalize()
            .map_err(|error| format!("could not resolve model library: {error}"))?;
        let destination = root.join(repo_id.replace('/', "--"));
        let metadata = fs::symlink_metadata(&destination)
            .map_err(|error| format!("could not inspect incomplete download: {error}"))?;
        if metadata.file_type().is_symlink() || !metadata.is_dir() {
            return Err("incomplete download path is not a regular directory".into());
        }
        let canonical_destination = destination
            .canonicalize()
            .map_err(|error| format!("could not resolve incomplete download: {error}"))?;
        if canonical_destination.parent() != Some(root.as_path())
            || !is_incomplete_download_dir(&canonical_destination)
            || is_managed_model_dir(&canonical_destination)
        {
            return Err("target is not an incomplete model download".into());
        }
        fs::remove_dir_all(canonical_destination)
            .map_err(|error| format!("could not delete incomplete download: {error}"))
    }

    pub fn delete_model(&mut self, model_path: &str) -> Result<Vec<ModelEntry>, String> {
        if self.api_running()
            || self
                .loaded_model
                .as_deref()
                .is_some_and(|loaded| same_filesystem_path(loaded, model_path))
        {
            return Err("unload the model before deleting it".into());
        }
        let destination = Path::new(model_path);
        let metadata = fs::symlink_metadata(destination)
            .map_err(|error| format!("could not inspect model directory: {error}"))?;
        if metadata.file_type().is_symlink() || !metadata.is_dir() {
            return Err("model path is not a regular directory".into());
        }
        let canonical_destination = destination
            .canonicalize()
            .map_err(|error| format!("could not resolve model directory: {error}"))?;
        let inside_library = self.model_directories.iter().any(|root| {
            root.canonicalize().is_ok_and(|root| {
                canonical_destination != root && canonical_destination.starts_with(root)
            })
        });
        if !inside_library || !is_managed_model_dir(&canonical_destination) {
            return Err("target is not a model inside a configured model library".into());
        }
        fs::remove_dir_all(&canonical_destination)
            .map_err(|error| format!("could not delete model: {error}"))?;
        Ok(self.models())
    }

    pub fn latest_report(&self) -> (usize, String) {
        let directory = self.report_directory();
        let mut reports: Vec<_> = fs::read_dir(directory)
            .into_iter()
            .flatten()
            .flatten()
            .filter(|entry| {
                entry
                    .path()
                    .extension()
                    .is_some_and(|ext| ext == "md" || ext == "txt")
            })
            .collect();
        reports.sort_by_key(|entry| entry.metadata().and_then(|meta| meta.modified()).ok());
        let count = reports.len();
        let latest = reports
            .last()
            .and_then(|entry| fs::read_to_string(entry.path()).ok())
            .unwrap_or_default();
        (count, latest)
    }

    pub fn report_directory(&self) -> PathBuf {
        if self.root.join("runtime/bin/python3").is_file() {
            std::env::var_os("HOME").map(PathBuf::from).unwrap_or_default().join(".minimlx/Reports/Performance")
        } else {
            self.root.join("Reports/Performance")
        }
    }

    pub fn download_context(&self) -> DownloadContext {
        DownloadContext {
            download_directory: self.model_directories.first().cloned().unwrap_or_else(|| {
                self.model_directory
                    .parent()
                    .unwrap_or(&self.model_directory)
                    .to_path_buf()
            }),
            huggingface_endpoint: self.configured_huggingface_endpoint.clone(),
            modelscope_endpoint: self.configured_modelscope_endpoint.clone(),
            http_proxy: self.configured_http_proxy.clone(),
            https_proxy: self.configured_https_proxy.clone(),
            no_proxy: self.configured_no_proxy.clone(),
            ca_bundle: self.configured_ca_bundle.clone(),
        }
    }

    pub fn search_models(
        context: &DownloadContext,
        query: &str,
        source: &str,
    ) -> Result<Vec<ModelSearchResult>, String> {
        let query = query.trim();
        if query.chars().count() < 2 {
            return Ok(Vec::new());
        }

        let (url, query_params): (String, Vec<(&str, &str)>) = match source {
            "huggingface" => (
                huggingface_model_search_url(context),
                huggingface_model_search_params(query),
            ),
            "modelscope" => (
                modelscope_model_search_url(context),
                modelscope_model_search_params(query),
            ),
            _ => return Err(format!("unsupported model hub: {source}")),
        };

        let client = build_download_client(context)?;
        let response = client
            .get(url)
            .timeout(Duration::from_secs(20))
            .query(&query_params)
            .send()
            .map_err(|error| format!("model search request failed: {error:#}"))?;
        if !response.status().is_success() {
            let status = response.status();
            let body = response
                .text()
                .unwrap_or_else(|error| format!("<could not read response body: {error}>"));
            return Err(format!(
                "model search request failed: HTTP {status}: {body}"
            ));
        }
        let payload: Value = response
            .json()
            .map_err(|error| format!("invalid model search response: {error}"))?;
        let models = if source == "huggingface" {
            payload.as_array()
        } else {
            payload
                .get("data")
                .and_then(|data| data.get("models"))
                .and_then(Value::as_array)
        }
        .ok_or_else(|| "model search returned an unexpected response".to_string())?;

        let results: Vec<_> = models
            .iter()
            .filter_map(|model| {
                let repo_id = model
                    .get("id")
                    .or_else(|| model.get("Path"))?
                    .as_str()?
                    .to_string();
                let safetensors = model.get("safetensors");
                Some(ModelSearchResult {
                    repo_id,
                    downloads: model
                        .get("downloads")
                        .or_else(|| model.get("Downloads"))
                        .and_then(Value::as_u64),
                    likes: model
                        .get("likes")
                        .or_else(|| model.get("Likes"))
                        .and_then(Value::as_u64),
                    parameters: safetensors
                        .and_then(|value| value.get("total"))
                        .and_then(Value::as_u64),
                    size_bytes: model
                        .get("usedStorage")
                        .or_else(|| model.get("used_storage"))
                        .or_else(|| model.get("size_bytes"))
                        .or_else(|| model.get("sizeBytes"))
                        .or_else(|| model.get("ModelSize"))
                        .or_else(|| model.get("model_size"))
                        .or_else(|| model.get("size"))
                        .and_then(Value::as_u64),
                })
            })
            .collect();

        Ok(results)
    }

    pub fn populate_model_sizes(
        context: &DownloadContext,
        results: &mut [ModelSearchResult],
    ) -> Result<(), String> {
        let client = build_download_client(context)?;
        // Use the configured Hub endpoint and the same proxy/TLS configuration as
        // search. Bounded workers keep metadata lookups from flooding a mirror.
        for start in (0..results.len()).step_by(4) {
            let end = (start + 4).min(results.len());
            let sizes = std::thread::scope(|scope| {
                let handles: Vec<_> = (start..end)
                    .filter(|&index| results[index].size_bytes.is_none())
                    .map(|index| {
                        let client = client.clone();
                        let context = context.clone();
                        let repo_id = results[index].repo_id.clone();
                        (
                            index,
                            scope.spawn(move || huggingface_repo_size(&client, &context, &repo_id)),
                        )
                    })
                    .collect();
                handles
                    .into_iter()
                    .filter_map(|(index, handle)| {
                        handle.join().ok().flatten().map(|size| (index, size))
                    })
                    .collect::<Vec<_>>()
            });
            for (index, size) in sizes {
                results[index].size_bytes = Some(size);
            }
        }
        Ok(())
    }

    pub fn configure_download(
        &mut self,
        huggingface_endpoint: String,
        modelscope_endpoint: String,
        http_proxy: String,
        https_proxy: String,
        no_proxy: String,
        ca_bundle: String,
    ) -> Result<(), String> {
        let endpoint = huggingface_endpoint.trim().trim_end_matches('/');
        let parsed = endpoint
            .strip_prefix("https://")
            .or_else(|| endpoint.strip_prefix("http://"))
            .ok_or_else(|| "Hugging Face endpoint must use http:// or https://".to_string())?;
        let host = parsed.split(['/', '?', '#']).next().unwrap_or_default();
        if host.is_empty() || endpoint.chars().any(char::is_whitespace) {
            return Err("Enter a valid Hugging Face endpoint URL".into());
        }
        let modelscope_endpoint = modelscope_endpoint.trim().trim_end_matches('/');
        let parsed = modelscope_endpoint
            .strip_prefix("https://")
            .or_else(|| modelscope_endpoint.strip_prefix("http://"))
            .ok_or_else(|| "ModelScope endpoint must use http:// or https://".to_string())?;
        let host = parsed.split(['/', '?', '#']).next().unwrap_or_default();
        if host.is_empty() || modelscope_endpoint.chars().any(char::is_whitespace) {
            return Err("Enter a valid ModelScope endpoint URL".into());
        }
        for (label, proxy) in [("HTTP", http_proxy.trim()), ("HTTPS", https_proxy.trim())] {
            if proxy.is_empty() {
                continue;
            }
            let authority = proxy
                .strip_prefix("http://")
                .or_else(|| proxy.strip_prefix("https://"))
                .ok_or_else(|| format!("{label} proxy must use http:// or https://"))?
                .split(['/', '?', '#'])
                .next()
                .unwrap_or_default();
            if authority.is_empty()
                || authority.contains('@')
                || proxy.chars().any(char::is_whitespace)
            {
                return Err(format!(
                    "Enter a valid {label} proxy URL without embedded credentials"
                ));
            }
        }
        self.configured_huggingface_endpoint = endpoint.to_string();
        self.configured_modelscope_endpoint = modelscope_endpoint.to_string();
        self.configured_http_proxy = http_proxy.trim().to_string();
        self.configured_https_proxy = https_proxy.trim().to_string();
        self.configured_no_proxy = no_proxy.trim().to_string();
        self.configured_ca_bundle = ca_bundle.trim().to_string();
        Ok(())
    }

    pub fn configure_storage(&mut self, directories: Vec<PathBuf>) -> Result<(), String> {
        if directories.is_empty() {
            return Err("At least one model directory is required".into());
        }
        let mut normalized = Vec::new();
        for (index, directory) in directories.into_iter().enumerate() {
            let directory = if let Ok(value) = directory.strip_prefix("~/") {
                std::env::var_os("HOME")
                    .map(PathBuf::from)
                    .unwrap_or_default()
                    .join(value)
            } else {
                directory
            };
            let path = if directory.is_absolute() {
                directory
            } else {
                std::env::current_dir()
                    .map_err(|error| error.to_string())?
                    .join(directory)
            };
            if is_cache_path(&path) {
                return Err(format!(
                    "model directories inside .cache are not supported: {}",
                    path.display()
                ));
            }
            if index == 0 {
                fs::create_dir_all(&path).map_err(|error| {
                    format!(
                        "could not create managed model directory {}: {error}",
                        path.display()
                    )
                })?;
            } else {
                validate_external_model_directory(&path)?;
            }
            let canonical = path.canonicalize().map_err(|error| error.to_string())?;
            if !normalized.contains(&canonical) {
                normalized.push(canonical);
            }
        }
        self.model_directories = normalized;
        Ok(())
    }

    pub fn load_model_with_options(
        &mut self,
        path: &str,
        model_options: Option<&ModelInferenceOptions>,
    ) -> Result<(), String> {
        if !is_model_dir(Path::new(path)) {
            return Err(format!(
                "model is incomplete or not loadable: {}",
                Path::new(path).display()
            ));
        }
        if !self.is_listed_model_path(path) {
            return Err(format!(
                "model is not available in the MiniMLX model list: {}",
                Path::new(path).display()
            ));
        }
        let execution_path = self.execution_model_path(path, model_options, None, false)?;
        // A failed health check does not mean the child process is gone. Stop
        // any tracked API child before switching to the in-process worker.
        if self.api_process.is_some() {
            self.stop_api();
        }
        if self.last_worker_model_options.as_ref() != model_options {
            self.worker.take();
            self.loaded_model = None;
            self.last_worker_model_options = None;
        }
        if let Some(worker) = self.worker.as_ref() {
            return match worker
                .request(&Request::Load {
                    model_dir: execution_path.clone(),
                })
                .map_err(|e| e.to_string())?
                .payload
            {
                ResponsePayload::Load { loaded: true, .. } => {
                    self.loaded_model = Some(path.to_string());
                    Ok(())
                }
                ResponsePayload::Error { error, .. } => Err(error),
                response => Err(format!("unexpected worker response: {response:?}")),
            };
        }
        let mtp_draft = model_options
            .and_then(|options| options.mtp_draft_model.as_deref())
            .unwrap_or("");
        let dflash_draft = model_options
            .and_then(|options| options.dflash_draft_model.as_deref())
            .unwrap_or("");
        let expert_cache = model_options
            .and_then(|o| o.expert_cache_gb)
            .map(|v| v.to_string())
            .unwrap_or_default();
        let force_text = if mtp_draft.is_empty() && dflash_draft.is_empty() {
            "0"
        } else {
            "1"
        };
        self.worker = Some(
            PythonWorker::start_with_env(
                &self.python,
                &self.root,
                &execution_path,
                &self.socket_path,
                &[
                    ("MINIMLX_MTP_DRAFT_MODEL", mtp_draft),
                    ("MINIMLX_DFLASH_DRAFT_MODEL", dflash_draft),
                    ("MINIMLX_FORCE_TEXT_RUNTIME", force_text),
                    ("MINIMLX_EXPERT_CACHE_GB", &expert_cache),
                ],
            )
            .map_err(|e| e.to_string())?,
        );
        self.last_worker_model_options = model_options.cloned();
        self.loaded_model = Some(path.to_string());
        Ok(())
    }

    pub fn unload_model(&mut self) -> Result<(), String> {
        // Stop a tracked API process even if its health endpoint has already
        // gone away; otherwise the stale `loaded_model` can survive unload.
        if self.api_process.is_some() {
            self.stop_api();
            if self.worker.is_none() {
                return Ok(());
            }
        }
        let Some(worker) = self.worker.as_ref() else {
            return Ok(());
        };
        match worker
            .request(&Request::Unload)
            .map_err(|e| e.to_string())?
            .payload
        {
            ResponsePayload::Unload { released: true, .. } => {
                self.loaded_model = None;
                Ok(())
            }
            ResponsePayload::Error { error, .. } => Err(error),
            response => Err(format!("unexpected worker response: {response:?}")),
        }
    }

    pub fn run_context_benchmark(
        &mut self,
        model_path: &str,
        model_options: Option<&ModelInferenceOptions>,
        benchmark_options: &ContextBenchmarkOptions,
    ) -> Result<String, String> {
        let max_context = benchmark_options
            .contexts
            .iter()
            .copied()
            .max()
            .unwrap_or(0) as u32;
        let execution_path =
            self.execution_model_path(model_path, model_options, Some(max_context), false)?;
        let restore_api = self.api_running();
        let restore_model = self.loaded_model.clone();
        let restore_worker_options = self.last_worker_model_options.clone();
        if restore_api {
            self.stop_api();
        }
        self.worker.take();
        self.loaded_model = None;
        let result =
            self.run_context_benchmark_isolated(&execution_path, model_options, benchmark_options);
        let restore_result = if restore_api {
            restore_model
                .as_deref()
                .map(|path| self.start_api(path))
                .unwrap_or(Ok(()))
        } else if let Some(path) = restore_model.as_deref() {
            self.load_model_with_options(path, restore_worker_options.as_ref())
        } else {
            Ok(())
        };
        match (result, restore_result) {
            (Ok(report), Ok(())) => Ok(report),
            (Err(error), Ok(())) => Err(error),
            (Ok(_), Err(error)) => Err(format!(
                "benchmark completed but service could not be restored: {error}"
            )),
            (Err(error), Err(restore)) => Err(format!(
                "{error}\nAdditionally, previous service could not be restored: {restore}"
            )),
        }
    }

    fn run_context_benchmark_isolated(
        &self,
        model_path: &str,
        model_options: Option<&ModelInferenceOptions>,
        benchmark_options: &ContextBenchmarkOptions,
    ) -> Result<String, String> {
        if benchmark_options.contexts.is_empty()
            || benchmark_options.contexts.contains(&0)
            || benchmark_options.warmup == 0
            || benchmark_options.repeats == 0
            || benchmark_options.max_tokens == 0
            || benchmark_options.prefill_step == 0
        {
            return Err("请选择上下文长度，并确保测试参数均大于 0".into());
        }
        let corpus_file = match benchmark_options.corpus_profile.as_str() {
            "code_python" => "code_python.txt",
            "code_mixed" => "code_mixed.txt",
            "novel_ko" => "novel_ko.txt",
            "novel_en" => "novel_en.txt",
            "novel_ja" => "novel_ja.txt",
            _ => return Err("请选择有效的测试语料".into()),
        };
        let prompt = self
            .root
            .join("Resources/BenchmarkCorpus")
            .join(corpus_file);
        let mtp_draft_model = model_options.and_then(|options| options.mtp_draft_model.as_deref());
        let dflash_draft_model =
            model_options.and_then(|options| options.dflash_draft_model.as_deref());
        let mtp_report = mtp_draft_model
            .map(|path| {
                Path::new(path)
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or(path)
                    .to_string()
            })
            .unwrap_or_else(|| "disabled".into());
        let dflash_report = dflash_draft_model
            .map(|path| {
                Path::new(path)
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or(path)
                    .to_string()
            })
            .unwrap_or_else(|| "disabled".into());
        let mut command = Command::new(&self.python);
        command
            .current_dir(&self.root)
            .env("PYTHONPATH", self.root.join("python"))
            .env(
                "MINIMLX_EXPERT_CACHE_GB",
                model_options
                    .and_then(|o| o.expert_cache_gb)
                    .map(|v| v.to_string())
                    .unwrap_or_default(),
            )
            .env("MINIMLX_SUPERVISOR_PID", std::process::id().to_string())
            .arg(self.root.join("tools/benchmark_context_matrix.py"))
            .arg(model_path)
            .arg(&prompt)
            .args(["--warmup", &benchmark_options.warmup.to_string()])
            .args(["--repeats", &benchmark_options.repeats.to_string()])
            .args(["--max-tokens", &benchmark_options.max_tokens.to_string()])
            .args(["--step", &benchmark_options.prefill_step.to_string()])
            .arg("--contexts")
            .args(benchmark_options.contexts.iter().map(usize::to_string))
            .arg("--text-only")
            .arg("--omlx-aligned-prompt")
            .args(["--context-profile", &benchmark_options.corpus_profile])
            .args(benchmark_options.ignore_eos.then_some("--ignore-eos"))
            .args(
                mtp_draft_model
                    .into_iter()
                    .flat_map(|path| ["--mtp-draft-model", path]),
            )
            .args(
                dflash_draft_model
                    .into_iter()
                    .flat_map(|path| ["--dflash-draft-model", path]),
            );
        let output = command
            .output()
            .map_err(|e| format!("could not start MiniMLX benchmark: {e}"))?;
        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr);
        if !output.status.success() {
            return Err(format!("{stderr}\n{stdout}"));
        }
        let mtp_report = if mtp_draft_model.is_none() && stdout.contains("decode backend (MTP)") {
            "embedded".to_string()
        } else {
            mtp_report
        };
        let reports = self.report_directory();
        fs::create_dir_all(&reports).map_err(|e| e.to_string())?;
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let report = reports.join(format!("minimlx-context-{stamp}.md"));
        let generated_at = Command::new("date")
            .args(["-u", "+%Y-%m-%dT%H:%M:%SZ"])
            .output()
            .ok()
            .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
            .unwrap_or_default();
        let os_version = Command::new("sw_vers")
            .args(["-productVersion"])
            .output()
            .ok()
            .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
            .unwrap_or_else(|| "unknown".into());
        let chip = Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
            .ok()
            .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
            .unwrap_or_else(|| "Apple Silicon".into());
        let revision = Command::new("git")
            .current_dir(&self.root)
            .args(["rev-parse", "--short", "HEAD"])
            .output()
            .ok()
            .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
            .unwrap_or_else(|| "unknown".into());
        let changes = Command::new("git")
            .current_dir(&self.root)
            .args(["diff", "--quiet"])
            .status()
            .map(|status| {
                if status.success() {
                    "clean"
                } else {
                    "modified"
                }
            })
            .unwrap_or("unknown");
        let model_config = fs::read_to_string(Path::new(model_path).join("config.json"))
            .ok()
            .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok());
        let model_type = model_config
            .as_ref()
            .and_then(|value| value.get("model_type"))
            .and_then(serde_json::Value::as_str)
            .unwrap_or("unknown");
        let architectures = model_config
            .as_ref()
            .and_then(|value| value.get("architectures"))
            .and_then(serde_json::Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .filter_map(serde_json::Value::as_str)
                    .collect::<Vec<_>>()
                    .join(", ")
            })
            .unwrap_or_else(|| "unknown".into());
        let quantization = model_config
            .as_ref()
            .and_then(|value| value.get("quantization_config"))
            .map(|value| value.to_string())
            .unwrap_or_else(|| "not declared in config.json".into());
        let prompt_hash = Command::new("shasum")
            .args(["-a", "256"])
            .arg(&prompt)
            .output()
            .ok()
            .map(|output| {
                String::from_utf8_lossy(&output.stdout)
                    .split_whitespace()
                    .next()
                    .unwrap_or("unknown")
                    .to_string()
            })
            .unwrap_or_else(|| "unknown".into());
        let versions = Command::new(&self.python)
            .args(["-c", "import importlib.metadata as m; names={d.metadata['Name'].lower() for d in m.distributions()}; print(', '.join(f'{n}={m.version(n)}' for n in ('mlx', 'mlx-lm') if n in names))"])
            .output().ok().map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
            .unwrap_or_else(|| "unavailable".into());
        let memory_total = self.system_memory().1;
        fs::write(
            &report,
            format!(
                "# MiniMLX Context Matrix\n\n- Generated (UTC): {generated_at}\n- Model: `{model_path}`\n- Model type: `{model_type}`\n- Architecture: `{architectures}`\n- Quantization metadata: `{quantization}`\n- Model disk size: {}\n- Hardware: {chip}\n- Unified memory: {}\n- macOS: {os_version}\n- Python packages: {versions}\n- Source revision: {revision} ({changes} working tree)\n- Benchmark corpus profile: `{}`\n- Benchmark corpus file: `{}`\n- Corpus SHA-256: `{prompt_hash}`\n- MTP draft: `{mtp_report}`\n- DFlash draft: `{dflash_report}`\n- Warm-up: {} tokens at 2K context\n- Measurement: {} run(s) per context; median and raw samples reported\n- Generation limit: {} tokens\n- Prefill step: {} tokens\n- Contexts: {:?}\n- Prompt protocol: oMLX-aligned corpus completion\n- Ignore EOS: {}\n- Runtime: MiniMLX only\n\n## Raw benchmark output\n\n```text\n{stdout}\n```\n",
                format_bytes(directory_size(Path::new(model_path))),
                format_bytes(memory_total),
                benchmark_options.corpus_profile,
                prompt.display(),
                benchmark_options.warmup,
                benchmark_options.repeats,
                benchmark_options.max_tokens,
                benchmark_options.prefill_step,
                benchmark_options.contexts,
                benchmark_options.ignore_eos,
            ),
        )
        .map_err(|e| e.to_string())?;
        Ok(format!("{}\n\n{}", report.display(), stdout))
    }

    pub fn download_model_with_progress(
        context: &DownloadContext,
        repo_id: &str,
        source: &str,
        mut on_progress: impl FnMut(u64, Option<u64>) -> bool,
    ) -> Result<Option<String>, String> {
        let repo_id = validate_repo_id(repo_id)?;
        if !matches!(source, "huggingface" | "modelscope") {
            return Err(format!("unsupported model hub: {source}"));
        }
        let _sleep_guard = DownloadSleepGuard::acquire()?;
        let destination = context.download_directory.join(repo_id.replace('/', "--"));
        fs::create_dir_all(&destination).map_err(|error| {
            format!(
                "could not create model download directory {}: {error}",
                destination.display()
            )
        })?;
        fs::write(destination.join(".minimlx-download-source"), source)
            .map_err(|error| format!("could not save download source: {error}"))?;
        fs::write(destination.join(".minimlx-repo-id"), repo_id)
            .map_err(|error| format!("could not save model repository ID: {error}"))?;
        let client = build_download_client(context)?;
        let files = if source == "huggingface" {
            list_huggingface_files(&client, context, repo_id)?
        } else {
            list_modelscope_files(&client, context, repo_id)?
        };
        if files.is_empty() {
            return Err("model repository contains no downloadable files".into());
        }
        save_download_manifest(&destination, &files)?;
        let total_size = files.iter().try_fold(0u64, |total, file| {
            file.size.and_then(|size| total.checked_add(size))
        });
        let downloaded = downloaded_bytes_for_files(&destination, &files);
        if !on_progress(downloaded, total_size) {
            return Ok(None);
        }
        for file in &files {
            let url = if source == "huggingface" {
                huggingface_file_url(context, repo_id, &file.revision, &file.path)?
            } else {
                modelscope_file_url(context, repo_id, &file.revision, &file.path)?
            };
            let downloaded = download_one_file(&client, &url, &destination, &file, &mut |_| {
                on_progress(downloaded_bytes_for_files(&destination, &files), total_size)
            })?;
            if !downloaded {
                return Ok(None);
            }
        }
        let _ = fs::remove_file(destination.join(".minimlx-download-source"));
        let _ = fs::remove_file(download_manifest_path(&destination));
        Ok(Some(destination.display().to_string()))
    }

    pub fn system_memory(&self) -> (u64, u64) {
        let total = Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .and_then(|v| v.trim().parse::<u64>().ok())
            .filter(|total| *total > 0)
            .unwrap_or_else(system_physical_memory_bytes);
        let vm = Command::new("vm_stat")
            .output()
            .ok()
            .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
            .unwrap_or_default();
        let page_size = vm
            .lines()
            .next()
            .and_then(|line| line.split("page size of ").nth(1))
            .and_then(|tail| tail.split_whitespace().next())
            .and_then(|word| word.parse::<u64>().ok())
            .unwrap_or(4096);
        let used_pages: u64 = [
            "Pages active:",
            "Pages wired down:",
            "Pages occupied by compressor:",
        ]
        .iter()
        .filter_map(|key| vm.lines().find(|line| line.starts_with(key)))
        .filter_map(|line| {
            line.split_whitespace()
                .last()?
                .trim_end_matches('.')
                .parse::<u64>()
                .ok()
        })
        .sum();
        (used_pages * page_size, total)
    }

    fn ensure_memory_budget(
        &self,
        model_path: &str,
        model_options: Option<&ModelInferenceOptions>,
        benchmark_context_tokens: Option<u32>,
        replaces_loaded_model: bool,
    ) -> Result<(), String> {
        let (used, total) = self.system_memory();
        if total == 0 {
            return Ok(());
        }

        let estimate = |path: &str| match benchmark_context_tokens {
            Some(context_tokens) => estimate_model_runtime_memory(path, context_tokens),
            None => estimate_model_weight_memory(path),
        };
        let paged = Path::new(model_path).join("offload_index.json").is_file();
        let mut required = estimate(model_path);
        let mut paging_working_memory = 0;
        if paged {
            let default_cache = fs::read_to_string(Path::new(model_path).join("config.json"))
                .ok().and_then(|s| serde_json::from_str::<Value>(&s).ok())
                .and_then(|v| v.get("model_type").and_then(Value::as_str).map(str::to_owned))
                .map_or(12.0, |kind| if kind == "qwen3_5_moe" { 2.0 } else { 12.0 });
            let cache_gb = model_options
                .and_then(|options| options.expert_cache_gb)
                .unwrap_or(default_cache);
            paging_working_memory = (cache_gb * 1_000_000_000.0) as u64;
            paging_working_memory = paging_working_memory
                .saturating_add(required.saturating_sub(estimate_model_weight_memory(model_path)));
            required = required.saturating_add((cache_gb * 1_000_000_000.0) as u64);
        }
        if let Some(options) = model_options {
            for draft_path in [
                options.mtp_draft_model.as_deref(),
                options.dflash_draft_model.as_deref(),
            ]
            .into_iter()
            .flatten()
            {
                required = required.saturating_add(estimate(draft_path));
            }
        }

        // API startup stops the current worker/API before loading the target.
        // Credit that model's resident weights now so admission reflects the
        // post-switch state, not the temporary overlap before teardown.
        let loaded_model_to_release = replaces_loaded_model
            .then_some(self.loaded_model.as_deref())
            .flatten();
        if let Some(loaded_model) = loaded_model_to_release {
            let previous_options = if self.api_process.is_some() {
                self.last_api_model_options.as_ref()
            } else {
                self.last_worker_model_options.as_ref()
            };
            let mut releasable = estimate_model_weight_memory(loaded_model);
            if let Some(options) = previous_options {
                for draft_path in [
                    options.mtp_draft_model.as_deref(),
                    options.dflash_draft_model.as_deref(),
                ]
                .into_iter()
                .flatten()
                {
                    releasable =
                        releasable.saturating_add(estimate_model_weight_memory(draft_path));
                }
            }
            required = required.saturating_sub(releasable);
        }

        let available = total.saturating_sub(used);
        // Model-load admission estimates resident weights only. KV cache grows
        // with actual requests and is checked by API admission; context matrix
        // benchmarks pass a context and reserve their full projected peak.
        // Keep the oMLX aggressive-tier reserve (2%, 1.5–4 GiB) on top.
        let headroom = memory_headroom(total);
        let needed = required.saturating_add(headroom);
        // File-backed paged weights can be reclaimed and faulted back in by
        // macOS. Requiring their entire estimated size to be free at once
        // defeats paging. Still require enough total RAM for the conservative
        // estimate and enough currently available RAM for cache and KV growth.
        let paging_can_proceed = paged
            && total >= needed
            && available >= paging_working_memory.saturating_add(headroom)
            && model_options.is_none_or(|options| {
                options.mtp_draft_model.is_none() && options.dflash_draft_model.is_none()
            });
        if available < needed && !paging_can_proceed {
            return Err(format!(
                "Model load blocked to protect unified memory: estimated additional use is {}, only {} is currently available, and {} is reserved for macOS. Choose a smaller model or context, or close other apps first.",
                format_bytes(required),
                format_bytes(available),
                format_bytes(headroom),
            ));
        }
        Ok(())
    }

    pub fn start_api(&mut self, model_path: &str) -> Result<(), String> {
        let options = self
            .last_api_model_path
            .as_deref()
            .filter(|last_path| *last_path == model_path)
            .and_then(|_| self.last_api_model_options.clone());
        self.start_api_with_options(model_path, options.as_ref())
    }

    pub fn start_api_with_options(
        &mut self,
        model_path: &str,
        model_options: Option<&ModelInferenceOptions>,
    ) -> Result<(), String> {
        let already_running = self.api_running();
        if !self.is_listed_model_path(model_path) {
            if !already_running {
                let _ = ensure_api_port_available(self.configured_api_port, &self.root);
            }
            return Err("Only models in the MiniMLX model list can be started".into());
        }
        if already_running {
            return Ok(());
        }
        let execution_path = self.execution_model_path(model_path, model_options, None, true)?;
        if self.configured_auth_required
            && self.configured_api_key.as_deref().is_none_or(str::is_empty)
        {
            return Err(
                "API-key authentication is enabled but no key is available in macOS Keychain"
                    .into(),
            );
        }
        self.stop_api();
        if self.worker.is_some() {
            self.unload_model()?;
            self.worker.take();
            self.loaded_model = None;
        }
        let port = self.configured_api_port;
        ensure_api_port_available(port, &self.root)?;
        let max_tokens = model_options
            .and_then(|options| options.max_tokens)
            .unwrap_or(self.configured_api_max_tokens);
        let max_context = model_options
            .and_then(|options| options.max_context_tokens)
            .unwrap_or(self.configured_max_context);
        let temperature = model_options
            .and_then(|options| options.temperature)
            .unwrap_or(self.configured_temperature);
        let top_p = model_options
            .and_then(|options| options.top_p)
            .unwrap_or(self.configured_top_p);
        let top_k = model_options
            .and_then(|options| options.top_k)
            .unwrap_or(self.configured_top_k);
        let min_p = model_options
            .and_then(|options| options.min_p)
            .unwrap_or(0.0);
        let repetition_penalty = model_options
            .and_then(|options| options.repetition_penalty)
            .unwrap_or(self.configured_repetition_penalty);
        let presence_penalty = model_options
            .and_then(|options| options.presence_penalty)
            .unwrap_or(0.0);
        let frequency_penalty = model_options
            .and_then(|options| options.frequency_penalty)
            .unwrap_or(0.0);
        let force_sampling = model_options
            .and_then(|options| options.force_sampling)
            .unwrap_or(false);
        let chat_template_kwargs = model_options
            .and_then(|options| options.chat_template_kwargs.as_ref())
            .map(serde_json::to_string)
            .transpose()
            .map_err(|error| format!("could not serialize chat template settings: {error}"))?;
        let mtp_draft_model = model_options.and_then(|options| options.mtp_draft_model.as_deref());
        let dflash_draft_model =
            model_options.and_then(|options| options.dflash_draft_model.as_deref());
        let run_log_path = self
            .configured_history_db
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join("logs/server.log");
        let mut child = Command::new(&self.python)
            .current_dir(&self.root)
            .env("PYTHONPATH", self.root.join("python"))
            .env(
                "MINIMLX_EXPERT_CACHE_GB",
                model_options
                    .and_then(|o| o.expert_cache_gb)
                    .map(|v| v.to_string())
                    .unwrap_or_default(),
            )
            .env("MINIMLX_SUPERVISOR_PID", std::process::id().to_string())
            .env_remove("MINIMLX_MTP_DRAFT_MODEL")
            .env_remove("MINIMLX_DFLASH_DRAFT_MODEL")
            .arg("-m")
            .arg("minimlx_runtime")
            .arg("api")
            .arg(&execution_path)
            .args([
                "--host",
                &self.configured_host,
                "--port",
                &port.to_string(),
                "--max-tokens",
                &max_tokens.to_string(),
                "--max-context-tokens",
                &max_context.to_string(),
                "--temperature",
                &temperature.to_string(),
                "--top-p",
                &top_p.to_string(),
                "--top-k",
                &top_k.to_string(),
                "--min-p",
                &min_p.to_string(),
                "--repetition-penalty",
                &repetition_penalty.to_string(),
                "--presence-penalty",
                &presence_penalty.to_string(),
                "--frequency-penalty",
                &frequency_penalty.to_string(),
                "--log-level",
                &self.configured_log_level,
                "--log-file",
                &run_log_path.to_string_lossy(),
                "--history-db",
                &self.configured_history_db.to_string_lossy(),
            ])
            .args(
                model_options
                    .and_then(|options| options.alias.as_deref())
                    .into_iter()
                    .flat_map(|alias| ["--model-alias", alias]),
            )
            .args(if self.configured_history {
                vec!["--record-history"]
            } else {
                vec!["--no-record-history"]
            })
            .args(force_sampling.then_some("--force-sampling"))
            .args(
                chat_template_kwargs
                    .as_deref()
                    .into_iter()
                    .flat_map(|value| ["--chat-template-kwargs", value]),
            )
            .envs(
                mtp_draft_model
                    .into_iter()
                    .map(|path| ("MINIMLX_MTP_DRAFT_MODEL", path)),
            )
            .envs(
                dflash_draft_model
                    .into_iter()
                    .map(|path| ("MINIMLX_DFLASH_DRAFT_MODEL", path)),
            )
            .envs(
                (!mtp_draft_model.is_none() || !dflash_draft_model.is_none())
                    .then_some(("MINIMLX_FORCE_TEXT_RUNTIME", "1")),
            )
            .env(
                "MINIMLX_API_KEY",
                self.configured_api_key.as_deref().unwrap_or(""),
            )
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|error| format!("could not start API server: {error}"))?;
        let deadline = Instant::now() + Duration::from_secs(120);
        loop {
            if api_healthcheck(port) {
                break;
            }
            if let Some(status) = child.try_wait().map_err(|error| error.to_string())? {
                return Err(format!(
                    "API server exited during startup ({status}); inspect {}",
                    run_log_path.display()
                ));
            }
            if Instant::now() >= deadline {
                let _ = child.kill();
                let _ = child.wait();
                return Err("API server did not become ready within 120 seconds".into());
            }
            std::thread::sleep(Duration::from_millis(250));
        }
        self.api_process = Some(child);
        self.api_port = Some(port);
        self.loaded_model = Some(model_path.to_string());
        self.last_api_model_path = Some(model_path.to_string());
        self.last_api_model_options = model_options.cloned();
        Ok(())
    }

    pub fn configure_api(
        &mut self,
        host: String,
        port: u16,
        max_tokens: u32,
        max_context: u32,
        temperature: f32,
        top_p: f32,
        top_k: u32,
        repetition_penalty: f32,
        log_level: String,
        history: bool,
        history_db: PathBuf,
        auth_required: bool,
        api_key: Option<String>,
    ) -> Result<(), String> {
        let loopback_host = matches!(host.as_str(), "127.0.0.1" | "localhost" | "::1");
        self.configured_auth_required = auth_required || !loopback_host;
        if !(1024..=65535).contains(&port) {
            return Err("Choose an unprivileged port between 1024 and 65535".into());
        }
        if !(1..=1_000_000).contains(&max_tokens) {
            return Err("Default maximum tokens must be between 1 and 1,000,000".into());
        }
        if !(1..=1_000_000).contains(&max_context) {
            return Err("Default context window must be between 1 and 1,000,000".into());
        }
        if !temperature.is_finite() || temperature < 0.0 {
            return Err("Temperature must be a finite, non-negative number".into());
        }
        if !top_p.is_finite() || !(0.0..=1.0).contains(&top_p) {
            return Err("Top P must be between 0 and 1".into());
        }
        if !repetition_penalty.is_finite() || repetition_penalty <= 0.0 {
            return Err("Repetition penalty must be greater than 0".into());
        }
        if !loopback_host && api_key.as_deref().is_none_or(str::is_empty) {
            return Err("LAN binding requires an API key".into());
        }
        if auth_required && api_key.as_deref().is_none_or(str::is_empty) {
            return Err("API-key authentication requires an API key".into());
        }
        if !["DEBUG", "INFO", "WARNING", "ERROR"].contains(&log_level.as_str()) {
            return Err("Unsupported log level".into());
        }
        self.configured_host = host;
        self.configured_api_port = port;
        self.configured_api_max_tokens = max_tokens;
        self.configured_max_context = max_context;
        self.configured_temperature = temperature;
        self.configured_top_p = top_p;
        self.configured_top_k = top_k;
        self.configured_repetition_penalty = repetition_penalty;
        self.configured_log_level = log_level;
        self.configured_history = history;
        self.configured_history_db = history_db;
        self.configured_api_key = api_key;
        Ok(())
    }

    pub fn api_running(&mut self) -> bool {
        let child_is_alive = self
            .api_process
            .as_mut()
            .is_some_and(|child| matches!(child.try_wait(), Ok(None)));
        child_is_alive
            || self
                .api_port
                .is_some_and(|port| minimlx_api_listener_alive(port, &self.root))
    }

    pub fn api_metrics(&mut self) -> Option<Value> {
        if !self.api_running() {
            return None;
        }
        http_get_json(self.api_port?, "/telemetry")
    }

    pub fn sync_state(&mut self) {
        let exited = self
            .api_process
            .as_mut()
            .is_some_and(|child| !matches!(child.try_wait(), Ok(None)));
        if exited {
            self.api_process.take();
            let listener_alive = self
                .api_port
                .is_some_and(|port| minimlx_api_listener_alive(port, &self.root));
            if !listener_alive {
                self.api_port = None;
                if self.worker.is_none() {
                    self.loaded_model = None;
                }
            }
        }
    }

    pub fn stop_api(&mut self) {
        let port = self.api_port.take();
        if let Some(mut child) = self.api_process.take() {
            if matches!(child.try_wait(), Ok(None)) {
                let _ = child.kill();
            }
            let _ = child.wait();
        }
        if let Some(port) = port {
            // A runtime can outlive its original Child handle (for example,
            // after the desktop process is replaced during development).
            // Only stop a listener that is verified as this MiniMLX runtime.
            let _ = ensure_api_port_available(port, &self.root);
        }
        self.api_port = None;
        if self.worker.is_none() {
            self.loaded_model = None;
        }
    }
}

pub fn validate_external_model_directory(path: &Path) -> Result<PathBuf, String> {
    if is_cache_path(path) {
        return Err(".cache directories are not scanned".into());
    }
    let canonical = path
        .canonicalize()
        .map_err(|error| format!("cannot access model directory {}: {error}", path.display()))?;
    if !canonical.is_dir() {
        return Err(format!("not a directory: {}", canonical.display()));
    }
    let contains_supported_model =
        !discover_model_directories(std::slice::from_ref(&canonical)).is_empty();
    if !contains_supported_model {
        return Err(format!(
            "no supported MiniMLX model found in {} or its immediate subfolders",
            canonical.display()
        ));
    }
    Ok(canonical)
}

fn discover_model_directories(configured_roots: &[PathBuf]) -> Vec<PathBuf> {
    let mut candidates = configured_roots.to_vec();
    let mut frontier = configured_roots.to_vec();
    // Support both <library>/<model> and <library>/<publisher>/<model>.
    for _ in 0..2 {
        let mut next = Vec::new();
        for directory in frontier {
            if is_cache_path(&directory) {
                continue;
            }
            if let Ok(entries) = fs::read_dir(directory) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_dir()
                        && !is_cache_path(&path)
                        && !path.file_name().is_some_and(|n| {
                            n.to_string_lossy().starts_with(".minimlx-ssd")
                                || n.to_string_lossy().starts_with(".minimlx-mtp")
                        })
                    {
                        candidates.push(path.clone());
                        next.push(path);
                    }
                }
            }
        }
        frontier = next;
    }
    candidates.sort();
    candidates.dedup();
    candidates
        .into_iter()
        .filter(|path| is_managed_model_dir(path))
        .collect()
}

/// A downloaded speculative checkpoint is a managed model-library artifact,
/// even when it intentionally lacks the tokenizer needed to run by itself.
fn is_managed_model_dir(path: &Path) -> bool {
    is_model_dir(path) || is_speculative_draft_dir(path)
}

fn is_speculative_draft_dir(path: &Path) -> bool {
    if is_cache_path(path) || !path.is_dir() {
        return false;
    }
    let Ok(config) = fs::read_to_string(path.join("config.json")) else {
        return false;
    };
    let Ok(config) = serde_json::from_str::<Value>(&config) else {
        return false;
    };
    let model_facts = format!(
        "{} {}",
        config
            .get("model_type")
            .and_then(Value::as_str)
            .unwrap_or_default(),
        config
            .get("architectures")
            .map(Value::to_string)
            .unwrap_or_default()
    )
    .to_ascii_lowercase();
    let is_draft = model_facts.contains("qwen3_5_mtp")
        || model_facts.contains("dflash") && model_facts.contains("draft");
    is_draft && complete_weight_files(path)
}

fn is_cache_path(path: &Path) -> bool {
    path.components()
        .any(|component| component.as_os_str() == ".cache")
}

fn same_filesystem_path(left: &str, right: &str) -> bool {
    left == right
        || Path::new(left).canonicalize().ok().is_some_and(|left| {
            Path::new(right)
                .canonicalize()
                .ok()
                .is_some_and(|right| left == right)
        })
}

fn ensure_api_port_available(port: u16, app_root: &Path) -> Result<(), String> {
    if TcpListener::bind(("127.0.0.1", port)).is_ok() {
        return Ok(());
    }

    let process_ids = listening_process_ids(port)?;
    if process_ids.is_empty()
        || process_ids
            .iter()
            .any(|pid| !is_minimlx_api_process(*pid, app_root))
    {
        return Err(format!(
            "MiniMLX API port {port} is already in use by another application"
        ));
    }

    for pid in &process_ids {
        let status = Command::new("/bin/kill")
            .args(["-TERM", &pid.to_string()])
            .status()
            .map_err(|error| format!("could not stop the previous MiniMLX API process: {error}"))?;
        if !status.success() {
            return Err(format!(
                "could not stop the previous MiniMLX API process (PID {pid})"
            ));
        }
    }

    for _ in 0..30 {
        if TcpListener::bind(("127.0.0.1", port)).is_ok() {
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(100));
    }

    let remaining = listening_process_ids(port)?;
    if remaining.is_empty()
        || remaining
            .iter()
            .any(|pid| !is_minimlx_api_process(*pid, app_root))
    {
        return Err(format!(
            "Port {port} is still occupied, but its current owner is not a verified MiniMLX API process"
        ));
    }
    for pid in &remaining {
        let status = Command::new("/bin/kill")
            .args(["-KILL", &pid.to_string()])
            .status()
            .map_err(|error| {
                format!("could not force-stop the previous MiniMLX API process: {error}")
            })?;
        if !status.success() {
            return Err(format!(
                "could not force-stop the previous MiniMLX API process (PID {pid})"
            ));
        }
    }
    for _ in 0..20 {
        if TcpListener::bind(("127.0.0.1", port)).is_ok() {
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(100));
    }

    Err(format!(
        "The previous MiniMLX API process did not release port {port}"
    ))
}

fn listening_process_ids(port: u16) -> Result<Vec<u32>, String> {
    let output = Command::new("lsof")
        .args(["-nP", "-t", &format!("-iTCP:{port}"), "-sTCP:LISTEN"])
        .output()
        .map_err(|error| {
            format!("could not identify the process using API port {port}: {error}")
        })?;
    if !output.status.success() && output.stdout.is_empty() {
        return Ok(Vec::new());
    }
    Ok(String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| line.trim().parse::<u32>().ok())
        .collect())
}

fn is_minimlx_api_process(pid: u32, app_root: &Path) -> bool {
    let command = Command::new("ps")
        .args(["-p", &pid.to_string(), "-o", "command="])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string());
    let Some(command) = command else {
        return false;
    };
    if !command.contains("-m minimlx_runtime api") {
        return false;
    }

    let cwd = Command::new("lsof")
        .args(["-a", "-p", &pid.to_string(), "-d", "cwd", "-Fn"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| {
            String::from_utf8_lossy(&output.stdout)
                .lines()
                .find_map(|line| line.strip_prefix('n').map(PathBuf::from))
        })
        .flatten();
    let (Some(cwd), Ok(app_root)) = (cwd, app_root.canonicalize()) else {
        return false;
    };
    cwd.canonicalize().is_ok_and(|cwd| cwd == app_root)
}

fn minimlx_api_listener_alive(port: u16, app_root: &Path) -> bool {
    let Ok(process_ids) = listening_process_ids(port) else {
        return false;
    };
    !process_ids.is_empty()
        && process_ids
            .iter()
            .all(|pid| is_minimlx_api_process(*pid, app_root))
}

fn api_healthcheck(port: u16) -> bool {
    http_get_json(port, "/health")
        .is_some_and(|response| response.get("ok").and_then(Value::as_bool) == Some(true))
}

fn http_get_json(port: u16, path: &str) -> Option<Value> {
    let Ok(mut stream) = TcpStream::connect_timeout(
        &format!("127.0.0.1:{port}")
            .parse()
            .expect("valid loopback address"),
        Duration::from_millis(500),
    ) else {
        return None;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_millis(500)));
    if stream
        .write_all(format!("GET {path} HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n").as_bytes())
        .is_err()
    {
        return None;
    }
    let mut response = Vec::new();
    if stream.read_to_end(&mut response).is_err() || !response.starts_with(b"HTTP/1.0 200") {
        return None;
    }
    let body = response
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|index| &response[index + 4..])?;
    serde_json::from_slice(body).ok()
}

impl Drop for Backend {
    fn drop(&mut self) {
        self.stop_api();
    }
}

fn is_model_dir(path: &Path) -> bool {
    if is_cache_path(path)
        || !path.is_dir()
        || !path.join("config.json").is_file()
        || !(path.join("tokenizer.json").is_file() || path.join("tokenizer_config.json").is_file())
    {
        return false;
    }

    // Hub caches also contain auxiliary OCR / speech checkpoints. They have
    // tokenizer/config files but are not loadable by MiniMLX's text runtime.
    let Ok(config) = fs::read_to_string(path.join("config.json")) else {
        return false;
    };
    let Ok(config) = serde_json::from_str::<Value>(&config) else {
        return false;
    };
    let has_model_identity = config
        .get("model_type")
        .and_then(Value::as_str)
        .is_some_and(|value| !value.is_empty())
        || config
            .get("architectures")
            .and_then(Value::as_array)
            .is_some_and(|values| !values.is_empty());
    let has_mlx_weights = complete_weight_files(path);
    if !has_model_identity || !has_mlx_weights {
        return false;
    }
    let model_facts = format!(
        "{} {}",
        config
            .get("model_type")
            .and_then(Value::as_str)
            .unwrap_or_default(),
        config
            .get("architectures")
            .map(Value::to_string)
            .unwrap_or_default()
    )
    .to_ascii_lowercase();
    !["paddleocr", "whisper", "speech", "asr"]
        .iter()
        .any(|unsupported| model_facts.contains(unsupported))
}

fn complete_weight_files(path: &Path) -> bool {
    let offload_index = path.join("offload_index.json");
    if offload_index.is_file() {
        let Some(manifest) = fs::read_to_string(path.join(".minimlx-ssd-manifest.json"))
            .ok()
            .and_then(|s| serde_json::from_str::<Value>(&s).ok())
        else {
            return false;
        };
        let Some(expected) = manifest.get("resident").and_then(Value::as_array) else {
            return false;
        };
        if expected.is_empty()
            || !expected.iter().all(|v| {
                v.as_str().is_some_and(|name| {
                    name.starts_with("resident-")
                        && name.ends_with(".safetensors")
                        && !name.contains('/')
                        && is_complete_safetensors(&path.join(name))
                })
            })
        {
            return false;
        }
        let Some(index) = fs::read_to_string(offload_index)
            .ok()
            .and_then(|s| serde_json::from_str::<Value>(&s).ok())
        else {
            return false;
        };
        let Some(layers) = index.get("layers").and_then(Value::as_array) else {
            return false;
        };
        if layers.is_empty()
            || index
                .get("num_experts")
                .and_then(Value::as_u64)
                .unwrap_or(0)
                == 0
        {
            return false;
        }
        if !layers.iter().all(|layer| {
            layer.as_u64().is_some_and(|n| {
                is_complete_safetensors(
                    &path
                        .join("experts")
                        .join(format!("layer_{n:04}.safetensors")),
                )
            })
        }) {
            return false;
        }
        let Ok(entries) = fs::read_dir(path) else {
            return false;
        };
        let resident: Vec<_> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|s| s.to_str())
                    .is_some_and(|n| n.starts_with("resident-") && n.ends_with(".safetensors"))
            })
            .collect();
        return !resident.is_empty() && resident.iter().all(|p| is_complete_safetensors(p));
    }
    let index_path = path.join("model.safetensors.index.json");
    if index_path.exists() {
        let Ok(index) = fs::read_to_string(index_path) else {
            return false;
        };
        let Ok(index) = serde_json::from_str::<Value>(&index) else {
            return false;
        };
        let Some(weight_map) = index.get("weight_map").and_then(Value::as_object) else {
            return false;
        };
        if weight_map.is_empty() || weight_map.values().any(|value| !value.is_string()) {
            return false;
        }
        let shards: std::collections::HashSet<_> =
            weight_map.values().filter_map(Value::as_str).collect();
        return shards.into_iter().all(|filename| {
            let relative = Path::new(filename);
            !relative.is_absolute()
                && relative
                    .components()
                    .all(|part| matches!(part, std::path::Component::Normal(_)))
                && is_complete_safetensors(&path.join(relative))
        });
    }

    let Ok(entries) = fs::read_dir(path) else {
        return false;
    };
    let files: Vec<_> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|file| {
            file.is_file()
                && file
                    .extension()
                    .is_some_and(|extension| extension == "safetensors" || extension == "npz")
        })
        .collect();
    !files.is_empty()
        && files.iter().all(|file| {
            if file
                .extension()
                .is_some_and(|extension| extension == "safetensors")
            {
                is_complete_safetensors(file)
            } else {
                is_complete_npz(file)
            }
        })
}

fn is_incomplete_download_dir(path: &Path) -> bool {
    let cache_downloads = path.join(".cache/huggingface/download");
    if contains_incomplete_file(&cache_downloads) {
        return true;
    }

    let index_path = path.join("model.safetensors.index.json");
    let Ok(index) = fs::read_to_string(index_path) else {
        return false;
    };
    let Ok(index) = serde_json::from_str::<Value>(&index) else {
        return false;
    };
    let Some(weight_map) = index.get("weight_map").and_then(Value::as_object) else {
        return false;
    };
    if weight_map.is_empty() || weight_map.values().any(|value| !value.is_string()) {
        return false;
    }
    let shards: std::collections::HashSet<_> =
        weight_map.values().filter_map(Value::as_str).collect();
    shards.into_iter().any(|filename| {
        let relative = Path::new(filename);
        relative.is_relative()
            && relative
                .components()
                .all(|part| matches!(part, std::path::Component::Normal(_)))
            && !is_complete_safetensors(&path.join(relative))
    })
}

fn contains_incomplete_file(directory: &Path) -> bool {
    let Ok(entries) = fs::read_dir(directory) else {
        return false;
    };
    entries.flatten().any(|entry| {
        let path = entry.path();
        if entry.file_type().is_ok_and(|kind| kind.is_dir()) {
            contains_incomplete_file(&path)
        } else {
            entry.file_name().to_string_lossy().ends_with(".incomplete")
        }
    })
}

fn is_complete_safetensors(path: &Path) -> bool {
    let Ok(metadata) = fs::metadata(path) else {
        return false;
    };
    let file_len = metadata.len();
    if !metadata.is_file() || file_len < 10 {
        return false;
    }
    let Ok(mut file) = fs::File::open(path) else {
        return false;
    };
    let mut prefix = [0_u8; 8];
    if file.read_exact(&mut prefix).is_err() {
        return false;
    }
    let header_len = u64::from_le_bytes(prefix);
    // Safetensors headers are small compared with their tensor data. Bound the
    // allocation so a malformed download cannot cause excessive memory use.
    if header_len == 0 || header_len > 100 * 1024 * 1024 {
        return false;
    }
    let Some(data_start) = 8_u64.checked_add(header_len) else {
        return false;
    };
    if data_start > file_len {
        return false;
    }
    let Ok(header_size) = usize::try_from(header_len) else {
        return false;
    };
    let mut header = vec![0_u8; header_size];
    if file.read_exact(&mut header).is_err() {
        return false;
    }
    let Ok(header) = serde_json::from_slice::<Value>(&header) else {
        return false;
    };
    let Some(object) = header.as_object() else {
        return false;
    };
    let mut tensor_count = 0_usize;
    let mut data_len = 0_u64;
    for (name, tensor) in object {
        if name == "__metadata__" {
            continue;
        }
        let Some(offsets) = tensor.get("data_offsets").and_then(Value::as_array) else {
            return false;
        };
        if offsets.len() != 2 {
            return false;
        }
        let (Some(start), Some(end)) = (offsets[0].as_u64(), offsets[1].as_u64()) else {
            return false;
        };
        if end < start {
            return false;
        }
        data_len = data_len.max(end);
        tensor_count += 1;
    }
    tensor_count > 0 && data_start.checked_add(data_len) == Some(file_len)
}

fn is_complete_npz(path: &Path) -> bool {
    let Ok(metadata) = fs::metadata(path) else {
        return false;
    };
    if !metadata.is_file() || metadata.len() < 22 {
        return false;
    }
    // An NPZ file is a ZIP archive. Require a valid end-of-central-directory
    // record within the maximum ZIP comment window, excluding truncated files.
    let tail_len = metadata.len().min(65_557) as usize;
    let Ok(mut file) = fs::File::open(path) else {
        return false;
    };
    if file.seek(SeekFrom::End(-(tail_len as i64))).is_err() {
        return false;
    }
    let mut tail = vec![0_u8; tail_len];
    if file.read_exact(&mut tail).is_err() {
        return false;
    }
    tail.windows(22)
        .any(|record| record.starts_with(b"PK\x05\x06"))
}

pub(crate) fn model_display_name(path: &Path) -> String {
    if let Ok(repo_id) = fs::read_to_string(path.join(".minimlx-repo-id")) {
        let repo_id = repo_id.trim();
        if validate_repo_id(repo_id).is_ok() {
            return repo_id.to_string();
        }
    }
    let is_hub_snapshot = path
        .parent()
        .and_then(Path::file_name)
        .is_some_and(|name| name == "snapshots");
    if is_hub_snapshot {
        if let Some(repo) = path
            .parent()
            .and_then(Path::parent)
            .and_then(Path::file_name)
            .and_then(|name| name.to_str())
            .and_then(|name| name.strip_prefix("models--"))
        {
            return repo.replace("--", "/");
        }
    }
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("Model");
    name.split_once("--")
        .map(|(publisher, model)| format!("{publisher}/{model}"))
        .unwrap_or_else(|| name.to_string())
}

/// Match the ID returned by the API's GET /v1/models endpoint.
pub(crate) fn model_api_id(path: &Path, alias: Option<&str>) -> String {
    if let Some(alias) = alias.filter(|alias| !alias.is_empty()) {
        return alias.to_string();
    }
    if let Ok(repo_id) = fs::read_to_string(path.join(".minimlx-repo-id")) {
        let repo_id = repo_id.trim();
        if repo_id.contains('/') {
            return repo_id.rsplit('/').next().unwrap_or(repo_id).to_string();
        }
    }
    if path.parent().and_then(Path::file_name).is_some_and(|name| name == "snapshots") {
        if let Some(repo) = path.parent().and_then(Path::parent).and_then(Path::file_name)
            .and_then(|name| name.to_str()).and_then(|name| name.strip_prefix("models--")) {
            return repo.split_once("--").map_or(repo, |(_, name)| name).to_string();
        }
    }
    let name = path.file_name().and_then(|name| name.to_str()).unwrap_or("minimlx");
    name.split_once("--").map_or(name, |(_, model)| model).to_string()
}

fn paged_external_ple_bytes(root: &Path) -> Option<u64> {
    if std::env::var("MINIMLX_QWEN4_PLE_SSD").ok().as_deref() == Some("0") {
        return None;
    }
    let config: Value = serde_json::from_slice(&fs::read(root.join("config.json")).ok()?).ok()?;
    if config.get("model_type").and_then(Value::as_str) != Some("qwen4_exp") {
        return None;
    }
    let source_index: Value = serde_json::from_slice(
        &fs::read(root.parent()?.join("model.safetensors.index.json")).ok()?,
    ).ok()?;
    let marker = ".ple.ple_embedding.ngram_embedding.shards.";
    let prefixes: std::collections::HashSet<_> = source_index.get("weight_map")?.as_object()?.keys()
        .filter(|key| key.contains(marker))
        .filter_map(|key| key.rsplit_once('.').map(|(prefix, _)| prefix))
        .collect();
    if prefixes.is_empty() {
        return None;
    }
    let source_config: Value = serde_json::from_slice(&fs::read(root.parent()?.join("config.json")).ok()?).ok()?;
    if source_config.get("model_type").and_then(Value::as_str) != Some("qwen4_exp") { return None; }
    let quant = source_config.get("quantization_config")
        .filter(|v| v.as_object().is_some_and(|obj| !obj.is_empty()))
        .or_else(|| source_config.get("quantization"))?;
    let mut layout = None;
    if !prefixes.iter().all(|prefix| {
        let params = quant.get(*prefix).unwrap_or(quant);
        let current = (params.get("bits").and_then(Value::as_u64),
                       params.get("group_size").and_then(Value::as_u64),
                       params.get("mode").and_then(Value::as_str));
        let valid = matches!(current, (Some(4), Some(32), Some("affine")) | (Some(4), Some(16), Some("nvfp4")));
        let consistent = layout.is_none() || layout == Some(current);
        layout = Some(current);
        valid && consistent
    }) { return None; }
    let mut ple_bytes = 0u64;
    let mut found = false;
    for entry in fs::read_dir(root).ok()? {
        let entry = entry.ok()?;
        let name = entry.file_name();
        let name = name.to_str()?;
        if !name.starts_with("resident-") || !name.ends_with(".safetensors") {
            continue;
        }
        let mut file = fs::File::open(entry.path()).ok()?;
        let length = file.metadata().ok()?.len();
        let mut prefix = [0u8; 8];
        file.read_exact(&mut prefix).ok()?;
        let header_len = u64::from_le_bytes(prefix);
        if header_len == 0 || header_len > 100 * 1024 * 1024 || header_len.checked_add(8)? > length {
            return None;
        }
        let mut header = vec![0; usize::try_from(header_len).ok()?];
        file.read_exact(&mut header).ok()?;
        let header: Value = serde_json::from_slice(&header).ok()?;
        for (key, tensor) in header.as_object()? {
            if !key.contains(marker) { continue; }
            let offsets = tensor.get("data_offsets")?.as_array()?;
            if offsets.len() != 2 { return None; }
            let (start, end) = (offsets[0].as_u64()?, offsets[1].as_u64()?);
            if end < start || end > length.checked_sub(header_len + 8)? { return None; }
            ple_bytes = ple_bytes.checked_add(end - start)?;
            found = true;
        }
    }
    found.then_some(ple_bytes)
}

pub(crate) fn supports_ssd_paging_config(config: &Value) -> bool {
    let kind = config.get("model_type").and_then(Value::as_str);
    let text = config.get("text_config").unwrap_or(config);
    kind == Some("qwen4_exp") || (kind == Some("qwen3_5_moe")
        && text.get("num_experts").and_then(Value::as_u64) == Some(256)
        && text.get("hidden_size").and_then(Value::as_u64) == Some(2048)
        && text.get("num_hidden_layers").and_then(Value::as_u64) == Some(40))
}

fn estimate_model_weight_memory(model_path: &str) -> u64 {
    // MLX loads packed/quantized tensors close to their on-disk size. Allow
    // 25% for loader/runtime overhead instead of doubling every model's size.
    let root = Path::new(model_path);
    let weight_size: u64 = if root.join("offload_index.json").is_file() {
        fs::read_dir(root)
            .ok()
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("resident-") && n.ends_with(".safetensors"))
            })
            .filter_map(|p| fs::metadata(p).ok().map(|m| m.len()))
            .sum()
    } else {
        fs::read_dir(root)
            .ok()
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                !p.file_name().is_some_and(|n| {
                    n.to_string_lossy().starts_with(".minimlx-ssd")
                        || n.to_string_lossy().starts_with(".minimlx-mtp")
                })
            })
            .map(|p| {
                if p.is_dir() {
                    directory_size(&p)
                } else {
                    fs::metadata(p).map(|m| m.len()).unwrap_or(0)
                }
            })
            .sum()
    };
    let resident_size = weight_size.saturating_sub(
        if root.join("offload_index.json").is_file() {
            paged_external_ple_bytes(root).unwrap_or(0)
        } else {
            0
        },
    );
    resident_size.saturating_mul(MODEL_WEIGHT_OVERHEAD_NUMERATOR) / MODEL_WEIGHT_OVERHEAD_DENOMINATOR
}

fn estimate_model_runtime_memory(model_path: &str, context_tokens: u32) -> u64 {
    let path = Path::new(model_path);
    let weight_memory = estimate_model_weight_memory(model_path);
    let config = fs::read_to_string(path.join("config.json"))
        .ok()
        .and_then(|contents| serde_json::from_str::<Value>(&contents).ok());
    let Some(config) = config else {
        return weight_memory;
    };
    let model_config = config.get("text_config").unwrap_or(&config);
    let config_u64 = |keys: &[&str]| {
        keys.iter()
            .find_map(|key| model_config.get(*key).and_then(Value::as_u64))
    };
    let layers = (config.get("model_type").and_then(Value::as_str) == Some("qwen4_exp"))
        .then(|| model_config.get("layer_types").and_then(Value::as_array))
        .flatten()
        .filter(|types| !types.is_empty())
        .map(|types| {
            types
                .iter()
                .filter(|kind| kind.as_str() == Some("full_attention"))
                .count() as u64
        })
        .or_else(|| config_u64(&["num_hidden_layers", "n_layer"]));
    let kv_heads = config_u64(&["num_key_value_heads", "num_kv_heads"]);
    let attention_heads = config_u64(&["num_attention_heads", "n_head"]);
    let hidden_size = config_u64(&["hidden_size", "n_embd"]);
    let head_dim = config_u64(&["head_dim"]).or_else(|| {
        hidden_size
            .zip(attention_heads)
            .and_then(|(hidden, heads)| (heads > 0).then_some(hidden / heads))
    });
    let kv_cache = layers
        .zip(kv_heads)
        .zip(head_dim)
        .map(|((layers, heads), head_dim)| {
            (context_tokens as u64)
                .saturating_mul(layers)
                .saturating_mul(heads)
                .saturating_mul(head_dim)
                .saturating_mul(4) // K and V in 16-bit storage.
        })
        .unwrap_or(0);
    weight_memory.saturating_add(kv_cache.saturating_mul(3) / 2)
}

fn memory_headroom(total_memory: u64) -> u64 {
    total_memory
        .saturating_mul(MEMORY_HEADROOM_PERCENT)
        .checked_div(100)
        .unwrap_or(0)
        .clamp(MINIMUM_MEMORY_HEADROOM, MAXIMUM_MEMORY_HEADROOM)
}

fn format_bytes(bytes: u64) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.1} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if bytes >= 1024 * 1024 {
        format!("{:.0} MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{bytes} B")
    }
}

fn directory_size(path: &Path) -> u64 {
    let Ok(entries) = fs::read_dir(path) else {
        return 0;
    };
    entries
        .flatten()
        .map(|entry| {
            let path = entry.path();
            if entry.file_type().is_ok_and(|kind| kind.is_dir()) {
                directory_size(&path)
            } else if entry.file_name().to_string_lossy().starts_with(".minimlx-") {
                0
            } else {
                fs::metadata(path)
                    .map(|metadata| metadata.len())
                    .unwrap_or(0)
            }
        })
        .sum()
}

const DOWNLOAD_MANIFEST: &str = ".minimlx-download-manifest.json";

fn download_manifest_path(directory: &Path) -> PathBuf {
    directory.join(DOWNLOAD_MANIFEST)
}

fn save_download_manifest(directory: &Path, files: &[RemoteModelFile]) -> Result<(), String> {
    let contents = serde_json::to_vec(files)
        .map_err(|error| format!("could not encode model download manifest: {error}"))?;
    fs::write(download_manifest_path(directory), contents)
        .map_err(|error| format!("could not save model download manifest: {error}"))
}

fn load_download_manifest(directory: &Path) -> Option<Vec<RemoteModelFile>> {
    let contents = fs::read(download_manifest_path(directory)).ok()?;
    let files: Vec<RemoteModelFile> = serde_json::from_slice(&contents).ok()?;
    (!files.is_empty()
        && files
            .iter()
            .all(|file| validate_remote_path(&file.path).is_ok()))
    .then_some(files)
}

fn downloaded_bytes_for_files(directory: &Path, files: &[RemoteModelFile]) -> u64 {
    files
        .iter()
        .map(|file| {
            let final_path = directory.join(&file.path);
            let partial_path = directory
                .join(".cache/huggingface/download")
                .join(format!("{}.incomplete", file.path));
            if let (Some(expected), Ok(metadata)) = (file.size, fs::metadata(&final_path)) {
                if metadata.is_file() && metadata.len() == expected {
                    return expected;
                }
            }
            fs::metadata(partial_path)
                .ok()
                .filter(|metadata| metadata.is_file())
                .map(|metadata| {
                    file.size
                        .map_or(metadata.len(), |expected| metadata.len().min(expected))
                })
                .unwrap_or(0)
        })
        .sum()
}

#[cfg(test)]
mod model_discovery_tests {
    use super::{
        Backend, DownloadContext, ModelSearchResult, ModelSearchSort, ModelSearchSortDirection,
        RemoteModelFile, discover_model_directories, download_retry_delay,
        downloaded_bytes_for_files, estimate_model_runtime_memory, huggingface_file_url,
        huggingface_model_search_params, huggingface_model_search_url, is_managed_model_dir,
        is_model_dir, is_unneeded_huggingface_metadata, model_display_name, modelscope_file_url,
        modelscope_model_search_params, modelscope_model_search_url, parse_content_range_start,
        parse_huggingface_repo_size, same_filesystem_path, save_download_manifest,
        should_retry_download_status, sort_model_search_results, validate_external_model_directory,
        validate_remote_path, validate_repo_id,
    };
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn write_safetensors(path: &PathBuf) {
        let header = br#"{"tensor":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let padding = (8 - header.len() % 8) % 8;
        let mut padded = header.to_vec();
        padded.resize(padded.len() + padding, b' ');
        let mut contents = (padded.len() as u64).to_le_bytes().to_vec();
        contents.extend(padded);
        contents.extend([0_u8; 4]);
        fs::write(path, contents).expect("complete safetensors fixture should be written");
    }

    #[test]
    fn model_api_id_uses_public_name_and_alias() {
        let path = std::path::Path::new("/models/mlx-community--Llama-3.2-3B-Instruct-4bit");
        assert_eq!(super::model_api_id(path, None), "Llama-3.2-3B-Instruct-4bit");
        assert_eq!(super::model_api_id(path, Some("my-alias")), "my-alias");
        let snapshot = std::path::Path::new("/cache/models--org--model--variant/snapshots/revision");
        assert_eq!(super::model_api_id(snapshot, None), "model--variant");
    }

    #[test]
    fn ssd_paging_recognizes_qwen36_35b_geometry() {
        let config = serde_json::json!({"model_type":"qwen3_5_moe", "text_config":{
            "num_experts":256,"hidden_size":2048,"num_hidden_layers":40}});
        assert!(super::supports_ssd_paging_config(&config));
        let mut other = config.clone();
        other["text_config"]["num_experts"] = serde_json::json!(128);
        assert!(!super::supports_ssd_paging_config(&other));
        assert!(super::supports_ssd_paging_config(&serde_json::json!({"model_type":"qwen4_exp"})));
    }

    #[test]
    fn paged_ple_memory_excludes_only_valid_external_tensor_ranges() {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let root = std::env::temp_dir().join(format!("minimlx-ple-memory-{}-{nonce}", std::process::id()));
        let paged = root.join(".minimlx-ssd");
        fs::create_dir_all(&paged).unwrap();
        let name = "model.ple.ple_embedding.ngram_embedding.shards.0.weight";
        let config = serde_json::json!({"model_type":"qwen4_exp", "quantization":{"bits":4,"group_size":32,"mode":"affine"}});
        fs::write(root.join("config.json"), config.to_string()).unwrap();
        fs::write(paged.join("config.json"), config.to_string()).unwrap();
        fs::write(root.join("model.safetensors.index.json"), serde_json::json!({"weight_map":{name:"weights.safetensors"}}).to_string()).unwrap();
        let header = serde_json::json!({name:{"dtype":"U32","shape":[2],"data_offsets":[0,8]}, "other":{"dtype":"U32","shape":[2],"data_offsets":[8,16]}}).to_string();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header.as_bytes());
        bytes.extend([0u8; 16]);
        fs::write(paged.join("resident-0001.safetensors"), &bytes).unwrap();
        assert_eq!(super::paged_external_ple_bytes(&paged), Some(8));
        bytes.truncate(bytes.len() - 12);
        fs::write(paged.join("resident-0001.safetensors"), &bytes).unwrap();
        assert_eq!(super::paged_external_ple_bytes(&paged), None);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn validates_hub_ids_and_remote_paths_before_file_writes() {
        assert_eq!(
            validate_repo_id(" mlx-community/example ").unwrap(),
            "mlx-community/example"
        );
        assert!(validate_repo_id("mlx-community/../outside").is_err());
        assert!(validate_repo_id("mlx-community/model?revision=main").is_err());
        assert!(validate_remote_path("weights/model-00001.safetensors").is_ok());
        assert!(validate_remote_path("../../outside").is_err());
        assert!(validate_remote_path("weights\\outside").is_err());
    }

    #[test]
    fn parses_byte_range_start_for_safe_resume() {
        assert_eq!(
            parse_content_range_start("bytes 4096-8191/16384"),
            Some(4096)
        );
        assert_eq!(parse_content_range_start("bytes */16384"), None);
        assert_eq!(parse_content_range_start("items 0-1/2"), None);
    }

    #[test]
    fn downloaded_size_uses_remote_manifest_and_excludes_unlisted_files() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "minimlx-download-size-{}-{nonce}",
            std::process::id()
        ));
        let cache = root.join(".cache/huggingface/download");
        fs::create_dir_all(&cache).expect("download cache should be created");
        fs::write(root.join("config.json"), [0_u8; 10]).expect("payload file should be written");
        fs::write(root.join("stale-file.bin"), [0_u8; 40]).expect("stale file should be written");
        fs::write(cache.join("weights.safetensors.incomplete"), [0_u8; 20])
            .expect("partial file should be written");
        let files = vec![
            RemoteModelFile {
                path: "config.json".into(),
                size: Some(10),
                revision: "rev".into(),
            },
            RemoteModelFile {
                path: "weights.safetensors".into(),
                size: Some(100),
                revision: "rev".into(),
            },
        ];
        assert_eq!(downloaded_bytes_for_files(&root, &files), 30);
        fs::remove_dir_all(root).expect("fixture directory should be removed");
    }

    #[test]
    fn builds_encoded_hub_file_urls_without_losing_endpoint_prefix() {
        let context = DownloadContext {
            download_directory: PathBuf::new(),
            huggingface_endpoint: "https://mirror.example/hub".into(),
            modelscope_endpoint: "https://modelscope.cn".into(),
            http_proxy: String::new(),
            https_proxy: String::new(),
            no_proxy: String::new(),
            ca_bundle: String::new(),
        };
        let url = huggingface_file_url(
            &context,
            "mlx-community/example",
            "rev-1",
            "weights/model shard.safetensors",
        )
        .unwrap();
        assert_eq!(
            url.as_str(),
            "https://mirror.example/hub/mlx-community/example/resolve/rev-1/weights/model%20shard.safetensors"
        );
    }

    fn model_fixture(config: &str, name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "minimlx-model-discovery-{}-{nonce}-{name}",
            std::process::id()
        ));
        fs::create_dir_all(&path).expect("fixture directory should be created");
        fs::write(path.join("config.json"), config).expect("config should be written");
        fs::write(path.join("tokenizer_config.json"), "{}").expect("tokenizer should be written");
        write_safetensors(&path.join("model.safetensors"));
        path
    }

    #[test]
    fn managed_ssd_copy_requires_all_manifest_files_and_is_not_a_second_model() {
        let root = model_fixture(r#"{"model_type":"qwen4_exp"}"#, "ssd");
        let copy = root.join(".minimlx-ssd");
        fs::create_dir_all(copy.join("experts")).unwrap();
        fs::write(copy.join("config.json"), r#"{"model_type":"qwen4_exp"}"#).unwrap();
        fs::write(copy.join("tokenizer_config.json"), "{}").unwrap();
        fs::write(
            copy.join("offload_index.json"),
            r#"{"layers":[0,1],"num_experts":2}"#,
        )
        .unwrap();
        fs::write(
            copy.join(".minimlx-ssd-manifest.json"),
            r#"{"resident":["resident-0001.safetensors","resident-0002.safetensors"]}"#,
        )
        .unwrap();
        write_safetensors(&copy.join("resident-0001.safetensors"));
        write_safetensors(&copy.join("resident-0002.safetensors"));
        write_safetensors(&copy.join("experts/layer_0000.safetensors"));
        assert!(!is_model_dir(&copy));
        write_safetensors(&copy.join("experts/layer_0001.safetensors"));
        assert!(is_model_dir(&copy));
        assert_eq!(
            discover_model_directories(std::slice::from_ref(&root)),
            vec![root.clone()]
        );
        fs::remove_file(copy.join("resident-0002.safetensors")).unwrap();
        assert!(!is_model_dir(&copy));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn only_models_in_the_model_inventory_are_loadable_or_startable() {
        let listed_fixture = model_fixture(
            r#"{"model_type":"qwen3_5","architectures":["Qwen3_5ForConditionalGeneration"]}"#,
            "listed-model-fixture",
        );
        let external = model_fixture(
            r#"{"model_type":"qwen3_5","architectures":["Qwen3_5ForConditionalGeneration"]}"#,
            "external-model",
        );
        let root = std::env::temp_dir().join(format!(
            "minimlx-loaded-inventory-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let models = root.join("models");
        fs::create_dir_all(&models).expect("managed models directory should be created");
        let listed = models.join("listed-model");
        fs::rename(&listed_fixture, &listed)
            .expect("listed model should be moved into the library");
        let mut backend = Backend::new(root.clone(), PathBuf::from("python"), models.clone());
        backend
            .configure_storage(vec![models])
            .expect("managed model directory should be configured");
        backend.loaded_model = Some(external.display().to_string());
        let inventory = backend.models();
        assert_eq!(inventory.len(), 1);
        assert!(same_filesystem_path(
            &inventory[0].path,
            &listed.display().to_string()
        ));
        assert!(backend.is_listed_model_path(&listed.display().to_string()));
        assert!(!backend.is_listed_model_path(&external.display().to_string()));
        assert!(
            backend
                .start_api_with_options(&external.display().to_string(), None)
                .unwrap_err()
                .contains("Only models in the MiniMLX model list can be started")
        );

        drop(backend);
        fs::remove_dir_all(external).expect("external model fixture should be removed");
        fs::remove_dir_all(root).expect("managed model fixture should be removed");
    }

    #[test]
    fn qwen4_context_estimate_counts_only_full_attention_layers() {
        let model = model_fixture(
            r#"{"model_type":"qwen4_exp","text_config":{"num_hidden_layers":4,"num_key_value_heads":1,"head_dim":16,"layer_types":["linear_attention","linear_attention","linear_attention","full_attention"]}}"#,
            "hybrid-memory-estimate",
        );
        let path = model.display().to_string();
        let base = estimate_model_runtime_memory(&path, 0);
        let with_context = estimate_model_runtime_memory(&path, 8);
        assert_eq!(with_context - base, 8 * 16 * 4 * 3 / 2);
        fs::remove_dir_all(model).unwrap();
    }

    #[test]
    fn memory_estimate_includes_context_kv_cache() {
        let model = model_fixture(
            r#"{"model_type":"test","num_hidden_layers":2,"num_key_value_heads":1,"hidden_size":16,"num_attention_heads":4}"#,
            "memory-estimate",
        );
        let weights = super::directory_size(&model)
            .saturating_mul(super::MODEL_WEIGHT_OVERHEAD_NUMERATOR)
            / super::MODEL_WEIGHT_OVERHEAD_DENOMINATOR;
        let zero_context = estimate_model_runtime_memory(&model.display().to_string(), 0);
        let with_context = estimate_model_runtime_memory(&model.display().to_string(), 8);
        let expected_kv_bytes = 8_u64 * 2 * 1 * 4 * 4 * 3 / 2;

        assert_eq!(zero_context, weights);
        assert_eq!(with_context, weights + expected_kv_bytes);
        fs::remove_dir_all(model).expect("model fixture should be removed");
    }

    #[test]
    fn memory_guard_keeps_a_scaled_but_bounded_os_reserve() {
        assert_eq!(
            super::memory_headroom(64 * super::GIB),
            1 * super::GIB + super::GIB / 2
        );
        assert_eq!(
            super::memory_headroom(16 * super::GIB),
            1 * super::GIB + super::GIB / 2
        );
        assert_eq!(
            super::memory_headroom(128 * super::GIB),
            2 * super::GIB + super::GIB * 56 / 100
        );
    }

    #[test]
    fn short_model_search_query_returns_without_network_or_python() {
        let context = DownloadContext {
            download_directory: PathBuf::new(),
            huggingface_endpoint: "not-a-valid-endpoint".into(),
            modelscope_endpoint: "https://modelscope.cn".into(),
            http_proxy: String::new(),
            https_proxy: String::new(),
            no_proxy: String::new(),
            ca_bundle: String::new(),
        };
        assert!(
            Backend::search_models(&context, "q", "huggingface")
                .expect("short query should be ignored")
                .is_empty()
        );
    }

    #[test]
    fn huggingface_search_uses_only_fields_supported_by_list_models() {
        let params = huggingface_model_search_params("qwen");
        let expand_values: Vec<_> = params
            .iter()
            .filter(|(key, _)| *key == "expand")
            .map(|(_, value)| *value)
            .collect();
        assert_eq!(expand_values, ["downloads", "likes", "safetensors"]);
        assert!(params.contains(&("filter", "mlx")));
        assert!(!params.iter().any(|(_, value)| *value == "usedStorage"));
    }

    #[test]
    fn huggingface_dflash_search_includes_non_mlx_draft_repositories() {
        let params = huggingface_model_search_params("Qwen DFlash2");
        assert!(params.contains(&("search", "Qwen DFlash2")));
        assert!(!params.iter().any(|(key, _)| *key == "filter"));
        assert!(params.contains(&("sort", "downloads")));
    }

    #[test]
    fn skips_only_gitattributes_from_huggingface_file_downloads() {
        assert!(is_unneeded_huggingface_metadata(".gitattributes"));
        assert!(!is_unneeded_huggingface_metadata("config.json"));
        assert!(!is_unneeded_huggingface_metadata("model.safetensors"));
    }

    #[test]
    fn retries_only_rate_limits_and_server_errors_with_backoff() {
        assert!(should_retry_download_status(
            reqwest::StatusCode::TOO_MANY_REQUESTS
        ));
        assert!(should_retry_download_status(
            reqwest::StatusCode::SERVICE_UNAVAILABLE
        ));
        assert!(!should_retry_download_status(
            reqwest::StatusCode::NOT_FOUND
        ));
        assert_eq!(
            download_retry_delay(Some("7"), 0),
            std::time::Duration::from_secs(7)
        );
        assert_eq!(
            download_retry_delay(Some("999"), 0),
            std::time::Duration::from_secs(60)
        );
        assert_eq!(
            download_retry_delay(None, 3),
            std::time::Duration::from_secs(8)
        );
    }

    #[test]
    fn huggingface_search_uses_the_configured_mirror_endpoint() {
        let context = DownloadContext {
            download_directory: PathBuf::new(),
            huggingface_endpoint: "https://mirror.example/hub/".into(),
            modelscope_endpoint: "https://modelscope.cn".into(),
            http_proxy: String::new(),
            https_proxy: String::new(),
            no_proxy: String::new(),
            ca_bundle: String::new(),
        };
        assert_eq!(
            huggingface_model_search_url(&context),
            "https://mirror.example/hub/api/models"
        );
    }

    #[test]
    fn modelscope_search_and_download_use_the_configured_endpoint() {
        let context = DownloadContext {
            download_directory: PathBuf::new(),
            huggingface_endpoint: "https://huggingface.co".into(),
            modelscope_endpoint: "https://models.example/hub/".into(),
            http_proxy: String::new(),
            https_proxy: String::new(),
            no_proxy: String::new(),
            ca_bundle: String::new(),
        };
        assert_eq!(
            modelscope_model_search_url(&context),
            "https://models.example/hub/openapi/v1/models"
        );
        let file_url = modelscope_file_url(
            &context,
            "owner/model",
            "master",
            "weights/model.safetensors",
        )
        .unwrap();
        assert_eq!(
            file_url.origin().ascii_serialization(),
            "https://models.example"
        );
        assert!(
            file_url
                .path()
                .starts_with("/hub/api/v1/models/owner/model/repo")
        );
    }

    #[test]
    fn modelscope_search_filters_for_mlx_library() {
        let params = modelscope_model_search_params("qwen");
        assert!(params.contains(&("filter.library", "mlx")));
    }

    #[test]
    fn sorts_model_search_results_by_name_size_downloads_and_likes() {
        let mut results = vec![
            ModelSearchResult {
                repo_id: "z/model".into(),
                downloads: Some(10),
                likes: Some(3),
                parameters: None,
                size_bytes: Some(100),
            },
            ModelSearchResult {
                repo_id: "A/model".into(),
                downloads: Some(30),
                likes: Some(1),
                parameters: None,
                size_bytes: Some(300),
            },
            ModelSearchResult {
                repo_id: "b/model".into(),
                downloads: Some(20),
                likes: Some(9),
                parameters: None,
                size_bytes: Some(200),
            },
        ];

        for (sort, direction, expected) in [
            (
                ModelSearchSort::Name,
                ModelSearchSortDirection::Ascending,
                ["A/model", "b/model", "z/model"],
            ),
            (
                ModelSearchSort::Size,
                ModelSearchSortDirection::Descending,
                ["A/model", "b/model", "z/model"],
            ),
            (
                ModelSearchSort::Downloads,
                ModelSearchSortDirection::Descending,
                ["A/model", "b/model", "z/model"],
            ),
            (
                ModelSearchSort::Likes,
                ModelSearchSortDirection::Descending,
                ["b/model", "z/model", "A/model"],
            ),
        ] {
            sort_model_search_results(&mut results, sort, direction);
            assert_eq!(
                results
                    .iter()
                    .map(|model| model.repo_id.as_str())
                    .collect::<Vec<_>>(),
                expected
            );
        }
    }

    #[test]
    fn aggregates_real_huggingface_repository_file_sizes() {
        let payload = serde_json::json!({
            "siblings": [
                {"size": 120},
                {"lfs": {"size": 880}},
                {"rfilename": "README.md"}
            ]
        });
        assert_eq!(parse_huggingface_repo_size(&payload), Some(1000));
        assert_eq!(
            parse_huggingface_repo_size(&serde_json::json!({"siblings": []})),
            None
        );
    }

    #[test]
    fn excludes_auxiliary_ocr_and_speech_checkpoints() {
        let paddle = model_fixture(
            r#"{"model_type":"paddleocr_vl","architectures":["PaddleOCRVLForConditionalGeneration"]}"#,
            "paddle",
        );
        let whisper = model_fixture(
            r#"{"model_type":"whisper","architectures":["WhisperForConditionalGeneration"]}"#,
            "whisper",
        );
        let qwen = model_fixture(
            r#"{"model_type":"qwen3_5","architectures":["Qwen3_5ForConditionalGeneration"]}"#,
            "qwen",
        );

        assert!(!is_model_dir(&paddle));
        assert!(!is_model_dir(&whisper));
        assert!(is_model_dir(&qwen));

        let cache_model = model_fixture(
            r#"{"model_type":"qwen3_5","architectures":["Qwen3_5ForConditionalGeneration"]}"#,
            "cached-qwen",
        );
        let cache_root = cache_model.with_file_name(format!(
            "cache-test-{}",
            cache_model.file_name().unwrap().to_string_lossy()
        ));
        let cache_parent = cache_root.join(".cache");
        fs::create_dir_all(&cache_parent).expect("cache fixture parent should be created");
        let cached_path = cache_parent.join("model");
        fs::create_dir_all(&cached_path).expect("cached model fixture should be created");
        fs::write(
            cached_path.join("config.json"),
            r#"{"model_type":"qwen3_5"}"#,
        )
        .expect("cached config should be written");
        fs::write(cached_path.join("tokenizer_config.json"), "{}")
            .expect("cached tokenizer should be written");
        fs::write(cached_path.join("model.safetensors"), "weights")
            .expect("cached weights should be written");
        assert!(validate_external_model_directory(&cached_path).is_err());

        let systran = model_fixture(r#"{"suppress_ids":[1,2,3]}"#, "systran");
        fs::remove_file(systran.join("model.safetensors"))
            .expect("fixture safetensors should be removed");
        fs::write(systran.join("model.bin"), "ct2 fixture").expect("ct2 weights should be written");
        assert!(!is_model_dir(&systran));

        fs::remove_dir_all(paddle).expect("paddle fixture should be removed");
        fs::remove_dir_all(whisper).expect("whisper fixture should be removed");
        fs::remove_dir_all(qwen).expect("qwen fixture should be removed");
        fs::remove_dir_all(systran).expect("systran fixture should be removed");
        fs::remove_dir_all(cache_model).expect("cached model fixture should be removed");
        fs::remove_dir_all(cache_root).expect("cache fixture root should be removed");
    }

    #[test]
    fn hub_snapshot_uses_repository_id_as_display_name() {
        let path =
            PathBuf::from("/cache/hub/models--mlx-community--Qwen3.5-4B-4bit/snapshots/abc123");
        assert_eq!(model_display_name(&path), "mlx-community/Qwen3.5-4B-4bit");
    }

    #[test]
    fn flattened_download_directory_keeps_repository_id_as_display_name() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "minimlx-model-display-name-{}-{nonce}",
            std::process::id()
        ));
        let directory = root.join("mlx-community--Qwen3.5-4B-4bit");
        fs::create_dir_all(&directory).expect("model directory should be created");
        assert_eq!(
            model_display_name(&directory),
            "mlx-community/Qwen3.5-4B-4bit"
        );
        fs::write(
            directory.join(".minimlx-repo-id"),
            "mlx-community/Qwen3.5-4B-4bit",
        )
        .expect("repository ID metadata should be saved");
        assert_eq!(
            model_display_name(&directory),
            "mlx-community/Qwen3.5-4B-4bit"
        );
        fs::remove_dir_all(root).expect("fixture directory should be removed");
    }

    #[test]
    fn incomplete_sharded_download_is_not_a_loadable_model() {
        let model = model_fixture(
            r#"{"model_type":"qwen3_5","architectures":["Qwen3_5ForConditionalGeneration"]}"#,
            "incomplete-qwen",
        );
        fs::remove_file(model.join("model.safetensors"))
            .expect("fixture single shard should be removed");
        fs::write(
            model.join("model.safetensors.index.json"),
            r#"{"weight_map":{"tensor_a":"model-00001-of-00002.safetensors","tensor_b":"model-00002-of-00002.safetensors"}}"#,
        )
        .expect("index should be written");

        assert!(
            !is_model_dir(&model),
            "index without weight shards must be excluded"
        );
        assert!(discover_model_directories(std::slice::from_ref(&model)).is_empty());
        write_safetensors(&model.join("model-00001-of-00002.safetensors"));
        assert!(
            !is_model_dir(&model),
            "one missing shard must keep model excluded"
        );
        assert!(discover_model_directories(std::slice::from_ref(&model)).is_empty());
        write_safetensors(&model.join("model-00002-of-00002.safetensors"));
        assert!(
            is_model_dir(&model),
            "all complete indexed shards should be loadable"
        );
        assert_eq!(
            discover_model_directories(std::slice::from_ref(&model)).len(),
            1
        );

        fs::write(model.join("model-00002-of-00002.safetensors"), b"truncated")
            .expect("shard should be truncated for test");
        assert!(
            !is_model_dir(&model),
            "truncated safetensors shard must be excluded"
        );
        fs::remove_dir_all(model).expect("fixture should be removed");
    }

    #[test]
    fn saved_huggingface_partial_is_listed_for_resume_not_as_installed() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "minimlx-incomplete-download-{}-{nonce}",
            std::process::id()
        ));
        let models = root.join("models");
        let partial = models.join("mlx-community--Qwen3.5-35B-A3B-4bit");
        fs::create_dir_all(partial.join(".cache/huggingface/download"))
            .expect("partial cache should be created");
        fs::write(
            partial.join("config.json"),
            r#"{"model_type":"qwen3_5","architectures":["Qwen3_5ForConditionalGeneration"]}"#,
        )
        .expect("model config should be written");
        fs::write(partial.join("tokenizer_config.json"), "{}")
            .expect("tokenizer config should be written");
        fs::write(
            partial.join("model.safetensors.index.json"),
            r#"{"weight_map":{"w":"model-00001-of-00002.safetensors"}}"#,
        )
        .expect("partial index should be written");
        fs::write(
            partial.join(".cache/huggingface/download/shard.abc.incomplete"),
            b"partial shard bytes",
        )
        .expect("resumable shard should be written");
        save_download_manifest(
            &partial,
            &[RemoteModelFile {
                path: "shard.abc".into(),
                size: Some(100),
                revision: "revision".into(),
            }],
        )
        .expect("remote file manifest should be saved");

        let mut backend =
            super::Backend::new(root.clone(), PathBuf::from("python"), models.clone());
        backend
            .configure_storage(vec![models.clone()])
            .expect("managed model directory should be configured");
        assert!(backend.models().is_empty());
        let queued = backend.incomplete_downloads();
        assert_eq!(queued.len(), 1);
        assert_eq!(queued[0].repo_id, "mlx-community/Qwen3.5-35B-A3B-4bit");
        assert_eq!(
            queued[0].size_bytes,
            Some(b"partial shard bytes".len() as u64)
        );

        drop(backend);
        fs::remove_dir_all(root).expect("fixture root should be removed");
    }

    #[test]
    fn deletes_only_the_requested_incomplete_model_directory() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "minimlx-delete-incomplete-{}-{nonce}",
            std::process::id()
        ));
        let models = root.join("models");
        let partial = models.join("mlx-community--partial-model");
        fs::create_dir_all(partial.join(".cache/huggingface/download"))
            .expect("partial cache should be created");
        fs::write(
            partial.join(".cache/huggingface/download/weights.incomplete"),
            b"partial bytes",
        )
        .expect("partial weights should be written");
        fs::create_dir_all(&models).expect("model library should be created");
        fs::write(models.join("keep.txt"), b"keep").expect("neighbor fixture should be written");

        let mut backend = Backend::new(root.clone(), PathBuf::from("python"), models.clone());
        backend
            .configure_storage(vec![models.clone()])
            .expect("model directory should be configured");
        backend
            .delete_incomplete_download("mlx-community/partial-model")
            .expect("requested incomplete model should be deleted");

        assert!(!partial.exists());
        assert_eq!(fs::read(models.join("keep.txt")).unwrap(), b"keep");
        drop(backend);
        fs::remove_dir_all(root).expect("fixture root should be removed");
    }

    #[test]
    fn discovers_all_models_under_publisher_model_layout() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "minimlx-publisher-models-{}-{nonce}",
            std::process::id()
        ));
        let publisher = root.join("mlx-community");
        fs::create_dir_all(&publisher).expect("publisher folder should be created");
        for model_name in ["Qwen-A", "Qwen-B"] {
            let fixture = model_fixture(
                r#"{"model_type":"qwen3_5","architectures":["Qwen3_5ForConditionalGeneration"]}"#,
                model_name,
            );
            fs::rename(fixture, publisher.join(model_name)).expect("model should be nested");
        }

        let discovered = discover_model_directories(std::slice::from_ref(&root));
        assert_eq!(discovered.len(), 2);
        fs::remove_dir_all(root).expect("model library fixture should be removed");
    }

    #[test]
    fn discovers_dflash_checkpoint_without_treating_it_as_loadable_model() {
        let root = model_fixture(
            r#"{"model_type":"qwen3","architectures":["DFlash2DraftModel"]}"#,
            "dflash-draft",
        );
        fs::remove_file(root.join("tokenizer_config.json"))
            .expect("draft fixture should not have a tokenizer");

        assert!(!is_model_dir(&root));
        assert!(is_managed_model_dir(&root));
        assert_eq!(
            discover_model_directories(std::slice::from_ref(&root)),
            vec![root.clone()]
        );

        fs::remove_dir_all(root).expect("DFlash draft fixture should be removed");
    }
}
