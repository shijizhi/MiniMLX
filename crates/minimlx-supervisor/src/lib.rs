use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use minimlx_protocol::{Request, Response};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SupervisorError {
    #[error("failed to start Python worker: {0}")]
    Start(#[from] std::io::Error),
    #[error("worker protocol error: {0}")]
    Protocol(#[from] serde_json::Error),
}

pub struct PythonWorker {
    child: Child,
    socket_path: PathBuf,
}

impl PythonWorker {
    pub fn start(
        python: impl AsRef<Path>,
        project_root: impl AsRef<Path>,
        model_dir: impl Into<String>,
        socket_path: impl Into<PathBuf>,
    ) -> Result<Self, SupervisorError> {
        Self::start_with_env(python, project_root, model_dir, socket_path, &[])
    }

    pub fn start_with_env(
        python: impl AsRef<Path>,
        project_root: impl AsRef<Path>,
        model_dir: impl Into<String>,
        socket_path: impl Into<PathBuf>,
        environment: &[(&str, &str)],
    ) -> Result<Self, SupervisorError> {
        let socket_path = socket_path.into();
        let worker_module = "minimlx_runtime";
        let mut command = Command::new(python.as_ref());
        command
            .current_dir(project_root.as_ref())
            .env("PYTHONPATH", project_root.as_ref().join("python"))
            .arg("-m")
            .arg(worker_module)
            .arg("worker")
            .arg(model_dir.into())
            .arg("--socket")
            .arg(&socket_path)
            .stdin(Stdio::null())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit());
        for (name, value) in environment {
            command.env(name, value);
        }
        let child = command.spawn()?;
        Ok(Self { child, socket_path })
    }

    pub fn socket_path(&self) -> &Path {
        &self.socket_path
    }

    pub fn is_running(&mut self) -> Result<bool, SupervisorError> {
        Ok(self.child.try_wait()?.is_none())
    }

    pub fn request(&self, request: &Request) -> Result<Response, SupervisorError> {
        let deadline = Instant::now() + Duration::from_secs(30);
        let mut stream = loop {
            match UnixStream::connect(&self.socket_path) {
                Ok(stream) => break stream,
                Err(error) if Instant::now() < deadline => {
                    let _ = error;
                    std::thread::sleep(Duration::from_millis(50));
                }
                Err(error) => return Err(SupervisorError::Start(error)),
            }
        };
        stream.write_all(&Self::encode_request(request)?)?;
        stream.flush()?;
        let mut line = String::new();
        BufReader::new(stream).read_line(&mut line)?;
        decode_response(line.as_bytes())
    }

    pub fn encode_request(request: &Request) -> Result<Vec<u8>, SupervisorError> {
        let mut bytes = serde_json::to_vec(request)?;
        bytes.push(b'\n');
        Ok(bytes)
    }
}

impl Drop for PythonWorker {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = std::fs::remove_file(&self.socket_path);
    }
}

pub fn decode_response(line: &[u8]) -> Result<Response, SupervisorError> {
    Ok(serde_json::from_slice(line)?)
}
