pub fn application_name() -> &'static str {
    "MiniMLX"
}

#[cfg(target_os = "macos")]
mod ui;

#[cfg(target_os = "macos")]
mod backend;

#[cfg(target_os = "macos")]
mod gpu_telemetry;

#[cfg(target_os = "macos")]
pub fn run_app(model_dir: String, root: std::path::PathBuf, python: std::path::PathBuf) {
    ui::run(model_dir, root, python);
}
