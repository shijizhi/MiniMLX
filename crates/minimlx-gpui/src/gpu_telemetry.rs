use std::process::Command;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

/// Latest system-wide Apple Silicon GPU device utilization, expressed as 0–100%.
/// `None` means the platform sampler has not produced a valid reading.
pub type GpuTelemetry = Arc<Mutex<Option<f32>>>;

/// Starts a process-lifetime sampler on a dedicated thread. The macOS
/// IORegistry exposes the AGX device's system-wide "Device Utilization %";
/// sampling is isolated from GPUI so command execution never blocks rendering.
pub fn start() -> GpuTelemetry {
    let latest = Arc::new(Mutex::new(None));
    let output = Arc::clone(&latest);

    let _ = thread::Builder::new()
        .name("minimlx-gpu-telemetry".into())
        .spawn(move || loop {
            let utilization = read_gpu_utilization();
            if let Ok(mut latest) = output.lock() {
                *latest = utilization;
            }
            thread::sleep(Duration::from_secs(1));
        });

    latest
}

fn read_gpu_utilization() -> Option<f32> {
    let output = Command::new("/usr/sbin/ioreg")
        .args(["-r", "-d", "1", "-c", "AGXAccelerator", "-l"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }

    let output = String::from_utf8_lossy(&output.stdout);
    let mut values = output.lines().filter_map(|line| {
        let (_, value) = line.split_once("\"Device Utilization %\"=")?;
        let digits = value
            .trim_start()
            .chars()
            .take_while(|ch| ch.is_ascii_digit() || *ch == '.')
            .collect::<String>();
        digits.parse::<f32>().ok().filter(|value| value.is_finite())
    });
    values.next().map(|value| value.clamp(0.0, 100.0))
}
