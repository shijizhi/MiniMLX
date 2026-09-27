//! Rust-side application/API boundary.
//!
//! HTTP or OpenAI-compatible routing will be added here. The inference
//! protocol remains a small local JSON-lines contract over the Python worker.

pub use minimlx_protocol::{Request, Response};
pub use minimlx_supervisor::PythonWorker;
