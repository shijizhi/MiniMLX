# MiniMLX v0.1.0

MiniMLX brings local MLX models to a native Apple Silicon desktop app. Discover and manage models, run an OpenAI-compatible inference service, and monitor activity from one place.

## Highlights

- Browse, download, and manage compatible MLX models.
- Run a local OpenAI-compatible API and monitor service activity.
- Measure prefill and generation performance and track system resources.
- Use SSD-backed paging and speculative decoding with supported models when available.
- Keep the service accessible from the macOS menu bar.

## Install

Download `MiniMLX-0.1.0-arm64.dmg`, open it, and drag MiniMLX into Applications. Models are downloaded separately. An Apple Silicon Mac is required; model compatibility, memory requirements, and performance vary by model and hardware.

The DMG is signed with Developer ID and notarized by Apple. Use the accompanying SHA-256 file to verify the downloaded image.

## Notes

- SSD paging and acceleration are available only for supported models and configurations.
- This is the first release; feedback and model compatibility reports are welcome.
