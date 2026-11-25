# MiniMLX v0.1.2

This update adds initial support for Inkling 2 Mini text generation on Apple Silicon.

- Run the MLX 4-bit Inkling 2 Mini checkpoint through MiniMLX's MLX-LM backend.
- Keep Inkling's reasoning channel out of user-visible text responses.
- Preserve the existing model routing and acceleration behavior for other supported models.

## Install

Download `MiniMLX-0.1.2-arm64.dmg`, open it, and drag MiniMLX into Applications. Models are downloaded separately. Inkling 2 Mini support currently covers text generation; image input and Inkling-specific speculative decoding are not included. An Apple Silicon Mac is required. The DMG is signed and notarized; the accompanying SHA-256 file can be used to verify it.
