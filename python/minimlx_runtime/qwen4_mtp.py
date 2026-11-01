"""Detect and prepare the embedded Qwen4-Exp next-token head."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path


def embedded_mtp_source(model_dir: str | Path) -> Path | None:
    path = Path(model_dir)
    source = path.parent if path.name == ".minimlx-ssd" else path
    try:
        config = json.loads((source / "config.json").read_text())
        text = config.get("text_config") or {}
        if config.get("model_type") != "qwen4_exp" or not text.get("mtp_num_hidden_layers"):
            return None
        index = json.loads((source / "model.safetensors.index.json").read_text())
        weights = index.get("weight_map", {})
        if not any(key.startswith(("mtp.", "language_model.mtp.", "model.mtp.")) for key in weights):
            return None
    except (OSError, ValueError, TypeError, AttributeError):
        return None
    return source


def prepare_embedded_mtp(source: Path) -> Path:
    """Extract only the draft head; keep its lifecycle with its source model."""
    destination = source / ".minimlx-mtp"
    if destination.is_symlink():
        raise RuntimeError("MTP cache must not be a symbolic link")
    if (destination / ".minimlx-mtp-ready").is_file():
        return destination
    if destination.exists():
        raise RuntimeError(f"Incomplete MTP cache: {destination}")
    from mlx_vlm.speculative.drafters.qwen4_exp_mtp.split import split_qwen4_exp_mtp

    staging = Path(tempfile.mkdtemp(prefix=".minimlx-mtp-preparing-", dir=source))
    try:
        split_qwen4_exp_mtp(str(source), str(staging))
        (staging / ".minimlx-mtp-ready").write_text("1\n")
        try:
            os.rename(staging, destination)
        except OSError:
            if not (destination / ".minimlx-mtp-ready").is_file() or destination.is_symlink():
                raise
        return destination
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def install_exact_mtp_verifier(model) -> None:
    """Verify each token with the ordinary target forward and cache transaction.

    The upstream batched verifier has produced different greedy output for
    this quantized, expert-paged checkpoint. Keep the serial reference path
    until a batched verifier passes the same-device equivalence tests.
    """
    import mlx.core as mx
    from mlx_vlm.speculative.cache_state import start_speculative_cache

    lm = model.language_model

    def verify(inputs, cache):
        transaction = start_speculative_cache(cache, inputs.shape[1])
        try:
            states = []
            for index in range(inputs.shape[1]):
                output = lm(inputs[:, index:index + 1], cache=cache,
                            return_hidden=True, skip_logits=True)
                states.append(output.hidden_states[-1])
            return mx.concatenate(states, axis=1), {}, transaction
        except BaseException:
            transaction.abort()
            raise

    def argmax(hidden):
        return mx.concatenate([
            mx.argmax(lm.speculative_logits_from_hidden(hidden[:, index:index + 1, :]), axis=-1)
            for index in range(hidden.shape[1])
        ], axis=1)

    lm.speculative_verify_hidden = verify
    lm.speculative_argmax_from_hidden = argmax
    # The native drafter accepts accumulated hidden states from chunked
    # prefill. This is essential for paged weights on long prompts.
    model.chunked_prefill_policy = lambda **kwargs: True
