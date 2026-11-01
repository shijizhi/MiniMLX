"""Layer-batched Qwen4 MTP verification.

Position-sensitive state updates and projections remain sequential inside each
layer, while expert routing is batched across the verify window.
"""
from __future__ import annotations


def _use_omlx_row_exact(mlx_version: str, requested: str) -> bool:
    if requested == "1" and mlx_version != "0.32.4":
        raise RuntimeError("oMLX row-exact verification requires mlx==0.32.4")
    return mlx_version == "0.32.4" and requested in ("1", "auto")


def install_batch_mtp_verifier(model) -> None:
    import os
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_vlm.models.qwen4_exp.language import Qwen4ExpBatchInvariantForward
    from mlx_vlm.speculative.cache_state import start_speculative_cache

    from .parallel_experts import exact_expert_rows

    from importlib.metadata import version
    mlx_version = version("mlx")
    requested = os.environ.get("MINIMLX_QWEN4_OMLX_VERIFY", "auto")
    use_omlx = _use_omlx_row_exact(mlx_version, requested)
    if use_omlx:
        from .omlx_kernels import row_exact_qmv

    class Verifier(Qwen4ExpBatchInvariantForward):
        def __init__(self):
            self.projection_mode = "omlx_row_exact" if use_omlx else "singleton_exact"

        def _linear(self, linear, x):
            if x.ndim != 3 or x.shape[0] != 1 or x.shape[1] <= 1:
                return linear(x)
            if use_omlx and isinstance(linear, nn.QuantizedLinear):
                return row_exact_qmv.quantized_linear(linear, x)
            return mx.concatenate([linear(x[:, i:i + 1]) for i in range(x.shape[1])], axis=1)

        def _linears(self, linears, x):
            if use_omlx and all(isinstance(linear, nn.QuantizedLinear) for linear in linears):
                return row_exact_qmv.quantized_linears(linears, x)
            return tuple(self._linear(linear, x) for linear in linears)

        def _switch_glu(self, switch_mlp, x, indices):
            with exact_expert_rows():
                return super()._switch_glu(switch_mlp, x, indices)

        def _gated_delta(self, layer, inputs, mask, cache):
            parent = super()._gated_delta
            if inputs.shape[1] <= 1:
                return parent(layer, inputs, mask, cache)
            return mx.concatenate([
                parent(layer, inputs[:, i:i + 1], None if mask is None else mask[:, i:i + 1], cache)
                for i in range(inputs.shape[1])
            ], axis=1)

        def _qsa_attention(self, attention, hidden, cache, positions, mask):
            parent = super()._qsa_attention
            if hidden.shape[1] <= 1:
                return parent(attention, hidden, cache, positions, mask)
            return mx.concatenate([
                parent(attention, hidden[:, i:i + 1], cache,
                       None if positions is None else positions[..., i:i + 1], None)
                for i in range(hidden.shape[1])
            ], axis=1)

        def _ple(self, module, hidden, ids, cache, mask):
            parent = super()._ple
            if hidden.shape[1] <= 1:
                return parent(module, hidden, ids, cache, mask)
            return mx.concatenate([
                parent(module, hidden[:, i:i + 1], ids[:, i:i + 1], cache,
                       None if mask is None else mask[:, i:i + 1])
                for i in range(hidden.shape[1])
            ], axis=1)

    lm = model.language_model
    exact_verify = lm.speculative_verify_hidden
    verifier = Verifier()
    object.__setattr__(model, "_minimlx_qwen4_batch_verifier", verifier)

    def verify(inputs, cache):
        batch, length = inputs.shape
        # The batched path is validated for single-sequence text. Preserve the
        # reference implementation for multi-sequence or multimodal positions.
        if batch != 1 or lm._position_ids is not None:
            return exact_verify(inputs, cache)
        mx.async_eval([entry.state for entry in cache if entry is not None])
        entry = cache[lm.model.fa_idx]
        offset = getattr(entry, "offset", 0)
        if isinstance(offset, mx.array) and offset.ndim > 0:
            offsets = offset[:batch].astype(mx.int64)
        else:
            offsets = mx.full((batch,), int(offset), dtype=mx.int64)
        if lm._rope_deltas is not None:
            offsets = offsets + lm._rope_deltas[:batch].reshape(-1).astype(mx.int64)
        positions = offsets[:, None] + mx.arange(length, dtype=mx.int64)[None]
        if lm._position_ids is not None and lm._position_ids.ndim == 3:
            positions = mx.broadcast_to(positions[None], (3, batch, length))
        transaction = start_speculative_cache(cache, length)
        try:
            output = verifier(lm, inputs, cache=cache, position_ids=positions, skip_logits=True)
            return output.hidden_states[-1], {}, transaction
        except BaseException:
            transaction.abort()
            raise

    lm.speculative_verify_hidden = verify
