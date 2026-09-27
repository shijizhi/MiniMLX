"""Incremental Qwen3.5 MTP-head cache priming during backbone prefill."""

from __future__ import annotations

import mlx.core as mx


class MTPPromptPrimer:
    """Fold each target-prefill chunk into the MTP KV cache immediately."""

    def __init__(self, drafter, token_dtype: mx.Dtype):
        self.drafter = drafter
        self.token_dtype = token_dtype
        self.pending_hidden = None
        self.last_hidden = None
        drafter._next_position = 0

    def add_chunk(self, input_ids: mx.array, hidden: mx.array) -> None:
        if input_ids.shape[1] == 0:
            return
        if self.pending_hidden is None:
            # The first prompt token has no preceding target hidden to pair
            # with. Every subsequent token pairs with the previous position.
            if input_ids.shape[1] > 1:
                tokens = input_ids[:, 1:].astype(self.token_dtype)
                states = hidden[:, :-1, :]
            else:
                tokens = None
        else:
            tokens = input_ids.astype(self.token_dtype)
            states = mx.concatenate([self.pending_hidden, hidden[:, :-1, :]], axis=1)

        if tokens is not None and tokens.shape[1]:
            self.last_hidden = self.drafter._forward_tokens(
                tokens, states, self.token_dtype
            )
            # Bound deferred work and KV growth to this target-prefill chunk.
            arrays = [self.last_hidden]
            for cache in self.drafter._cache:
                for name in ("keys", "values"):
                    value = getattr(cache, name, None)
                    if isinstance(value, mx.array):
                        arrays.append(value)
            mx.async_eval(arrays)

        self.pending_hidden = hidden[:, -1:, :]

    def finish(self, bonus_token, sampler, greedy: bool) -> None:
        if self.pending_hidden is None:
            raise ValueError("cannot finalize MTP priming without target hidden")
        if isinstance(bonus_token, int):
            bonus = mx.array([[bonus_token]], dtype=self.token_dtype)
        else:
            bonus = bonus_token[:, None].astype(self.token_dtype)
        self.last_hidden = self.drafter._forward_tokens(
            bonus, self.pending_hidden, self.token_dtype
        )
        self.drafter._set_seed_from_hidden(self.last_hidden[:, -1:, :], sampler, greedy)


def prefill_mtp_from_target_hidden(
    drafter,
    input_ids: mx.array,
    hidden: mx.array,
    bonus_token,
    sampler,
    token_dtype: mx.Dtype,
    greedy: bool,
    chunk_size: int,
) -> None:
    """Prime the MTP KV cache a prompt chunk at a time.

    The concatenated token/hidden timeline is identical to mlx-vlm's stock
    ``prefill_from_target_hidden``. Chunk boundaries only bound graph and KV
    materialization, following the chunked KV materialization pattern in
    OMLX's Qwen3.5 VLM prompt-priming path. The caller still invokes this at
    MTP initialization, after the backbone prefill has completed.
    """
    if input_ids.shape[1] == 0:
        return
    if isinstance(bonus_token, int):
        bonus = mx.array([[bonus_token]], dtype=token_dtype)
    else:
        bonus = bonus_token[:, None].astype(token_dtype)

    shifted = mx.concatenate([input_ids[:, 1:].astype(token_dtype), bonus], axis=1)
    drafter._next_position = 0
    last_hidden = None
    for start in range(0, shifted.shape[1], max(1, int(chunk_size))):
        end = min(start + max(1, int(chunk_size)), shifted.shape[1])
        last_hidden = drafter._forward_tokens(
            shifted[:, start:end],
            hidden[:, start:end, :],
            token_dtype,
        )

        # Materialize each chunk's MTP KV before building the next graph, as
        # the OMLX prompt-priming path does. This bounds the prefill graph and
        # prevents a long prompt from remaining one deferred head operation.
        evals = [last_hidden]
        for cache in drafter._cache:
            for name in ("keys", "values"):
                value = getattr(cache, name, None)
                if isinstance(value, mx.array):
                    evals.append(value)
        mx.async_eval(evals)

    drafter._set_seed_from_hidden(last_hidden[:, -1:, :], sampler, greedy)
