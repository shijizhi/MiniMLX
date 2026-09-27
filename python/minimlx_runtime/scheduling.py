from __future__ import annotations


def choose_prefill_step_size(prompt_tokens: int, requested: int) -> int:
    """Choose a conservative prefill chunk size.

    Positive values are explicit caller choices and are never changed. Zero is
    the protocol's opt-in ``auto`` value. The thresholds describe workload
    size, not a model family, so this can later be replaced by measured
    hardware profiles without changing the request protocol.
    """
    if requested > 0:
        return requested
    if prompt_tokens <= 4096:
        return 1024
    if prompt_tokens <= 16384:
        return 2048
    return 4096
