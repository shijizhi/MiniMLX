"""Avoid rebuilding Qwen3.5 MTP cache state when the whole block is accepted."""

from __future__ import annotations


def install_mtp_full_accept_cache_fastpath() -> bool:
    """Keep already-current cache state on a full speculative acceptance.

    ArraysCache records temporal history so a rejected suffix can be selected
    at commit time. On full acceptance its live slots already contain the
    final state; selecting the recorded window again needlessly materializes a
    contiguous copy (notably for every Qwen3.5 GDN convolution window).
    Preserve the upstream path for every partial acceptance.
    """
    try:
        from mlx_vlm.models.cache import ArraysCache
    except ImportError:
        return False

    if getattr(ArraysCache, "_minimlx_mtp_full_accept_fastpath", False):
        return True

    original = ArraysCache.commit_speculation

    def commit_speculation(self, lengths, generation=None):
        transaction, normalized = self.validate_speculation(lengths, generation)
        if all(value == transaction["length"] for value in normalized):
            self._restore_speculative_metadata(
                normalized, transaction["initial_metadata"]
            )
            self._speculation = None
            return
        return original(self, normalized, generation)

    ArraysCache.commit_speculation = commit_speculation
    ArraysCache._minimlx_mtp_full_accept_fastpath = True
    ArraysCache._minimlx_mtp_full_accept_original = original
    return True
