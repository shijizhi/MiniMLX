"""Version-scoped performance overlay for Qwen3.5 baseline instances."""

from __future__ import annotations

import os

from .qwen35_optimization import Qwen35OptimizationMixin


class Qwen35PerformanceOverlay(Qwen35OptimizationMixin):
    """Install Qwen3.5-only optimizations on an already loaded baseline adapter.

    The host owns model weights, tokenizer, processor, and lifecycle. This
    overlay only installs the Qwen3.5 execution hooks and optional caches.
    """

    name = "qwen35-performance-overlay"

    def __init__(self, host) -> None:
        object.__setattr__(self, "host", host)
        object.__setattr__(self, "_environment_before", {})
        object.__setattr__(self, "_kv_cache_step_before", None)

    def __getattr__(self, name):
        return getattr(self.host, name)

    def __setattr__(self, name, value):
        if name in {"host", "_environment_before", "_kv_cache_step_before"}:
            object.__setattr__(self, name, value)
        else:
            setattr(self.host, name, value)

    def install_text(self) -> None:
        self._install_qwen35_skip_logits()
        self._install_gdn_decode_conv_silu(self.model)
        if (
            os.environ.get("MINIMLX_GDN_DECODE_CUSTOM_RECURRENCE") == "1"
            or os.environ.get("MINIMLX_GDN_DECODE_FUSED_QK_NORM") == "1"
        ):
            from ..kernels.gdn_decode import install_gdn_decode_custom_recurrence

            if not install_gdn_decode_custom_recurrence():
                raise RuntimeError(
                    "GDN recurrence experiment could not install its dispatch hooks"
                )
        self._install_gdn_prefill_kernel()
        # Avoid retaining duplicate BF16 gate/up/down weights for long prefill.
        if os.environ.get("MINIMLX_ENABLE_Q4_NATIVE_PREFILL") != "1":
            self._install_q4_mlp_prefill_dense()
        self._install_q4_native_prefill()
        self._install_q4_native_mlp()
        self._install_q4_mlp_prefill_fusion()
        self._install_native_fa256_prefill()
        if os.environ.get("MINIMLX_FULL_ATTENTION_PREFILL_FORCE_FUSED") == "1":
            self._environment_before["MINIMLX_FA256_FORCE_FUSED"] = os.environ.get(
                "MINIMLX_FA256_FORCE_FUSED"
            )
            os.environ["MINIMLX_FA256_FORCE_FUSED"] = "1"
        self._install_fa256_force_fused()
        try:
            from mlx_lm.models.cache import KVCache

            self._kv_cache_step_before = KVCache.step
        except (ImportError, AttributeError):
            self._kv_cache_step_before = None
        self._install_full_attention_kv_cache_growth()
        if os.environ.get("MINIMLX_FULL_ATTENTION_DECODE_FORCE_FUSED") == "1":
            from ..kernels.fa_decode import install_full_attention_decode_force_fused

            install_full_attention_decode_force_fused()
        self._install_ane_projection()
        self._install_ane_gdn_stage()
        self._greedy_skip_logsumexp = (
            os.environ.get("MINIMLX_GREEDY_SKIP_LOGSUMEXP") == "1"
        )
        self._prefill_keep_allocator_cache = (
            os.environ.get("MINIMLX_PREFILL_KEEP_ALLOCATOR_CACHE") == "1"
        )
        if os.environ.get("MINIMLX_ENABLE_PREFIX_CACHE") == "1":
            from ..prompt_cache import PrefixCache

            max_bytes = int(
                os.environ.get("MINIMLX_PREFIX_CACHE_MAX_BYTES", str(4 * 1024**3))
            )
            self._prefix_cache = PrefixCache(self.model, max_bytes)

    def install_vlm(self) -> None:
        self._install_gdn_decode_conv_silu(self.model)
        from mlx_vlm.vision_cache import VisionFeatureCache

        self._vision_feature_cache = VisionFeatureCache(
            max_size=max(
                1, int(os.environ.get("MINIMLX_VISION_FEATURE_CACHE_SIZE", "20"))
            )
        )

    def generate(self, *args, **kwargs):
        """Run Qwen3.5's specialized generation over the loaded baseline model."""
        return self._generate_optimized(*args, **kwargs)

    def uninstall(self) -> None:
        for name, value in self._environment_before.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        self._environment_before.clear()
        if self._kv_cache_step_before is not None:
            try:
                from mlx_lm.models.cache import KVCache

                KVCache.step = self._kv_cache_step_before
            except (ImportError, AttributeError):
                pass
            self._kv_cache_step_before = None
