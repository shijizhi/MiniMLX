"""DeepSeek V4.1 Flash 1017 adapter and SSD checkpoint preparation."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from dataclasses import replace

from ..deepseek41_offload import DeepSeek41TensorLayout, EngramPageStore
from ..api.memory import read_system_memory
from ..offload import (
    Qwen38OffloadSession,
    build_safetensors_offload_index,
    qwen38_should_use_ssd_offload,
)
from ..types import ExecutionPlan, ModelDescriptor
from .mlx_vlm import MLXVLMAdapter


class DeepSeek41Adapter(MLXVLMAdapter):
    """Load the MLX-VLM checkpoint with a prepared mmap-backed SSD index.

    The upstream MLX-VLM loader remains responsible for model execution. This
    adapter owns the storage contract: the checkpoint index is built before
    loading, Engram tensors are exposed as page reads, and the index/session is
    released together with the model. This prevents the large Engram table from
    being accidentally treated as an ordinary dense shard by MiniMLX.
    """

    name = "deepseek41-mlx-vlm"

    def __init__(self) -> None:
        super().__init__()
        self.offload_session: Qwen38OffloadSession | None = None
        self.offload_index: Path | None = None
        self.tensor_layout: DeepSeek41TensorLayout | None = None
        self.engram_store: EngramPageStore | None = None
        self.use_ssd_offload = False

    @classmethod
    def can_load(cls, descriptor: ModelDescriptor) -> bool:
        return bool(descriptor.metadata.get("is_deepseek41_flash"))

    def load(
        self, descriptor: ModelDescriptor, plan: ExecutionPlan | None = None
    ) -> "DeepSeek41Adapter":
        if plan is not None and plan.backend_adapter != "deepseek41-mlx-vlm-runtime":
            raise ValueError(f"DeepSeek41Adapter cannot load backend {plan.backend_adapter!r}")

        root = Path(descriptor.model_dir).expanduser().resolve()
        offload_mode = os.environ.get("MINIMLX_DEEPSEEK41_OFFLOAD", "auto").lower()
        if offload_mode not in {"auto", "0", "1", "true", "false"}:
            raise ValueError("MINIMLX_DEEPSEEK41_OFFLOAD must be auto, 0, 1, true, or false")
        snapshot = read_system_memory()
        auto_offload = qwen38_should_use_ssd_offload(
            int(descriptor.metadata.get("total_weight_bytes") or 0),
            total_memory_bytes=snapshot.total_bytes,
            available_memory_bytes=snapshot.available_bytes,
        )
        self.use_ssd_offload = offload_mode in {"1", "true"} or (
            offload_mode == "auto" and auto_offload
        )
        if not self.use_ssd_offload:
            return super().load(descriptor, plan)

        index_path = root / "model.safetensors.index.json"
        if not index_path.is_file():
            raise RuntimeError(
                "DeepSeek V4.1 Flash requires model.safetensors.index.json for "
                "Engram/DSpark SSD placement"
            )
        try:
            upstream_offload_root = self._ensure_upstream_offload_dir(root)
            self.offload_index = root / ".minimlx-deepseek41-offload.json"
            if not self.offload_index.is_file():
                build_safetensors_offload_index(root, output_path=self.offload_index)
            self.offload_session = Qwen38OffloadSession.open(
                root, index_path=self.offload_index,
                cache_bytes=max(
                    0,
                    int(os.environ.get("MINIMLX_DEEPSEEK41_PAGE_CACHE_BYTES", str(512 * 1024 * 1024))),
                ),
                prefetch_batch_size=max(
                    1, int(os.environ.get("MINIMLX_DEEPSEEK41_PREFETCH_BATCH", "2"))
                ),
            )
            self.tensor_layout = DeepSeek41TensorLayout.from_safetensors_index(index_path)
            if self.tensor_layout.engram:
                self.engram_store = EngramPageStore(
                    self.offload_session, self.tensor_layout.engram,
                    page_rows=max(1, int(os.environ.get("MINIMLX_DEEPSEEK41_ENGRAM_PAGE_ROWS", "4096"))),
                )
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
            self.unload()
            raise RuntimeError(f"DeepSeek V4.1 SSD index preparation failed: {exc}") from exc

        try:
            # mlx-vlm detects this directory's offload_index.json and patches
            # the model's routed-expert modules so forward() pages only the
            # selected experts. MiniMLX's own index remains available for
            # Engram page accounting and diagnostics.
            load_descriptor = replace(
                descriptor, model_dir=str(upstream_offload_root)
            )
            super().load(load_descriptor, plan)
            self.descriptor = descriptor
        except Exception:
            self.unload()
            raise
        return self

    @staticmethod
    def _ensure_upstream_offload_dir(root: Path) -> Path:
        """Build the native MLX-VLM offload layout once and reuse it."""
        target = root / ".minimlx-deepseek41-offload"
        if (target / "offload_index.json").is_file():
            return target
        target.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "mlx_vlm",
            "moe_offload",
            "--build",
            str(root),
            "--out",
            str(target),
        ]
        completed = subprocess.run(
            command,
            cwd=str(root),
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0 or not (target / "offload_index.json").is_file():
            detail = (completed.stderr or completed.stdout or "no diagnostic output").strip()
            raise RuntimeError(
                "mlx-vlm native MoE offload repack failed "
                f"(exit={completed.returncode}): {detail[-2000:]}"
            )
        return target

    def unload(self) -> None:
        super().unload()
        self.engram_store = None
        self.tensor_layout = None
        self.offload_session = None
        self.offload_index = None
        self.use_ssd_offload = False

