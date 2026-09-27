"""Optional fixed-shape Core ML/ANE projection backend.

This module is intentionally opt-in. It is an experiment for long prefill;
unsupported shapes and all decode calls remain on the normal MLX path.
"""

from __future__ import annotations

import ctypes
import os
import sys
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np


class ANEProjection:
    def __init__(self, library: str, model: str, tokens: int, input_dim: int, output_dim: int):
        self.tokens = tokens
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._lib = ctypes.CDLL(library)
        self._lib.minimlx_ane_create.argtypes = [ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
        self._lib.minimlx_ane_create.restype = ctypes.c_void_p
        self._lib.minimlx_ane_predict.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self._lib.minimlx_ane_predict.restype = ctypes.c_int32
        self._lib.minimlx_ane_destroy.argtypes = [ctypes.c_void_p]
        self._lib.minimlx_ane_destroy.restype = None
        self._handle = self._lib.minimlx_ane_create(
            os.fsencode(model), tokens, input_dim, output_dim
        )
        if not self._handle:
            raise RuntimeError(f"failed to load ANE model: {model}")
        self._calls = 0
        self._total_ms = 0.0

    def __del__(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle:
            self._lib.minimlx_ane_destroy(handle)
            self._handle = None

    def __call__(self, value: Any) -> Any:
        if value.ndim != 3 or value.shape[0] != 1 or value.shape[1] != self.tokens:
            raise ValueError("ANE projection received an unsupported shape")
        if value.shape[2] != self.input_dim:
            raise ValueError("ANE projection received an unsupported input width")
        # MLX's bfloat16 does not expose a NumPy PEP 3118 buffer. Convert at
        # the explicit runtime boundary; the Core ML model consumes float16.
        value = value.astype(mx.float16)
        mx.eval(value)
        host_input = np.asarray(value).astype(np.float16, copy=False).reshape(
            self.tokens, self.input_dim
        )
        host_output = np.empty((self.tokens, self.output_dim), dtype=np.float16)
        start = time.perf_counter()
        status = self._lib.minimlx_ane_predict(
            self._handle,
            host_input.ctypes.data_as(ctypes.c_void_p),
            host_output.ctypes.data_as(ctypes.c_void_p),
        )
        if status != 0:
            raise RuntimeError(f"ANE projection failed with status {status}")
        self._calls += 1
        self._total_ms += (time.perf_counter() - start) * 1000.0
        if os.environ.get("MINIMLX_ANE_TRACE") == "1" and self._calls == 1:
            print(
                f"MiniMLX ANE projection hit: tokens={self.tokens}, "
                f"input_dim={self.input_dim}, output_dim={self.output_dim}",
                file=sys.stderr,
                flush=True,
            )
        return mx.array(host_output.reshape(1, self.tokens, self.output_dim))

    def stats(self) -> tuple[int, float]:
        return self._calls, self._total_ms


class ANEGDNStage:
    """One-boundary GDN projection group for long-prefill experiments."""

    def __init__(
        self,
        projection: ANEProjection,
        qkv_dim: int,
        z_dim: int,
        beta_dim: int,
        a_dim: int,
    ):
        self.projection = projection
        self.qkv_dim = qkv_dim
        self.z_dim = z_dim
        self.beta_dim = beta_dim
        self.a_dim = a_dim
        expected = qkv_dim + z_dim + beta_dim + a_dim
        if expected != projection.output_dim:
            raise ValueError("GDN stage output dimensions do not match ANE model")
        self._cached_input: Any | None = None
        self._cached_output: Any | None = None

    def _run(self, value: Any) -> Any:
        if self._cached_input is not value:
            self._cached_input = value
            # Upload and convert the combined result once. Four independent
            # casts after slicing would create four materialization points in
            # the surrounding MLX graph.
            self._cached_output = self.projection(value).astype(value.dtype)
        return self._cached_output

    def _slice(self, value: Any, start: int, end: int) -> Any:
        return self._run(value)[..., start:end]

    def qkv(self, value: Any) -> Any:
        return self._slice(value, 0, self.qkv_dim)

    def z(self, value: Any) -> Any:
        return self._slice(value, self.qkv_dim, self.qkv_dim + self.z_dim)

    def beta(self, value: Any) -> Any:
        start = self.qkv_dim + self.z_dim
        return self._slice(value, start, start + self.beta_dim)

    def a(self, value: Any) -> Any:
        start = self.qkv_dim + self.z_dim + self.beta_dim
        return self._slice(value, start, start + self.a_dim)


def load_from_environment(
    input_dim: int,
    output_dim: int,
    *,
    enable_env: str = "MINIMLX_ENABLE_ANE",
) -> ANEProjection | None:
    """Load the explicit experiment backend, or return None when disabled."""
    if os.environ.get(enable_env) != "1":
        return None
    library = os.environ.get("MINIMLX_ANE_LIBRARY")
    model = os.environ.get("MINIMLX_ANE_MODEL")
    tokens = os.environ.get("MINIMLX_ANE_TOKENS")
    if not (library and model and tokens):
        raise RuntimeError(
            "MINIMLX_ENABLE_ANE=1 requires MINIMLX_ANE_LIBRARY, "
            "MINIMLX_ANE_MODEL, and MINIMLX_ANE_TOKENS"
        )
    return ANEProjection(library, model, int(tokens), input_dim, output_dim)
