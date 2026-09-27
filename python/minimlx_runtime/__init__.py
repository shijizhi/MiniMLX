"""MiniMLX's model-runtime boundary.

The first implementation deliberately keeps MLX in Python.  Rust/GPUI can
talk to this package through the line protocol in :mod:`minimlx_runtime.api`.
"""

from .runtime import Runtime

__all__ = ["Runtime"]
