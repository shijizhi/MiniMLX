"""MiniMLX's model-runtime boundary.

Importing data contracts and checkpoint discovery is intentionally headless;
the MLX runtime is loaded only when callers explicitly request ``Runtime``.
"""

__all__ = ["ModelInstance", "ModelInstanceState", "Runtime"]


def __getattr__(name: str):
    if name == "Runtime":
        from .runtime import Runtime

        return Runtime
    if name in {"ModelInstance", "ModelInstanceState"}:
        from .model_instance import ModelInstance, ModelInstanceState

        return {"ModelInstance": ModelInstance, "ModelInstanceState": ModelInstanceState}[name]
    raise AttributeError(name)
