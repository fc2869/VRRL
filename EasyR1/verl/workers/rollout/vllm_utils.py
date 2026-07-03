from __future__ import annotations

import inspect
from functools import lru_cache
from typing import Any, Dict, Set


@lru_cache
def _engine_args_params() -> Set[str]:
    try:
        from vllm.engine.arg_utils import EngineArgs
    except Exception:
        return set()
    try:
        return set(inspect.signature(EngineArgs.__init__).parameters)
    except (TypeError, ValueError):
        return set()


def add_engine_arg_if_supported(engine_kwargs: Dict[str, Any], name: str, value: Any) -> None:
    """Add an EngineArgs kwarg only when the current vLLM version supports it."""
    if name in _engine_args_params():
        engine_kwargs[name] = value







