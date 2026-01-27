"""
This package provides efficient decoding-time KV cache compression methods.
"""

__version__ = "0.1.0"

from .kvcache import EvictCache
from .load_gate import load_gate
from .monkeypatch import replace_llama, replace_qwen2, replace_qwen3

__all__ = ["replace_llama", "replace_qwen2", "replace_qwen3", "EvictCache"]
