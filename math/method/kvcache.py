# ------------------------------------------------------------------------------
# Original Code developed by Jang-Hyun Kim
# Licensed under The MIT License
# GitHub Repository: https://github.com/snu-mllab/KVzip
# ------------------------------------------------------------------------------
from typing import Optional

import torch
from transformers import DynamicCache

try:
    from tiny_api_cuda import update_flatten_view
except:
    pass

from .compression import H2O, R1KV, FastKVzip, SnapKV, StreamingLLM

KV_COMPRESSION_MAP = {
    "rkv": R1KV,
    "snapkv": SnapKV,
    "streamingllm": StreamingLLM,
    "h2o": H2O,
    "fastkvzip": FastKVzip,
}


class EvictCache(DynamicCache):
    """KV cache that evicts KV from the cache before decoding."""

    def __init__(self, model, compression_config):
        DynamicCache.__init__(self)
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.n_layers = model.config.num_hidden_layers
        self.n_heads = model.config.num_attention_heads
        self.n_heads_kv = model.config.num_key_value_heads
        self.n_group_kv = self.n_heads // self.n_heads_kv

        self._seen_tokens = 0

        self.method = compression_config["method"]
        self.do_compress = compression_config["compression"]
        self.method_config = compression_config["method_config"]
        if self.method in KV_COMPRESSION_MAP:
            method_fn = KV_COMPRESSION_MAP[self.method]
            self.compressor = method_fn(
                **self.method_config,
                n_layers=self.n_layers,
                n_heads_kv=self.n_heads_kv,
            )

        # for unstructured KV eviction
        self.flatten = compression_config["method"] == "fastkvzip"
        if self.flatten:
            self.info = {
                "len_k": [
                    torch.zeros(self.n_heads_kv, dtype=torch.int32, device=self.device)
                    for _ in range(self.n_layers)
                ],
                "cu_len_k": [
                    torch.zeros(
                        self.n_heads_kv + 1, dtype=torch.int32, device=self.device
                    )
                    for _ in range(self.n_layers)
                ],
                "max_len_k": [0 for _ in range(self.n_layers)],
            }
            self.cu_head = torch.arange(
                self.n_heads_kv + 1, dtype=torch.int32, device=self.device
            )
            self.cu_len_q = None

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if len(self.key_cache) <= layer_idx:
            return 0
        else:
            return self._seen_tokens

    def _mem(self):
        """Returns the memory usage of the cache in GB."""
        mem = 0
        for i in range(self.n_layers):
            mem += self.key_cache[i].numel() * self.key_cache[i].element_size()
        mem *= 2  # key + value
        return round(mem / 10**9, 1)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs=dict(),
    ):
        """Update KV cache and return"""
        if layer_idx == 0:
            cache_position = cache_kwargs.get("cache_position", None)
            if cache_position is not None:
                seen_token = cache_position.size(-1)
            else:
                seen_token = key_states.size(-2)
            self._seen_tokens += seen_token

        # Update the cache
        if self.flatten:
            _, _, seq, dim = key_states.shape
            key_states = key_states.contiguous().view(-1, dim)
            value_states = value_states.contiguous().view(-1, dim)

            if len(self.key_cache) <= layer_idx:
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
            else:
                self.key_cache[layer_idx] = update_flatten_view(
                    self.key_cache[layer_idx],
                    key_states,
                    self.info["len_k"][layer_idx],
                    self.info["cu_len_k"][layer_idx],
                )
                self.value_cache[layer_idx] = update_flatten_view(
                    self.value_cache[layer_idx],
                    value_states,
                    self.info["len_k"][layer_idx],
                    self.info["cu_len_k"][layer_idx],
                )

            if layer_idx == 0:
                self.cu_len_q = seq * self.cu_head
            self.info["cu_len_k"][layer_idx] += self.cu_len_q
            self.info["len_k"][layer_idx] += seq
            self.info["max_len_k"][layer_idx] += seq

        else:
            if len(self.key_cache) <= layer_idx:
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
            else:
                self.key_cache[layer_idx] = torch.cat(
                    [self.key_cache[layer_idx], key_states], dim=-2
                )
                self.value_cache[layer_idx] = torch.cat(
                    [self.value_cache[layer_idx], value_states], dim=-2
                )

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def compress(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ):
        self.key_cache[layer_idx], self.value_cache[layer_idx] = self.compressor(
            query_states,
            key_states,
            value_states,
            layer_idx,
        )
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def compress_post(self):
        self.key_cache, self.value_cache, self.info = self.compressor(
            self.key_cache, self.value_cache, self.info
        )

    def prepare(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ):
        """Subsample KV and flatten features for var_len FlashAttention"""
        bsz, _, q_len, dim = query_states.shape
        query_states = (
            query_states.view(bsz, self.n_heads_kv, self.n_group_kv, q_len, dim)
            .transpose(2, 3)
            .contiguous()
        )  # bsz x head x seq, group, dim

        info = {
            "cu_len_q": self.cu_len_q,
            "cu_len_k": self.info["cu_len_k"][layer_idx],
            "max_len_q": q_len,
            "max_len_k": self.info["max_len_k"][layer_idx],
        }

        return (
            query_states.view(-1, self.n_group_kv, dim),
            key_states.view(-1, 1, dim),
            value_states.view(-1, 1, dim),
            info,
        )
