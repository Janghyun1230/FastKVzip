import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers.models.qwen3.modeling_qwen3 import repeat_kv


class Head:
    def __init__(self, score):
        self.score = score
        self.name = "head"

    def __call__(self, hidden_states: torch.Tensor):
        nseq = hidden_states.shape[-2]  # sequence x dim
        score = self.score.expand(1, -1, nseq)  # layer x head x seq

        return score


class SnapKV:
    def __init__(self):
        self.name = "snap"
        self.sink = 4
        self.kernel_size = 7

    def compute_attention_scores(self, query_states, key_states):
        batch_size, q_heads, q_len, head_dim = query_states.shape
        kv_heads = key_states.shape[1]
        query_group_size = q_heads // kv_heads

        if query_group_size == 1:
            attn_weights = torch.matmul(
                query_states, key_states.transpose(2, 3)
            ) / math.sqrt(head_dim)
        else:
            query_states = query_states.view(
                batch_size, kv_heads, query_group_size, q_len, head_dim
            )
            key_states = key_states.unsqueeze(2)

            # shape: [batch_size, kv_heads, query_group_size, q_len, kv_len]
            attn_weights = torch.matmul(
                query_states, key_states.transpose(3, 4)
            ) / math.sqrt(head_dim)

            # apply pooling over query_group_size dimension
            attn_weights = attn_weights.amax(dim=2)

        return attn_weights

    def __call__(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
    ) -> torch.Tensor:

        window_size = 32
        if key_states.shape[-2] < 1000:
            window_size = 16

        qlen = query_states.shape[-2]
        if qlen == key_states.shape[-2]:  # initial chunk
            sink = 0
        else:  # for next chunkes maintain sink keys
            sink = self.sink
        key_states = torch.cat(
            [key_states[:, :, :sink], key_states[:, :, -qlen:]], dim=2
        )

        query_states = query_states[:, :, -window_size:, :]
        attn_weights = self.compute_attention_scores(query_states, key_states)

        attn_weights_sum = (
            nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
            .mean(dim=-2)
            .to(query_states.dtype)
        )
        attn_weights_sum = attn_weights_sum[..., sink:]

        scores = F.max_pool1d(
            attn_weights_sum,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            stride=1,
        )
        return scores


class ExpectedAttentionPress:
    # SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
    # SPDX-License-Identifier: Apache-2.0
    # Codes are copied from the NVIDIA KVpress, with minimal modifications.
    # Refer to the code in https://github.com/NVIDIA/kvpress/blob/main/kvpress/presses/expected_attention_press.py

    def __init__(self, model, cov=False):
        self.name = "expect"
        self.n_future_positions: int = 512
        self.n_sink: int = 0  # we retain system prompt's KV, so do not need to set this
        self.use_covariance: bool = cov  # this decreases performance in our experiments
        self.use_vnorm: bool = True
        self.epsilon: float = 0.02  # default value used in KVPress evaluation

        self.config = model.config
        self.rotary_emb = model.model.rotary_emb
        self.head_dim = getattr(
            self.config,
            "head_dim",
            self.config.hidden_size // self.config.num_attention_heads,
        )

    def get_query_statistics(self, queries_pre: torch.Tensor, cache_position):
        """
        Compute the mean and covariance matrix of the queries
        """

        q_len = queries_pre.shape[2]
        # Remove first hidden_states that likely contain outliers
        sink = 4
        query_states = queries_pre[:, :, min(sink, q_len - 1) :]

        # Query mean
        mu = query_states.mean(dim=2, keepdim=True)

        # Query covariance
        cov = None
        if self.use_covariance:
            centered_states = query_states - mu
            cov = (
                torch.einsum("bnsi,bnsj->bnij", centered_states, centered_states)
                / query_states.shape[1]
            )
        mu = mu.squeeze(2)

        # Apply RoPE to the mean and covariance matrix of the queries
        q_len_total = cache_position[-1] + 1  # considering chunked prefill
        mu, cov = self.apply_avg_rope(mu, cov, q_len_total)

        return mu, cov

    def apply_avg_rope(self, mu: torch.Tensor, cov: torch.Tensor, q_len: int):
        position_ids = (
            torch.arange(q_len, q_len + self.n_future_positions)
            .unsqueeze(0)
            .to(mu.device)
        )
        head_dim = self.head_dim
        cos, sin = self.rotary_emb(mu, position_ids)
        cos, sin = cos[0], sin[0]
        Id = torch.eye(head_dim, device=cos.device, dtype=cos.dtype)
        P = torch.zeros((head_dim, head_dim), device=cos.device, dtype=cos.dtype)
        P[head_dim // 2 :, : head_dim // 2], P[: head_dim // 2, head_dim // 2 :] = (
            torch.eye(head_dim // 2),
            -torch.eye(head_dim // 2),
        )
        R = cos.unsqueeze(1) * Id + sin.unsqueeze(1) * P
        R = R.mean(dim=0).to(mu.device)
        mu = torch.matmul(mu, R.T)
        if cov is not None:
            cov = torch.matmul(R, torch.matmul(cov, R.T))
        return mu, cov

    def __call__(
        self,
        queries_pre: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        cache_position: torch.Tensor,
    ) -> torch.Tensor:

        # Remove sink tokens
        assert (
            keys.size(2) > self.n_sink
        ), f"Input should contain more tokens than n_sink={self.n_sink}"
        keys = keys[:, :, self.n_sink :]
        values = values[:, :, self.n_sink :]

        # Compute query statistics
        mean_query, cov_query = self.get_query_statistics(queries_pre, cache_position)

        # Compute scores
        bsz, num_key_value_heads, kv_len, d = keys.shape
        num_key_value_groups = self.config.num_attention_heads // num_key_value_heads

        keys = repeat_kv(keys, num_key_value_groups).transpose(2, 3)
        scores = torch.matmul(mean_query.unsqueeze(2), keys).squeeze(2) / math.sqrt(d)
        if self.use_covariance:
            scores += (
                torch.einsum("bhin, bhij, bhjn->bhn", keys, cov_query, keys) / d / 2
            )
        scores = F.softmax(scores, dim=-1, dtype=torch.float32).to(scores.dtype)

        # Average scores across groups
        scores = scores.view(bsz, num_key_value_heads, num_key_value_groups, kv_len)
        scores = scores.mean(dim=2)

        # Rescale scores by the norm of the values
        if self.use_vnorm:
            scores = (scores + self.epsilon) * values.norm(dim=-1)

        # Add back the sink tokens. Use max score to make sure they are not pruned.
        scores = F.pad(scores, (self.n_sink, 0), value=scores.max().item())

        return scores
