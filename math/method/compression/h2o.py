import torch
import torch.nn as nn

from . import compute_attention_scores


class H2O:
    def __init__(
        self,
        budget=128,
        window_size=8,
        **kwargs,
    ):
        assert budget - window_size > 0, "budget must be greater than window_size"
        self.budget = budget
        self.window_size = 1

    def __call__(
        self,
        query_states,
        key_states,
        value_states,
        layer_idx,
    ):
        head_dim = query_states.shape[-1]
        kv_cache_len = key_states.shape[-2]

        if kv_cache_len <= self.budget:
            return key_states, value_states
        else:
            query_states = query_states[:, :, -1:, :]
            attn_weights = compute_attention_scores(query_states, key_states).squeeze(2)

            attn_weights_sum = (
                nn.functional.softmax(
                    attn_weights[:, :, : -self.window_size],
                    dim=-1,
                    dtype=torch.float32,
                )
                .mean(dim=-2)
                .to(query_states.dtype)
            )

            # shape: (bsz, num_kv_heads, budget - window_size)
            indices = attn_weights_sum.topk(
                self.budget - self.window_size, dim=-1
            ).indices

            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

            k_past_compress = key_states[:, :, : -self.window_size, :].gather(
                dim=2, index=indices
            )
            v_past_compress = value_states[:, :, : -self.window_size, :].gather(
                dim=2, index=indices
            )
            k_cur = key_states[:, :, -self.window_size :, :]
            v_cur = value_states[:, :, -self.window_size :, :]
            key_states = torch.cat([k_past_compress, k_cur], dim=2)
            value_states = torch.cat([v_past_compress, v_cur], dim=2)
            return key_states, value_states
