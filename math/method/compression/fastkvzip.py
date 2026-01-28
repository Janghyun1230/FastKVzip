# ==============================================================================
# Official implementation of "Fast KVzip: Efficient and Accurate LLM Inference with Gated KV Eviction"
# Authors: Jang-Hyun Kim, Dongyoon Han, Sangdoo Yun
# Affiliation: NAVER AI Lab
# Paper: https://arxiv.org/abs/2601.17668
# ==============================================================================
import torch


class FastKVzip:
    def __init__(
        self,
        budget=128,
        window_size=8,
        first_tokens=4,
        n_layers=0,
        n_heads_kv=0,
        gate=None,
        device="cuda",
        **kwargs,
    ):
        assert budget - window_size > 0, "budget must be greater than window_size"
        self.budget = budget
        self.window_size = window_size
        self.first_tokens = first_tokens

        self.n_layers = n_layers
        self.n_heads_kv = n_heads_kv
        self.budget_total = budget * n_layers * n_heads_kv
        self.device = device

        self.gate = gate
        self._init_score()

    def _init_score(self):
        self.score = [
            [torch.zeros((1, 0), device=self.device) for _ in range(self.n_heads_kv)]
            for _ in range(self.n_layers)
        ]
        self.hidden_cache = [None] * self.n_layers

        self.pad_window = torch.ones(self.window_size, device=self.device)

    def _update_hidden_cache(self, hidden_states: torch.Tensor, layer_idx: int):
        if self.hidden_cache[layer_idx] is None:
            self.hidden_cache[layer_idx] = hidden_states
        else:
            # [batch, seq_len, head_dim]
            self.hidden_cache[layer_idx] = torch.cat(
                [self.hidden_cache[layer_idx], hidden_states], dim=1
            )

    def _update_score(self, layer_idx: int):
        score = self.gate[layer_idx](self.hidden_cache[layer_idx])
        if self.score[layer_idx][0].size(-1) == 0:
            score[..., : self.first_tokens] = 1.0  # maximum score to retain

        for i in range(self.n_heads_kv):
            self.score[layer_idx][i] = torch.cat(
                [self.score[layer_idx][i], score[:, i]], dim=-1
            )
        self.hidden_cache[layer_idx] = None  # reset hidden cache

    def __call__(self, key_cache, value_cache, info):
        n_total = sum([k.shape[0] for k in key_cache])  # seq x dim

        if n_total <= self.budget_total:
            return key_cache, value_cache, info
        else:
            score_pad_window = []
            for l in range(self.n_layers):
                # assume bsz=1
                score_pad_window.append(
                    [
                        torch.cat([s[0, : -self.window_size], self.pad_window])
                        for s in self.score[l]
                    ]
                )

            n_remove = n_total - self.budget_total
            score_total = torch.cat([torch.cat(s) for s in score_pad_window])
            thres = torch.sort(score_total, descending=False).values[n_remove]

            zero = torch.tensor([0], dtype=torch.int32, device=self.device)
            for l in range(self.n_layers):
                valid_layer = []
                for i in range(self.n_heads_kv):
                    valid = score_pad_window[l][i] >= thres

                    self.score[l][i] = self.score[l][i][:, valid]
                    info["len_k"][l][i] = self.score[l][i].size(-1)
                    valid_layer.append(valid)

                info["cu_len_k"][l] = info["len_k"][l].cumsum(0, dtype=torch.int32)
                info["cu_len_k"][l] = torch.cat([zero, info["cu_len_k"][l]])
                info["max_len_k"][l] = info["len_k"][l].max()

                valid_layer = torch.cat(valid_layer)
                key_cache[l] = key_cache[l][valid_layer].contiguous()
                value_cache[l] = value_cache[l][valid_layer].contiguous()

            return key_cache, value_cache, info
