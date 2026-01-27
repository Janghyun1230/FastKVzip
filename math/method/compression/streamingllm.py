import torch


class StreamingLLM:
    def __init__(
        self,
        budget=128,
        first_tokens=4,
        **kwargs,
    ):
        assert budget - first_tokens > 0, "budget must be greater than first_tokens"
        self.budget = budget
        self.first_tokens = first_tokens

    def __call__(
        self,
        query_states,
        key_states,
        value_states,
        layer_idx,
    ):
        kv_cache_len = key_states.shape[-2]

        if kv_cache_len <= self.budget:
            return key_states, value_states
        else:
            local_window_size = self.budget - self.first_tokens
            # only select the first self.first_tokens tokens and the last local_window_size tokens
            key_states = torch.cat(
                [
                    key_states[:, :, : self.first_tokens],
                    key_states[:, :, -local_window_size:],
                ],
                dim=2,
            )
            value_states = torch.cat(
                [
                    value_states[:, :, : self.first_tokens],
                    value_states[:, :, -local_window_size:],
                ],
                dim=2,
            )
            return key_states, value_states
