# ==============================================================================
# Official implementation of "Fast KVzip: Efficient and Accurate LLM Inference with Gated KV Eviction"
# Authors: Jang-Hyun Kim, Dongyoon Han, Sangdoo Yun
# Affiliation: NAVER AI Lab
# Paper: https://arxiv.org/abs/2601.17668
# ==============================================================================
import math
import os
import re

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm


class Weight(nn.Module):
    def __init__(
        self,
        index: int,
        input_dim: int,
        output_dim: int,
        nhead: int,
        ngroup: int,
        dtype,
        sink=1,
    ):
        super().__init__()
        self.index = index
        self.output_dim = output_dim
        self.nhead = nhead
        self.ngroup = ngroup
        self.sink = sink

        self.q_proj = nn.Linear(
            input_dim, nhead * ngroup * output_dim, bias=True, dtype=dtype
        )
        self.k_proj = nn.Linear(input_dim, nhead * output_dim, bias=False, dtype=dtype)
        self.q_norm = Qwen3RMSNorm(output_dim)
        self.k_norm = Qwen3RMSNorm(output_dim)
        self.k_base = nn.Parameter(torch.zeros([nhead, 1, sink, output_dim]))
        self.b = nn.Parameter(torch.zeros([nhead, 1, ngroup], dtype=dtype))

        self.d = math.sqrt(self.output_dim)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states.squeeze(0)  # bsz = 1
        nseq = hidden_states.shape[0]  # sequence x dim
        hidden_shape = (nseq, self.nhead, -1, self.output_dim)

        queries = self.q_norm(self.q_proj(hidden_states).view(hidden_shape))
        keys = self.k_norm(self.k_proj(hidden_states).view(hidden_shape))
        queries = queries.transpose(0, 1).transpose(-1, -2)
        keys = keys.transpose(0, 1)

        # head x seq x 1 x group
        logit = torch.matmul(keys, queries) / self.d + self.b.unsqueeze(2)
        # head x 1 x sink x group
        logit_base = torch.matmul(self.k_base, queries) / self.d
        score = 1 / (1 + torch.exp(logit_base - logit).sum(2, keepdim=True))

        score = score.mean(-1)  # n_head, seq, 1
        return score.squeeze(-1).unsqueeze(0)  # bsz x n_head x seq

    def extra_repr(self):
        # Customize the print output
        repr_str = f"index={self.index}, output_dim={self.output_dim}, nhead={self.nhead}, ngroup={self.ngroup}\n"
        if self.sink != 0:
            repr_str += f"k_base shape: {self.k_base.shape}\n"
        repr_str += f"b shape: {self.b.shape}\n"
        return repr_str


def load_gate(model_name="Qwen/Qwen3-8B", file_name="fastkvzip", device="cuda"):
    if not model_name:
        raise AssertionError("Model_name is empty. Please check load_gate.")
    state_dict, gate_id = get_gate_weight(model_name, file_name)

    dtype = state_dict[0]["q_proj.weight"].dtype
    head_group_outdim, input_dim = state_dict[0]["q_proj.weight"].shape
    head_outdim, _ = state_dict[0]["k_proj.weight"].shape
    output_dim = state_dict[0]["q_norm.weight"].shape[-1]
    nhead = head_outdim // output_dim
    ngroup = head_group_outdim // head_outdim

    m = re.search(r"sink(\d+)", gate_id)
    sink = int(m.group(1)) if m else 1

    modules = []
    for l, weight in enumerate(state_dict):
        module = Weight(l, input_dim, output_dim, nhead, ngroup, dtype, sink=sink).to(
            device
        )
        module.load_state_dict(weight)
        modules.append(module)

    print(f"load gate {gate_id} ({module})")
    return modules


def get_gate_id(model_name, file_name="fastkvzip"):
    if file_name == "fastkvzip":
        config = AutoConfig.from_pretrained(model_name)
        if hasattr(config, "text_config"):
            config = config.text_config
        ngroup = config.num_attention_heads // config.num_key_value_heads
        file_name = f"q{ngroup}_dim16_sink16"

    model_name = model_name.split("/")[-1].lower()
    gate_id = os.path.join(model_name, file_name + ".pt")
    return gate_id


def get_gate_weight(model_name, file_name):
    gate_id = get_gate_id(model_name, file_name)

    try:
        file_path = hf_hub_download(
            repo_id="Jang-Hyun/Fast-KVzip", filename=gate_id, repo_type="model"
        )
    except:
        base_path = "~/FastKVzip"  ## Fix this!
        file_path = os.path.join(base_path, "result_gate", file_name)

    # Load the PyTorch tensor/dictionary
    weights = torch.load(file_path, weights_only=False)["module"]
    return weights, gate_id
