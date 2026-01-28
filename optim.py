# ==============================================================================
# Official implementation of "Fast KVzip: Efficient and Accurate LLM Inference with Gated KV Eviction"
# Authors: Jang-Hyun Kim, Dongyoon Han, Sangdoo Yun
# Affiliation: NAVER AI Lab
# Paper: https://arxiv.org/abs/2601.17668
# ==============================================================================
import argparse
import math
import os
import re
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from load_data import load_data, split_fn
from utils import acc_fn, gate_paths, loss_fn, lr_schedule, plot


def load_gate(model_name, file_name=""):
    path = gate_paths(model_name, file_name=file_name)

    state_dict = torch.load(path, weights_only=False)["module"]
    dtype = state_dict[0]["q_proj.weight"].dtype

    head_group_outdim, input_dim = state_dict[0]["q_proj.weight"].shape
    head_outdim, _ = state_dict[0]["k_proj.weight"].shape
    output_dim = state_dict[0]["q_norm.weight"].shape[-1]
    nhead = head_outdim // output_dim
    ngroup = head_group_outdim // head_outdim

    m = re.search(r"sink(\d+)", path)
    sink = int(m.group(1)) if m else 0

    modules = []
    for l, weight in enumerate(state_dict):
        module = Weight(
            l,
            input_dim,
            output_dim,
            nhead,
            ngroup,
            dtype,
            sink=sink,
        ).cuda()
        module.load_state_dict(weight)
        modules.append(module)

    print(f"load {path} ({module})")
    return modules


class Weight(nn.Module):
    def __init__(
        self,
        index: int,
        input_dim: int,
        output_dim: int,
        nhead: int,
        ngroup: int,
        dtype,
        sink=16,
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
        self.q_norm = Qwen3RMSNorm(output_dim)  # fp32
        self.k_norm = Qwen3RMSNorm(output_dim)  # fp32
        self.b = nn.Parameter(torch.zeros([nhead, 1, ngroup], dtype=dtype))

        if self.sink != 0:
            self.k_base = nn.Parameter(torch.zeros([nhead, 1, sink, output_dim]))

        self.d = math.sqrt(self.output_dim)

    def forward(self, hidden_states: torch.Tensor):
        nseq = hidden_states.shape[0]  # sequence x dim
        hidden_shape = (nseq, self.nhead, -1, self.output_dim)

        queries = self.q_norm(self.q_proj(hidden_states).view(hidden_shape))
        keys = self.k_norm(self.k_proj(hidden_states).view(hidden_shape))

        queries = queries.transpose(0, 1).transpose(-1, -2)  # head x seq x dim x group
        keys = keys.transpose(0, 1)

        logit = torch.matmul(keys, queries) / self.d  # head x seq x 1 x group

        if self.sink != 0:
            logit += self.b.unsqueeze(2)
            # head x 1 x sink x group
            logit += self.b.unsqueeze(2)
            logit_base = torch.matmul(self.k_base, queries) / self.d
            score = 1 / (1 + torch.exp(logit_base - logit).sum(2, keepdim=True))
        else:
            score = 1 / (1 + torch.exp(-logit))

        score = score.mean(-1)  # head, seq, 1
        return score.squeeze(-1)

    def extra_repr(self):
        # Customize the print output
        repr_str = f"index={self.index}, output_dim={self.output_dim}, nhead={self.nhead}, ngroup={self.ngroup}\n"
        if self.sink != 0:
            repr_str += f"k_base shape: {self.k_base.shape}\n"
        repr_str += f"b shape: {self.b.shape}\n"
        return repr_str


def optimize(args, inputs, scores, test_freq=50, batch_size=1000):
    input_dim = inputs.shape[-1]
    n_layer, nhead, nseq = scores.shape
    x_train, x_test, y_train, y_test = split_fn(inputs, scores)

    modules = nn.ModuleList(
        [
            Weight(
                l,
                input_dim,
                args.dim,
                nhead,
                args.ngroup,
                inputs.dtype,
                sink=args.sink,
            ).cuda()
            for l in range(n_layer)
        ]
    )
    print(modules[0], "x", len(modules))

    optimizer = optim.SGD(modules.parameters(), lr=args.lr, momentum=0.9)

    MAX_STEP = 5000
    n_samples = x_train.shape[-2]
    n_split = n_samples // batch_size
    print(f"\nStart training with {n_samples} samples, {n_split} split")

    s = time.time()
    losses, acc, losses_test, acc_test = [], [], [], []
    for i in range(MAX_STEP):
        if i % test_freq == 0:
            losses.append([None] * n_layer)
            acc.append([None] * n_layer)
            losses_test.append([None] * n_layer)
            acc_test.append([None] * n_layer)

        split = i % n_split
        if split == 0:
            indices = torch.randperm(n_samples, device=x_train.device)
            print("Reshuffled!")

        indices_batch = indices[split * batch_size : (split + 1) * batch_size]
        x = x_train[..., indices_batch, :].cuda()  # layer x sequence x dim
        y = y_train[..., indices_batch].cuda()  # layer x head x sequence

        # Forward + backward pass
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for l in range(n_layer):
            pred = modules[l](x[l])
            loss = loss_fn(pred, y[l], type=args.loss)
            total_loss += loss
            if i % test_freq == 0:
                losses[-1][l] = loss.item()
                acc[-1][l] = acc_fn(pred, y[l]).item()

        total_loss.backward()

        # Manually adjust learning rate
        alpha = lr_schedule(i, MAX_STEP)
        for g in optimizer.param_groups:
            g["lr"] = args.lr * alpha
        optimizer.step()

        if i % test_freq == 0:
            with torch.no_grad():
                for l in range(n_layer):
                    pred_test = modules[l](x_test[l])
                    losses_test[-1][l] = loss_fn(
                        pred_test, y_test[l], type=args.loss
                    ).item()
                    acc_test[-1][l] = acc_fn(pred_test, y_test[l]).item()

            print(
                f"[{i:5d}] loss: {np.mean(losses[-1]):.3f} ({np.mean(losses_test[-1]):.3f})"
                f" acc: {np.mean(acc[-1])*100:.1f}% ({np.mean(acc_test[-1])*100:.1f}%)"
                f" (time: {time.time()-s:.0f}s)"
            )

    losses, acc = np.array(losses), np.array(acc)
    losses_test, acc_test = np.array(losses_test), np.array(acc_test)
    return modules, losses, acc, losses_test, acc_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", "-m", type=str, default="Qwen/Qwen2.5-7B-Instruct-1M"
    )
    parser.add_argument("--dim", "-d", type=int, default=16)
    parser.add_argument("--ngroup", "-q", type=int, default=-1, help="query group size")
    parser.add_argument("--sink", type=int, default=16)
    # fixed
    parser.add_argument("--loss", type=str, default="cnt")
    parser.add_argument("--lr", type=float, default=2e-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    if args.ngroup == -1:
        config = AutoConfig.from_pretrained(args.model)
        args.ngroup = config.num_attention_heads // config.num_key_value_heads

    tag = f"q{args.ngroup}_dim{args.dim}"
    tag += f"_sink{args.sink}" if args.sink >= 0 else ""
    tag += f"_{args.tag}" if args.tag else ""
    tag += f"_{args.loss}" if args.loss != "cnt" else ""
    tag += f"_lr{args.lr}" if args.lr != 2e-1 else ""

    print(f"\nQuery group size: {args.ngroup} ({tag})")
    torch.manual_seed(args.seed)

    name = args.model.split("/")[-1].lower()
    inputs, scores = load_data(path=f"./data/{name}", input_name="hidden")
    modules, losses, acc, losses_test, acc_test = optimize(args, inputs, scores)

    path = f"./result_gate/{name}"
    os.makedirs(path, exist_ok=True)
    torch.save(
        {
            "module": [m.state_dict() for m in modules],
            "loss": losses_test,
            "acc": acc_test,
            "loss_train": losses,
            "acc_train": acc,
        },
        f"{path}/{tag}.pt",
    )
    plot(losses, losses_test, name=f"loss_{tag}", log=True, path=path)
    plot(acc, acc_test, name=f"acc_{tag}", path=path)
    print("Finished!")
