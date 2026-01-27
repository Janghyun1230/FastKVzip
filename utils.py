import os
import re

import matplotlib.pyplot as plt
import torch.nn.functional as F
from transformers import AutoConfig

colors = plt.colormaps.get_cmap("tab20")
linestyles = ["-", "--", "-.", ":"] * 10


def lr_schedule(cur, max_step):
    if cur < 0.01 * max_step:
        alpha = 0.01 * cur
    elif 0.5 * max_step <= cur and cur < 0.75 * max_step:
        alpha = 0.5
    elif 0.75 * max_step <= cur:
        alpha = 0.25
    else:
        alpha = 1.0
    return alpha


def loss_fn(pred, target, type="mse"):
    target = target.to(pred.dtype)
    if type == "mse":
        loss = (pred - target).square()
    elif type == "l1":
        loss = (pred - target).abs()
    elif type.startswith("l1_focal"):
        alpha = float(type.split("_")[-1])
        loss = (1 + alpha * target) * (pred - target).abs()
    elif type.startswith("l1_scale"):
        alpha = float(type.split("_")[-1])
        scale = target.amax(-1, keepdim=True)
        loss = (1 + alpha * target / scale) * (pred - target).abs()
    elif type == "cnt":
        loss = F.binary_cross_entropy(pred, target, reduction="none")
    elif type == "binary":
        thres = 0.05
        target = (target >= thres).float()
        loss = F.binary_cross_entropy(pred, target, reduction="none")
    else:
        raise AssertionError("Undefined loss function")

    loss = loss.mean(-1).sum(-1)
    return loss


def acc_fn(pred, target, thres=0.05):
    return ((pred >= thres) == (target >= thres)).float().mean()


def plot(train_losses, test_losses, name="", log=False, path="./result"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    titles = ["Training", "Validation"]
    y_min = min(train_losses.min(), test_losses.min())
    y_max = max(train_losses.max(), test_losses.max())
    if "acc" in name:
        y_min, y_max = 0.0, 1.0

    for ax, losses, title in zip(axes, [train_losses, test_losses], titles):
        for l in range(losses.shape[1]):
            ax.plot(
                losses[:, l],
                label=f"layer {l}",
                color=colors(l % 20),
                linestyle=linestyles[l // 20],
                linewidth=1.5,
                alpha=0.9,
            )

        if log:
            ax.set_yscale("log")
        ax.set_ylim(y_min, y_max)

        ax.set_title(f"{title} — {name}", fontsize=14, pad=10)
        ax.set_xlabel("Step", fontsize=12)
        ax.set_ylabel("Value", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.5)

    # Single legend for both subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=12,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, 1.05),
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{path}/{name}.png", dpi=300, bbox_inches="tight")
    plt.close()


def get_model_id(name: str):
    """We support abbreviated model names such as:
    llama3.1-8b, llama3.2-*b, qwen2.5-*b, qwen3-*b, and gemma3-*b.
    The full model ID, such as "meta-llama/Llama-3.1-8B-Instruct", is also supported.
    """

    match = re.search(r"(\d+)b", name)
    if match is not None:
        size = match.group(1)

    if name == "llama3.1-8b":
        return "meta-llama/Llama-3.1-8B-Instruct"

    elif name.startswith("llama3.2-"):
        assert size in ["1", "3"], "Model is not supported!"
        return f"meta-llama/Llama-3.2-{size}B-Instruct"

    elif name.startswith("qwen2.5-"):
        assert size in ["7", "14"], "Model is not supported!"
        return f"Qwen/Qwen2.5-{size}B-Instruct-1M"

    elif name.startswith("qwen3-"):
        assert size in ["0.6", "1.7", "4", "8", "14", "32"], "Model is not supported!"
        return f"Qwen/Qwen3-{size}B"

    elif name.startswith("gemma3-"):
        assert size in ["1", "4", "12", "27"], "Model is not supported!"
        return f"google/gemma-3-{size}b-it"

    else:
        return name  # Warning: some models might not be compatible and cause errors


def gate_paths(model_name, file_name=""):
    base_path = "/root/code/result_gate/"

    if not file_name:
        model_id = get_model_id(model_name)
        config = AutoConfig.from_pretrained(model_id)
        ngroup = config.num_attention_heads // config.num_key_value_heads
        file_name = f"q{ngroup}_dim16_sink16"

    path = os.path.join(base_path, model_name, file_name + ".pt")
    return path
