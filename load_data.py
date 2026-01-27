import os

import torch
from datasets import load_dataset


def load_folder(
    path="/root/code/data/",
    folder="scbench_summary_tiny",
    input_name="hidden",
    ranges=(0, 1),
    add_self_score=False,
):
    inputs, scores = [], []
    for i in range(ranges[0], ranges[1]):
        path_hidden = os.path.join(path, folder, f"{input_name}_{i}.pt")
        path_score = os.path.join(path, folder, f"score_{i}.pt")
        if os.path.isfile(path_score):
            feat = torch.load(path_hidden)
            score = torch.load(path_score)
            if add_self_score:
                selfscore = torch.load(
                    os.path.join(path, folder, f"selfscore_w32_{i}.pt")
                )
                score = torch.maximum(score, selfscore)

            inputs.append(feat)
            scores.append(score)
        print(f"{i}/{ranges[1]-ranges[0]}", end="\r")

    inputs = torch.cat(inputs, dim=-2)
    scores = torch.cat(scores, dim=-1)
    print(f"loaded {folder}/{input_name}, {inputs.shape}")
    return inputs, scores


def load_data(
    path="/root/code/data/",
    input_name="hidden",
    add_self_score=False,
):
    folders = [
        # ("fineweb_10k", (0, 15)),
        # ("fineweb_10k_cat", (0, 2)),
        ("fineweb_10k", (0, 29)),
        ("fineweb_10k_cat", (0, 5)),
    ]
    inputs, scores = [], []
    print(f"Loading training data from {path}/{input_name}..")
    for folder, ranges in folders:
        input, score = load_folder(path, folder, input_name, ranges, add_self_score)
        inputs.append(input)
        scores.append(score)
    inputs = torch.cat(inputs, dim=-2)
    scores = torch.cat(scores, dim=-1)

    print(f"Total x: {inputs.shape}, y: {scores.shape}, {inputs.dtype}")
    return inputs, scores


def split_fn(inputs, scores):
    n = inputs.shape[-2]
    indices = torch.arange(n)
    test_mask = (indices + 1) % 100 == 0  # every 100th sample (1-based)
    train_mask = ~test_mask

    x_train, y_train = inputs[..., train_mask, :], scores[..., train_mask]
    x_test, y_test = inputs[..., test_mask, :], scores[..., test_mask]
    print(f"Split x_train: {x_train.shape}, x_test: {x_test.shape}")

    return x_train, x_test.cuda(), y_train, y_test.cuda()


if __name__ == "__main__":
    fw = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train")
    print(fw)

    from pdb import set_trace

    set_trace()
