import argparse
from concurrent.futures import TimeoutError
from parser import *

import numpy as np
from grader import *
from pebble import ProcessPool
from python_executor import PythonExecutor
from tqdm import tqdm

from utils import load_jsonl


def evaluate(
    data_name,
    prompt_type,
    samples: list = None,
    file_path: str = None,
    max_num_samples=None,
    execute=False,
):
    assert samples or file_path, "samples or file_path must be provided"
    if not samples:
        samples = list(load_jsonl(file_path))
    if "idx" in samples[0]:
        samples = {sample["idx"]: sample for sample in samples}.values()
        samples = sorted(samples, key=lambda x: x["idx"])
    else:
        samples = [dict(idx=idx, **sample) for idx, sample in enumerate(samples)]

    if max_num_samples:
        print(f"max_num_samples: {max_num_samples} / {len(samples)}")
        samples = samples[:max_num_samples]

    # parse gt
    for sample in samples:
        sample["gt_cot"], sample["gt"] = parse_ground_truth(sample, data_name)
    params = [
        (idx, pred, sample["gt"])
        for idx, sample in enumerate(samples)
        for pred in sample["pred"]
    ]

    scores = []
    with ProcessPool(max_workers=1) as pool:
        future = pool.map(math_equal_process, params, timeout=10)
        iterator = future.result()
        while True:
            try:
                result = next(iterator)
                scores.append(result)
            except StopIteration:
                break
            except TimeoutError:
                print("Timeout – skipping example")
                result = False
                scores.append(result)
                continue
            except Exception as error:
                print(error.traceback)
                exit()

    idx = 0
    score_mat = []

    lengths = {
        "8k": [0, 0],
        "16k": [0, 0],
        "32k": [0, 0],
        "max": [0, 0],
    }
    for sample in samples:
        sample["score"] = scores[idx : idx + len(sample["pred"])]
        assert len(sample["score"]) == len(sample["pred"])
        score_mat.append(sample["score"])
        idx += len(sample["pred"])

        # print(idx, sample["len"], sum(sample["score"]))

        if sample["len"] < 8912:
            key = "8k"
        elif sample["len"] < 16384:
            key = "16k"
        elif sample["len"] < 32768:
            key = "32k"
        else:
            key = "max"

        lengths[key][0] += sum(sample["score"])
        lengths[key][1] += len(sample["score"])

    max_len = max([len(s) for s in score_mat])

    for i, s in enumerate(score_mat):
        if len(s) < max_len:
            score_mat[i] = s + [s[-1]] * (max_len - len(s))  # pad

    # output mean of each column of scores
    col_means = np.array(score_mat).mean(axis=0)
    mean_score = list(np.round(col_means * 100, decimals=1))

    result_json = {
        "num_samples": len(samples),
        "lengths": lengths,
        "average_len": int(np.mean([s["len"] for s in samples])),
        "correct_len": int(np.mean([s["len"] for s in samples if sum(s["score"]) > 0])),
        "acc": mean_score[0],
    }

    # each type score
    if "type" in samples[0]:
        type_scores = {}
        for sample in samples:
            if sample["type"] not in type_scores:
                type_scores[sample["type"]] = []
            type_scores[sample["type"]].append(sample["score"][-1])
        type_scores = {
            k: np.round(np.array(v).mean() * 100, decimals=1)
            for k, v in type_scores.items()
        }
        type_scores = {
            k: v for k, v in sorted(type_scores.items(), key=lambda item: item[0])
        }
        result_json["type_acc"] = type_scores

    print(result_json)
    return samples, result_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="math")
    parser.add_argument("--prompt_type", type=str, default="tool-integrated")
    parser.add_argument("--file_path", type=str, default=None, required=True)
    parser.add_argument("--max_num_samples", type=int, default=None)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        data_name=args.data_name,
        prompt_type=args.prompt_type,
        file_path=args.file_path,
        max_num_samples=args.max_num_samples,
        execute=args.execute,
    )
