# Code borrowed from R-KV (https://github.com/Zefan-Cai/R-KV)

import argparse
import json
import random

import numpy as np
import torch
from method import EvictCache, load_gate, replace_llama, replace_qwen2, replace_qwen3
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

dataset2key = {
    "aime24": ["question", "answer"],
    "math": ["problem", "answer"],
}

dataset2max_length = {
    "math": 16384,
    "aime24": 32768,
}


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


prompt_template = "You are given a math problem.\n\nProblem: {question}\n\n You need to solve the problem step by step. First, you need to provide the chain-of-thought, then provide the final answer.\n\n Provide the final answer in the format: Final answer:  \\boxed{{}}"

prompt_answer = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>.\n\n"


def main(args, model, tokenizer, compression_config):
    fout = open(args.save_path, "w")
    print(f"Save folder: {args.save_path}")

    prompts = []
    test_data = []

    if args.max_length <= args.offset:
        args.force = False

    with open(args.dataset_path) as f:
        for index, line in enumerate(f):
            example = json.loads(line)
            question_key = dataset2key[args.dataset_name][0]

            question = example[question_key]
            example["question"] = question
            prompt = prompt_template.format(**example)

            # [Fixed] Use the official HuggingFace template (reduced performance variance)
            messages = [{"role": "user", "content": prompt}]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=args.max_length > args.offset,
            )

            example["prompt"] = prompt
            example["index"] = index
            prompts.append(prompt)
            test_data.append(example)

    if args.force:
        tokenized_prompt_answer = tokenizer(
            prompt_answer, return_tensors="pt", add_special_tokens=False
        )["input_ids"].to("cuda")
        max_length_think = args.max_length - args.offset
        print(f"Do forced answer generation (offset: {args.offset})")
    else:
        max_length_think = args.max_length

    print(f"Start evaluation with {args.dataset_name} with {len(prompts)} examples")
    for i in tqdm(range(0, len(prompts), args.eval_batch_size)):
        batch_prompts = prompts[i : i + args.eval_batch_size]
        tokenized_prompts = tokenizer(
            batch_prompts,
            padding="longest",
            return_tensors="pt",
            add_special_tokens=False,
        ).to("cuda")

        prefill_lengths = tokenized_prompts["attention_mask"].sum(dim=1).tolist()

        # modified KV cache
        past_key_values = EvictCache(model, compression_config)  # define KV cache

        output = model.generate(
            **tokenized_prompts,
            past_key_values=past_key_values,
            max_length=max_length_think,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,  # [Fixed] original R-KV code used the greedy decoding
            return_dict_in_generate=True,
        )

        res = output.sequences
        if args.force and res.shape[-1] == max_length_think:
            if torch.isin(res, model.after_think_token_ids[-1]).any():
                input_ids = res  # continue to generate after thinking
            else:
                input_ids = torch.cat([res, tokenized_prompt_answer], dim=-1)

            output = model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                past_key_values=output.past_key_values,
                max_length=args.max_length,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                return_dict_in_generate=True,
            )

        output = output.sequences

        batch_token_stats = []
        for j in range(output.size(0)):
            total_tokens = int((output[j] != tokenizer.pad_token_id).sum().item())

            prefill = prefill_lengths[j]
            output_tokens = total_tokens - prefill

            batch_token_stats.append(
                {
                    "sample_idx": i + j,
                    "prefill_tokens": prefill,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                }
            )

        batch_outputs = tokenizer.batch_decode(
            [output[j][prefill_lengths[j] :] for j in range(output.size(0))],
            skip_special_tokens=True,
        )

        del output
        torch.cuda.empty_cache()

        for j in range(len(batch_outputs)):
            sample_idx = batch_token_stats[j]["sample_idx"]
            test_data[sample_idx]["prompt"] = batch_prompts[j]
            test_data[sample_idx]["output"] = batch_outputs[j]
            test_data[sample_idx]["prefill_tokens"] = batch_token_stats[j][
                "prefill_tokens"
            ]
            test_data[sample_idx]["output_tokens"] = batch_token_stats[j][
                "output_tokens"
            ]
            test_data[sample_idx]["total_tokens"] = batch_token_stats[j]["total_tokens"]
            test_data[sample_idx]["sample_idx"] = batch_token_stats[j]["sample_idx"]

            fout.write(json.dumps(test_data[sample_idx], ensure_ascii=False) + "\n")

    fout.close()


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-8B")
    # deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
    parser.add_argument("--dataset_name", type=str, default="aime24")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=-1)
    parser.add_argument("--eval_batch_size", type=int, default=1)

    # method config
    parser.add_argument(
        "--method",
        type=str,
        default="fastkvzip",
        choices=[
            "fastkvzip",
            "rkv",
            "snapkv",
            "streamingllm",
            "h2o",
            "fullkv",
        ],
    )
    parser.add_argument(
        "--kv_budget",
        type=int,
        default=4096,
        help="the size of KV cache budget token length",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=128,
        help="the size of local window to retain",
    )
    # Other methods' hyperparameters
    parser.add_argument("--mix_lambda", type=float, default=0.1)
    parser.add_argument("--retain_ratio", type=float, default=0.2)
    parser.add_argument(
        "--retain_direction", type=str, default="last", choices=["last", "first"]
    )
    parser.add_argument("--first_tokens", type=int, default=4)

    # model config
    parser.add_argument("--divide_length", type=int, default=128, help="buffer size")
    parser.add_argument("--folder_tag", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    args.force = True  # force to generate answers if total token lengths become "max_length - args.offset"
    args.offset = 2048

    if args.method != "fastkvzip":
        args.window_size = (
            8  # this is the default setting provided in baselines (better than 128)
        )
    return args


def deduplicate(save_path):
    if os.path.exists(save_path):
        base, ext = os.path.splitext(save_path)
        idx = 1
        while True:
            candidate = f"{base}_{idx}{ext}"
            if not os.path.exists(candidate):
                return candidate
            idx += 1
    return save_path


if __name__ == "__main__":
    import os

    args = parse_arguments()
    set_seed(args.seed)

    args.dataset_path = f"./data/{args.dataset_name}.jsonl"
    if args.max_length == -1:
        args.max_length = dataset2max_length[args.dataset_name]
    else:
        args.folder_tag += f"{args.max_length}"
    print(
        f"Run {args.method} with {args.kv_budget} budget, max len {args.max_length} (seed {args.seed})"
    )

    args.folder_tag = "_" + args.folder_tag if args.folder_tag else ""
    path = f"./results/{args.dataset_name}{args.folder_tag}/{args.model_path.split('/')[-1]}"
    tag = "" if args.kv_budget is None else f"_{args.kv_budget}"
    if args.method in ["rkv", "snapkv", "fastkvzip"]:
        tag += f"_w{args.window_size}"
    tag += f"_s{args.seed}"
    args.save_path = deduplicate(f"{path}/{args.method}{tag}.jsonl")
    os.makedirs(path, exist_ok=True)

    # ====== build compression config ======
    gate = None
    if args.method == "fastkvzip":
        gate = load_gate(args.model_path, device="cuda")

    compression_config = {
        "compression": True,  # compress initial prefilling
        "method": args.method,
        "method_config": {
            "budget": args.kv_budget,
            "window_size": args.window_size,
            "mix_lambda": args.mix_lambda,
            "retain_ratio": args.retain_ratio,
            "retain_direction": args.retain_direction,
            "first_tokens": args.first_tokens,
            "gate": gate,
        },
    }
    model_config = {
        "divide_length": args.divide_length,
    }

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, use_fast=True, padding_side="left"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # apply monkey patch
    if args.method.lower() != "fullkv":
        if "llama" in args.model_path.lower():
            replace_llama()
        elif "qwen3" in args.model_path.lower():
            replace_qwen3()
        elif "qwen" in args.model_path.lower():
            replace_qwen2()
        else:
            raise ValueError(f"Unsupported model: {args.model_path}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        device_map="auto",
        use_cache=True,
        attn_implementation="flash_attention_2",
    )
    model.eval()

    model.config.update(model_config)
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.after_think_token_ids = tokenizer.encode("</think>")

    print(f"{args.model_path}")
    main(args, model, tokenizer, compression_config)
