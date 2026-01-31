# ==============================================================================
# Benchmark evaluation with chunked-prefill-evict (MRCR)
# ==============================================================================

from collections import defaultdict
from difflib import SequenceMatcher


def grade(response: str, answer: str, random_string_to_prepend: str) -> float:
    """Compare response and answer using SequenceMatcher ratio"""
    if not response.startswith(random_string_to_prepend):
        return 0.0
    response = response.removeprefix(random_string_to_prepend)
    answer = answer.removeprefix(random_string_to_prepend)
    return float(SequenceMatcher(None, response, answer).ratio())


def set_ratios():
    """Set compression ratios for evaluation"""
    return [1.0, 0.75, 0.5, 0.4, 0.3, 0.2]


if __name__ == "__main__":
    from args import args
    from model import ModelKVzip

    from data import load_dataset_all
    from utils import TimeStamp, save_result

    args.data = "mrcr"
    args.tag += f"_chunk{args.prefill_chunk//1000}k_w{args.window_size}"
    print(f"tag: {args.tag}")

    model = ModelKVzip(args.model, args.kv_type, args.gate_path_or_name)
    dataset = load_dataset_all(args.data, model.tokenizer, n_data=2400)

    tt = TimeStamp(True)
    max_idx = min(args.idx + args.num, len(dataset))
    print("=" * 80, f"\nStart evaluation with {args.idx}~{max_idx} samples")

    scores_by_ratio = defaultdict(list)
    for data_idx in range(args.idx, max_idx):
        sample = dataset[data_idx]
        ctx_ids = model.encode(sample["prompt"])
        query_ids = model.apply_template(sample["query"])

        outputs = {}
        for t, ratio in enumerate(set_ratios()):
            kv = model.prefill(
                ctx_ids,
                prefill_chunk_size=args.prefill_chunk,
                window_size=args.window_size,
                chunk_ratio=ratio,
                level=args.level,
            )

            print(
                f"# prefill {model.name} mrcr-{data_idx}: "
                f"{len(ctx_ids[0])} tokens, KV cache {kv._mem()} GB, {kv.key_cache[0].dtype}"
            )

            response = model.generate(query_ids, kv=kv)
            score = grade(
                response, sample["answer"], sample["random_string_to_prepend"]
            )
            scores_by_ratio[ratio].append(score)

            outputs[ratio] = {
                "score": round(score, 4),
                "response": response,
                "ground-truth": sample["answer"],
                "n_tokens": sample["n_tokens"],
            }
            del kv

        save_result(model.name, args, outputs, data_idx)
        tt(f"[mrcr-{data_idx}]\n")

    print("\n" + "=" * 70)
    print(f"MRCR Evaluation Results (%) ({args.model}, {args.tag})")
    for ratio in sorted(scores_by_ratio.keys(), reverse=True):
        scores = scores_by_ratio[ratio]
        if scores:
            avg_score = sum(scores) / len(scores)
            print(f"Ratio {ratio}: {avg_score * 100:.2f}")
