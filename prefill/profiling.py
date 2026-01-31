# ==============================================================================
# Profiling speed and memory efficiency of Fast KVzip
# ==============================================================================
import argparse

from model import ModelKVzip

from utils.func import TimeStamp

parser = argparse.ArgumentParser()
parser.add_argument(
    "-r",
    "--ratio",
    default=0.3,
    type=float,
    help="compression ratio (1 - eviction ratio)",
)
parser.add_argument("-n", "--model_name", default="Qwen/Qwen2.5-7B-Instruct-1M")
parser.add_argument("-p", "--prefill_len", default=-1, type=int)
args = parser.parse_args()

## Load Model
model = ModelKVzip(args.model_name, kv_type="evict")
model.gen_kwargs["max_new_tokens"] = 128

## Load Data
with open("./data/repo.txt", "r") as file:
    context = file.read()
queries = [
    "What must max_num_tokens be a multiple of when creating a cache?",
    "What bit ranges are allowed for keys and values in quantized cache layers?",
    "Which C++/CUDA file handles the implementation of dequant_cache_paged?",
]
queries = [q + "\nAnswer without explanation." for q in queries]
answers = ["256", "From 2 to 8 bits", "exllamav3/exllamav3_ext/cache/q_cache.cu"]

## Prepare context ids
ctx_ids = model.encode(context)
if args.prefill_len > 0:
    ctx_ids = ctx_ids[:, : args.prefill_len]

# Time and memory profiling
stamp = TimeStamp(verbose=True, unit="ms")

## Chunked-prefill-evict
print(
    f"\nPrefilling context length of {ctx_ids.shape[-1]} with {args.ratio} compression ratio"
)
kv = model.prefill(ctx_ids, chunk_ratio=args.ratio)
stamp(f"KV cache size: {kv._mem()} GB")

## Decoding
print("-" * 70)
for q, a in zip(queries, answers):
    query_ids = model.apply_template(q)
    output = model.generate(query_ids, kv=kv, update_cache=False)  # reset kv cache
    print(model.decode(query_ids), output, f"\n(Ground-truth: {a})")

    q_len = query_ids.shape[1]
    dec_len = model.encode(output).shape[1] + 1  # eos token
    stamp(
        f"Decoding time ({q_len} query, {dec_len} decoded)",
        denominator=1,
    )
    print("-" * 70)
