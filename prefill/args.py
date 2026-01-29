import argparse

parser = argparse.ArgumentParser(description="")
# Method
parser.add_argument("-g", "--gate_path_or_name", type=str, default="fastkvzip")
parser.add_argument("--prefill_chunk", type=int, default=16000)
parser.add_argument("--window_size", type=int, default=4096)
parser.add_argument(
    "-r", "--ratio", type=float, default=0.3, help="compression ratio (= retained/full)"
)
parser.add_argument(
    "--kv_type",
    type=str,
    default="retain",
    choices=["evict", "retain"],
    help="retain: full cache in storage for effcient evaluation over multiple compression ratios",
)
parser.add_argument(
    "--level",
    type=str,
    default="pair",
    choices=["pair", "pair-head", "pair-layer", "head"],
    help="pair-head/layer: uniform head/layer-budget. head: context-independent head-level eviction.",
)
# Model and Data
parser.add_argument("-m", "--model", type=str, default="Qwen/Qwen2.5-7B-Instruct-1M")
parser.add_argument(
    "-d",
    "--data",
    type=str,
    default="squad",
    help="check the dataset list in data/load.py (e.g., squad, scbench_kv)",
)
parser.add_argument("--idx", type=int, default=0, help="the index of a data example")
parser.add_argument(
    "--num", type=int, default=100, help="the total number of eval data"
)
parser.add_argument("--tag", type=str, default="", help="evaluation folder name tag")
args = parser.parse_args()


if args.tag:
    args.tag = f"_{args.tag}"

if args.gate_path_or_name:
    args.tag = "_" + args.gate_path_or_name.split("/")[-1] + args.tag
