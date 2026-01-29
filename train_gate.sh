model="${1:-Qwen/Qwen2.5-7B-Instruct-1M}"
device="${2:-0}"

cd kvzip
CUDA_VISIBLE_DEVICES=$device python -B feature.py -m $model -g ""

cd ..
CUDA_VISIBLE_DEVICES=$device python -B optim.py -m $model
