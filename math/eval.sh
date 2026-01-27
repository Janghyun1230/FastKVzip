#!/bin/bash

method="${1:-fastkvzip}"
model="${2:-Qwen3-8B}"
folder="${3:-aime24}"
dataset="${4:-aime24}"

# This will evaluate files where the name includes $method
python -B evaluation/eval_math.py \
    --exp_name "evaluation" \
    --output_dir "." \
    --base_dir "./results/$folder/$model" \
    --dataset $dataset \
    --tag $method
