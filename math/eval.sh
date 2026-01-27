#!/bin/bash

method="${1:-fastkvzip}"
model="${2:-Qwen3-8B}"
folder="${3:-aime24}"

python -B evaluation/eval_math.py \
    --exp_name "evaluation" \
    --output_dir "." \
    --base_dir "./results/$folder/$model" \
    --dataset aime24 \
    --tag $method

