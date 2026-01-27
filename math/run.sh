#!/usr/bin/env bash

MODEL=Qwen/Qwen3-8B
DATA=aime24
SEED=0

for kv_budget in 4096 3072 2048; do 
    for method in fastkvzip snapkv rkv streamingllm; do
        python -B run_math.py --method $method --kv_budget $kv_budget --seed $SEED --model_path $MODEL --dataset_name $DATA
    done
done

for max_length in -1 4096 3072 2048; do  # early stopping
    python -B run_math.py --method fullkv --max_length $max_length --seed $SEED --model_path $MODEL --dataset_name $DATA
done

