#!/usr/bin/env bash

for MODEL in Qwen/Qwen2.5-7B-Instruct-1M Qwen/Qwen3-8B; do 
    python -B eval_chunk.py -w gate -m $MODEL -d all  # FastKVzip
    python -B eval.py -w "" -m $MODEL -d all  # KVzip
    python -B eval_chunk.py -w head -m $MODEL -d all  # DuoAttention
    python -B eval_chunk.py -w expect --level pair-head -m $MODEL -d all  # Expected Attention
    python -B eval_chunk.py -w snap --level pair-head -m $MODEL -d all  # SnapKV
done
