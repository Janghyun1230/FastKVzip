#!/usr/bin/env bash

for MODEL in Qwen/Qwen2.5-7B-Instruct-1M Qwen/Qwen3-8B; do 
    python -B eval_chunk.py -g fastkvzip -m $MODEL -d all  # FastKVzip
    python -B eval.py -g "" -m $MODEL -d all  # KVzip
    python -B eval_chunk.py -g head -m $MODEL -d all  # DuoAttention
    python -B eval_chunk.py -g expect --level pair-head -m $MODEL -d all  # Expected Attention
    python -B eval_chunk.py -g snap --level pair-head -m $MODEL -d all  # SnapKV
done
