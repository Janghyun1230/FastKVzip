#!/usr/bin/env bash

for MODEL in qwen2.5-14b gemma3-12b Qwen/Qwen3-8B-FP8; do  # qwen2.5-7b qwen2.5-14b qwen3-8b Qwen/Qwen3-8B-FP8 gemma3-12b
    python -B eval.py -w "" -m $MODEL -d all  # kvzip
    python -B eval_chunk.py -w gate -m $MODEL -d all
    python -B eval_chunk.py -w head -m $MODEL -d all
    python -B eval_chunk.py -w expect --level pair-head -m $MODEL -d all
    python -B eval_chunk.py -w snap --level pair-head -m $MODEL -d all
done
