## Prefill-Intensive Tasks

### Reproducing Benchmark Results
```bash
python -B eval_chunk.py -g fastkvzip -m $MODEL_ID -d all 
```
- Results will be saved at the ```./prefill/results``` folder. 
- We provide the implementation of other baselines compared in our paper. Please refer to `run.sh`.
- We release gates for the following ```$MODEL_ID```:
    - Qwen/Qwen2.5-{7,14}B-Instruct-1M 
    - Qwen/Qwen3-{8,14}B
    - Qwen/Qwen3-8B-FP8
    - Qwen/Qwen3-4B-Instruct-2507
    - google/gemma-3-12b-it

> [!Note]  
> - We store full KV cache in memory for evaluation while conducting attention with evicted KV cache (`--kv_type retain`), following KVzip. 
> - To remove the evicted KV features from memory, use `--kv_type evict`.

To get task scores,
```bash
python -B -m results.parse -m qwen2.5-7b-instruct-1m_fastkvzip_chunk16k_w4096 -d all
```
- Please set the folder name of your method for ```-m``` like above. 
- See `./prefill/results/parse.py` for more details.

### Example-Level Analysis
- By running the test code below, you can observe the detailed changes in prediction induced by KV eviction.
```python
python -B test.py --kv_type evict -g fastkvzip -d scbench_kv
```

### Efficiency Measurement
You can measure the memory and decoding speed:
```python
python -B profiling.py -p $context_len -r $compression_ratio
```
- set `-r 1.0` for full cache speed.
