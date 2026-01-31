## Prefill-Intensive Tasks

### Reproducing Benchmark Results
```bash
python -B eval_chunk.py -g fastkvzip -m $MODEL_ID -d all 
```
- Results will be saved at the ```./prefill/results``` folder. 
- We provide the implementation of other baselines compared in our paper. Please refer to `run.sh`.
- Available data names are listed in `data/load.py`. For MRCR, please run `eval_chunk_mrcr.py`.
- We release gates for the following ```$MODEL_ID```:
    - Qwen/Qwen2.5-{7,14}B-Instruct-1M 
    - Qwen/Qwen3-{8,14}B
    - Qwen/Qwen3-8B-FP8
    - Qwen/Qwen3-4B-Instruct-2507
    - google/gemma-3-12b-it

> [!Note]  
> - In our experiments, we use `--kv_type retain`, which preserves the full KV cache in memory while performing attention over a reduced KV cache via subsampling, following KVzip.
> - For improved speed and lower peak memory usage, use `--kv_type evict`. This option may cause marginal differences in prediction results due to GPU numerical variability.

To get task scores,
```bash
python -B -m results.parse -m qwen2.5-7b-instruct-1m_fastkvzip_chunk16k_w4096 -d all
```
- Please set the folder name for the method using `-m`, as shown above.
- See `./prefill/results/parse.py` for more details.

### Example-Level Analysis
- To check the detailed changes in predictions induced by KV eviction, run
```python
python -B test.py --kv_type evict -g fastkvzip -d scbench_kv
```

### Efficiency Measurement
You can measure the memory and decoding speed:
```python
python -B profiling.py -p $context_len -r $compression_ratio
```
- Set `-r 1.0` to profile a case using the full KV cache.
