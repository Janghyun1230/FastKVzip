# Fast KVzip: Efficient and Accurate LLM Inference with Gated KV Eviction

[[Paper](http://arxiv.org/abs/2601.17668)]


## Installation

```bash
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
cd csrc
make
cd prefill
pip install -r requirements.txt
```

## Prefilling-intensive tasks
```bash
cd prefill
python -B eval_chunk.py -w gate -m $MODEL -d all 
```
- Results will be saved at the ```./prefill/results``` folder. 
- For other baselines, please refer to `run.sh`.

To get scores,
```bash
python -B -m results.parse -m qwen2.5-7b-instruct-1m_gate_chunk16k_w4096
```
- See `./prefill/results/parse.py` for more details.


## Decoding-intensive tasks
Please install required packages before evalution:
```bash
pip install -r requirements.txt
```
We borrowed the evaluation source codes from [R-KV](https://github.com/Zefan-Cai/R-KV).

```bash
cd math
python -B run_math.py --method fastkvzip --kv_budget 4096 --model_path Qwen/Qwen3-8B --dataset_name aime24 --seed 0
```
- See `source run.sh` for reproducing other baselines.
- Results will be saved at the ```./math/results``` folder. 
- To get scores,
```bash
source eval.sh
```
- See `./math/evaluation/eval_math.py` for more details.

## Train gates
```bash
source train_gate.sh $model_name
```
- Results will be save at the ```./result_gate``` folder.
- After training gates, please corrects the `file_path` in the `get_gate_weight` function from `prefill/attention/gate.py` and `math/method/load_gate.py`.

## Citation
```bibtex
@article{kim2026fastkvzip,
        title={Fast KVzip: Efficient and Accurate LLM Inference with Gated KV Eviction}, 
        author={Jang-Hyun Kim and Dongyoon Han and Sangdoo Yun},
        year={2026},
        journal={arXiv preprint arXiv:2601.17668},
}
```

