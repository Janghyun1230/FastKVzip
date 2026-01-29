## Decoding-Intensive Tasks
Please install required packages before evalution:
```bash
pip install -r requirements.txt
```
We borrowed the evaluation source codes from [R-KV](https://github.com/Zefan-Cai/R-KV).

```bash
python -B run_math.py --method fastkvzip --kv_budget 4096 --model_path Qwen/Qwen3-8B --dataset_name aime24 --seed 0
```
- See `source run.sh` for reproducing other baselines.
- Results will be saved at the ```./math/results``` folder. 

To get task scores,
```bash
source eval.sh
```
- See `./math/evaluation/eval_math.py` for more details.
