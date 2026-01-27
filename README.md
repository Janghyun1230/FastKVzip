## Installation

```bash
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
cd csrc
make
cd prefill
pip install -r requirements.txt
```

## Scripts
### Long-context tasks (Prefilling intensive)
```bash
cd prefill
source run.sh $model_name
```

Results will be saved at the ```./prefill/results``` folder. For eval, (refer to ./prefill/results/parse.py)
```bash
python -B -m results.parse -m [folder_name]
```

### Reasoning tasks (Decoding intensive)
```bash
cd R-KV
source run.sh
```

Results will be saved at the ```./R-KV/results``` folder. For eval, (refer to ./R-KV/evaluation/eval_math.py)
```bash
source eval.sh
```
Please install required packages before evalution:
```bash
pip install -r requirements.txt
```

### Train gates
```bash
source train_gate.sh $model_name
```
Results will be save at the ```./result_gate``` folder.
