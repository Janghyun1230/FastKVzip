import torch
from model import ModelKVzip

from data import DataWrapper, load_dataset_all
from utils import Evaluator, TimeStamp

if __name__ == "__main__":
    from args import args

    model = ModelKVzip(args.model, kv_type=args.kv_type, gate=args.weight_path)

    dataset = load_dataset_all(args.data, model.tokenizer)  # list of data
    dataset = DataWrapper(args.data, dataset, model)

    tt = TimeStamp(verbose=True)  # for time measurement

    kv = dataset.prefill_context(
        args.idx,
        load_score=args.level == "head",
        prefill_chunk=args.prefill_chunk,
        window_size=args.window_size,
    )
    tt("[prefill context and get importance score]")

    inputs, info = dataset.generate_answer(args.idx, kv)
    tt("[get answers and prediction probabilities for evaluation]")
    print()

    kv.prune(args.ratio, args.level)  # evict KV
    eval = Evaluator(model, inputs, info, verbose=True)

    for task in info.keys():
        # tt.set()
        eval.generation(kv, task)  # compare generation results (full vs evicted cache)
        # tt(f"[generation at ratio {args.ratio}]")
        eval.forward(kv, task)  # compare output probabilites on answers
