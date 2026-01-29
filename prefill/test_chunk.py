import torch
from model import ModelKVzip

from data import DataWrapper, load_dataset_all
from utils import Evaluator, TimeStamp

if __name__ == "__main__":
    from args import args

    model = ModelKVzip(
        args.model, kv_type=args.kv_type, gate_path_or_name=args.gate_path_or_name
    )

    dataset = load_dataset_all(args.data, model.tokenizer)  # list of data
    dataset = DataWrapper(args.data, dataset, model)

    tt = TimeStamp(verbose=True)  # for time measurement

    kv = dataset.prefill_context(args.idx, do_score=False)
    inputs, info = dataset.generate_answer(args.idx, kv)
    eval = Evaluator(model, inputs, info, verbose=True)
    del kv
    tt("[get full cache answer]")

    kv = dataset.prefill_context(
        args.idx,
        load_score=args.level == "head",
        prefill_chunk=args.prefill_chunk,
        window_size=args.window_size,
        chunk_ratio=args.ratio,
        level=args.level,
        do_score=False,
    )
    tt("[chunked prefill]")

    for task in info.keys():
        tt.set()
        eval.generation(kv, task)  # compare generation results (full vs evicted cache)
        tt(f"[generation at ratio {args.ratio}]")
        eval.forward(kv, task)  # compare output probabilites on answers
