# ==============================================================================
# Example level analysis of KV eviction
# ==============================================================================
from model import ModelKVzip

from data import DataWrapper, load_dataset_all
from utils import Evaluator, TimeStamp

if __name__ == "__main__":
    from args import args

    model = ModelKVzip(args.model, args.kv_type, args.gate_path_or_name)

    dataset = load_dataset_all(args.data, model.tokenizer)  # list of data
    dataset = DataWrapper(args.data, dataset, model)

    tt = TimeStamp(verbose=True)  # for time measurement

    print("\nObtaining full-cache generation results")
    kv = dataset.prefill_context(args.idx)
    inputs, info = dataset.generate_answer(args.idx, kv)
    eval = Evaluator(model, inputs, info, verbose=True)
    del kv
    tt("[obatin full cache answer]")

    print("\nObtaining chunked-prefill-evict generation results")
    kv = dataset.prefill_context(
        args.idx,
        prefill_chunk=args.prefill_chunk,
        window_size=args.window_size,
        chunk_ratio=args.ratio,
        level=args.level,
    )
    tt("[chunked prefill]")

    for task in info.keys():
        tt.set()
        eval.generation(kv, task)  # compare generation results (full vs evicted cache)
        tt(f"[generation at ratio {args.ratio}]")
        eval.forward(kv, task)  # compare output probabilites on answers
