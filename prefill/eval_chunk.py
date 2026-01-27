from collections import defaultdict

from eval import get_data_list, set_ratios

if __name__ == "__main__":
    from args import args
    from model import ModelKVzip

    from data import DataWrapper, load_dataset_all
    from utils import Evaluator, TimeStamp, save_result, set_gen_length

    args.tag += f"_chunk{args.prefill_chunk//1000}k_w{args.window_size}"
    print(f"tag: {args.tag}")

    args.kv_type = "retain"  # RetainCache enables efficient evaluation across multiple compression ratios with a single prefilling.
    model = ModelKVzip(args.model, kv_type=args.kv_type, gate=args.weight_path)

    for args.data in get_data_list(args.data, model.name):
        dataset = load_dataset_all(args.data, model.tokenizer)  # list of data
        dataset = DataWrapper(args.data, dataset, model)
        set_gen_length(args.data, model)

        tt = TimeStamp(True)
        max_idx = min(args.idx + args.num, len(dataset))
        print("=" * 80, f"\nStart evaluation with {args.idx}~{max_idx} samples")

        for data_idx in range(args.idx, max_idx):
            kv = dataset.prefill_context(data_idx, do_score=False)
            inputs, info = dataset.generate_answer(data_idx, kv, prob=False)
            eval = Evaluator(model, inputs, info)
            del kv

            outputs = defaultdict(list)
            for t, ratio in enumerate(set_ratios()):
                kv = dataset.prefill_context(
                    data_idx,
                    load_score=args.level == "head",
                    prefill_chunk=args.prefill_chunk,
                    window_size=args.window_size,
                    chunk_ratio=ratio,
                    level=args.level,
                    do_score=False,
                )

                thres, ratio_true = 0, 0
                results = eval(kv, generate=True)  # generation

                for fmt, v in results.items():
                    outputs[fmt].append(
                        [[ratio, round(ratio_true, 4), round(thres, 4)], v]
                    )

                del kv

            save_result(model.name, args, outputs, data_idx)

            tt(f"[{args.data}-{data_idx}]\n")
            del inputs, info, eval
        print("Finished.")
