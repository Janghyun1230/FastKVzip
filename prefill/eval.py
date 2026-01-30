from collections import defaultdict


def set_ratios():
    ratios = [0.75, 0.5, 0.4, 0.3, 0.2]
    return ratios


def get_data_list(dataname, modelname=""):
    short = [
        "squad",  # 203 (502)
        "gsm",  # 86 (120)
    ]
    mid = [
        "scbench_many_shot",  # 26474
        "scbench_mf",  # 149860  (use _mid for qwen3)
        "scbench_choice_eng",  # 119299
        "scbench_qa_eng",  # 122101
        "scbench_repoqa",  # 72499
    ]
    long = [
        "scbench_kv",  # 169428  (use _short for qwen3)
        "scbench_prefix_suffix",  # 112635
        "scbench_summary",  # 117806
        "scbench_vt",  # 124551
    ]
    multi = [
        "scbench_summary_with_needles",  # 113241
        "scbench_repoqa_and_kv",  # 68064
    ]

    if dataname == "short":
        data_list = short
    elif dataname == "mid":
        data_list = mid
    elif dataname == "long":
        data_list = long
    elif dataname == "multi":
        data_list = multi
    elif dataname == "all":
        data_list = long + short + mid
    else:
        data_list = [dataname]

    if any(k in modelname.lower() for k in ("qwen3", "gemma3", "gemma-3")):
        # Evaluate performance on shorter version for models that achieve near zero performance on specific tasks.
        data_list = [
            f"{x}_short" if x == "scbench_prefix_suffix" else x for x in data_list
        ]
        if not "instruct" in modelname.lower():
            data_list = [f"{x}_short" if x == "scbench_kv" else x for x in data_list]
            data_list = [f"{x}_mid" if x == "scbench_mf" else x for x in data_list]

    print(data_list)
    return data_list


if __name__ == "__main__":
    from args import args
    from attention.gate import load_gate
    from model import ModelKVzip

    from data import DataWrapper, load_dataset_all
    from utils import Evaluator, TimeStamp, save_result, set_gen_length

    if args.gate_path_or_name:
        args.tag += f"_w{args.window_size}"
        print(f"tag: {args.tag}")

    args.kv_type = "retain"  # RetainCache enables efficient evaluation across multiple compression ratios with a single prefilling.
    model = ModelKVzip(
        args.model, kv_type=args.kv_type, gate_path_or_name=args.gate_path_or_name
    )

    for args.data in get_data_list(args.data, model.name):
        dataset = load_dataset_all(args.data, model.tokenizer)  # list of data
        dataset = DataWrapper(args.data, dataset, model)
        set_gen_length(args.data, model)

        tt = TimeStamp(True)
        max_idx = min(args.idx + args.num, len(dataset))
        print("=" * 80, f"\nStart evaluation with {args.idx}~{max_idx} samples")

        for data_idx in range(args.idx, max_idx):
            kv = dataset.prefill_context(
                data_idx,
                load_score=args.level == "head",
                window_size=args.window_size,
                do_score=True,
            )
            inputs, info = dataset.generate_answer(data_idx, kv, prob=False)
            eval = Evaluator(model, inputs, info)

            outputs = defaultdict(list)
            for ratio in set_ratios():
                thres, ratio_true = kv.prune(ratio, args.level)
                results = eval(kv, generate=True)  # generation

                for fmt, v in results.items():
                    outputs[fmt].append(
                        [[ratio, round(ratio_true, 4), round(thres, 4)], v]
                    )

            save_result(model.name, args, outputs, data_idx)

            tt(f"[{args.data}-{data_idx}]\n")
            del kv, inputs, info, eval
        print("Finished.")
