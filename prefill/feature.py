import torch
from model import ModelKVzip

from data import DataWrapper, load_dataset_all
from utils import TimeStamp


def size_mb(tensor: torch.Tensor) -> float:
    num_elements = tensor.numel()
    bytes_per_element = tensor.element_size()
    size_mb = (num_elements * bytes_per_element) / (1024**2)
    print(f"Tensor size: {size_mb:.2f} MB")
    return size_mb


if __name__ == "__main__":
    import os

    from args import args

    model = ModelKVzip(args.model, kv_type=args.kv_type)

    folders = [
        ("fineweb_10k", 29),
        ("fineweb_10k_cat", 5),
    ]

    for args.data, args.num in folders:
        path = f"/root/code/data/{model.name}/{args.data}"
        os.makedirs(path, exist_ok=True)

        dataset = load_dataset_all(args.data, model.tokenizer)  # list of data
        dataset = DataWrapper(args.data, dataset, model)
        tt = TimeStamp(verbose=True)  # for time measurement

        for args.idx in range(args.num):
            tag = f"{args.idx}"

            kv = dataset.prefill_context(
                args.idx,
                prefill_chunk=args.prefill_chunk,
                save_hidden=True,
                # do_score=False,
            )
            tt("prefill context and get importance score")

            hidden = torch.stack(kv.hidden_cache, dim=0).squeeze()
            hidden = hidden[..., kv.start_idx : kv.end_idx, :]
            torch.save(hidden, path + f"/hidden_{tag}.pt")

            score = torch.stack(kv.score, dim=0).squeeze()
            torch.save(score.cpu(), path + f"/score_{tag}.pt")
            print(score.shape, hidden.shape, kv.prefill_ids.shape)

            del kv
            tt("Save time")
