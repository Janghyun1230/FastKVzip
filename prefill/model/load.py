import re

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def get_model_id(name: str):
    """We support abbreviated model names such as:
    llama3.1-8b, llama3.2-*b, qwen2.5-*b, qwen3-*b, and gemma3-*b.
    The full model ID, such as "meta-llama/Llama-3.1-8B-Instruct", is also supported.
    """

    match = re.search(r"(\d+)b", name)
    if match is not None:
        size = match.group(1)

    if name == "llama3.1-8b":
        return "meta-llama/Llama-3.1-8B-Instruct"

    elif name.startswith("llama3.2-"):
        assert size in ["1", "3"], "Model is not supported!"
        return f"meta-llama/Llama-3.2-{size}B-Instruct"

    elif name.startswith("qwen2.5-"):
        assert size in ["7", "14"], "Model is not supported!"
        return f"Qwen/Qwen2.5-{size}B-Instruct-1M"

    elif name.startswith("qwen3-"):
        assert size in ["0.6", "1.7", "4", "8", "14", "32"], "Model is not supported!"
        return f"Qwen/Qwen3-{size}B"

    elif name.startswith("gemma3-"):
        assert size in ["1", "4", "12", "27"], "Model is not supported!"
        return f"google/gemma-3-{size}b-it"

    else:
        return name  # Warning: some models might not be compatible and cause errors


def load_model(model_name: str, **kwargs):
    model_id = get_model_id(model_name)
    from model.monkeypatch import replace_attn

    replace_attn(model_id)

    config = AutoConfig.from_pretrained(model_id)
    if "Qwen3-" in model_id and "Instruct" not in model_id:
        config.rope_scaling = {
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 32768,
        }
        config.max_position_embeddings = 131072
        print("Max context length extended")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2",
        config=config,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if "llama" in model_id.lower():
        model.generation_config.pad_token_id = tokenizer.pad_token_id = 128004

    if "gemma-3" in model_id.lower():
        model = model.language_model

    model.eval()
    model.name = model_id.split("/")[-1].lower()
    model.name_or_path = model_id
    print(f"\nLoad {model_id} with {model.dtype}")
    return model, tokenizer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="llama3-8b")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model(args.name)
    print(model)

    messages = [
        {
            "role": "user",
            "content": "How many helicopters can a human eat in one sitting?",
        }
    ]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print(input_text)

    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
    outputs = model.generate(input_ids, max_new_tokens=30)
    print(tokenizer.decode(outputs[0]))
