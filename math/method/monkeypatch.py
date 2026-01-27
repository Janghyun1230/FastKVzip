from transformers import Qwen3MoeForCausalLM
from transformers.models.llama import modeling_llama
from transformers.models.qwen2 import modeling_qwen2
from transformers.models.qwen3 import modeling_qwen3

from .modeling import CausalLM_forward, QwenLlamaAttention_forward


def replace_llama():
    modeling_llama.LlamaAttention.forward = QwenLlamaAttention_forward
    modeling_llama.LlamaForCausalLM.forward = CausalLM_forward


def replace_qwen2():
    modeling_qwen2.Qwen2Attention.forward = QwenLlamaAttention_forward
    modeling_qwen2.Qwen2ForCausalLM.forward = CausalLM_forward


def replace_qwen3():
    modeling_qwen3.Qwen3Attention.forward = QwenLlamaAttention_forward
    modeling_qwen3.Qwen3ForCausalLM.forward = CausalLM_forward
