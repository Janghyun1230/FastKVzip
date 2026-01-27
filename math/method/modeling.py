from typing import Callable, List, Optional, Tuple, Union

import torch
from flash_attn import flash_attn_varlen_func
from method.utils import TimeStamp
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
from transformers.processing_utils import Unpack
from transformers.utils import logging

logger = logging.get_logger(__name__)


def QwenLlamaAttention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    if isinstance(self, Qwen3Attention):
        query_states = self.q_norm(
            self.q_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        key_states = self.k_norm(
            self.k_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
    else:
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

        # =============== Enable Cache ============
        if past_key_value.method in ["rkv", "h2o", "snapkv"]:
            if not hasattr(past_key_value, "query_cache"):
                past_key_value.query_cache = {}

            if self.layer_idx not in past_key_value.query_cache:
                # prefill stage
                bsz, n_heads, _, head_dim = query_states.shape
                past_key_value.query_cache[self.layer_idx] = torch.empty(
                    bsz, n_heads, 0, head_dim
                )
                past_key_value.query_cache[self.layer_idx] = query_states[
                    :, :, -past_key_value.method_config["window_size"] :, :
                ]
            else:
                # Add current query to cache
                past_key_value.query_cache[self.layer_idx] = torch.cat(
                    (past_key_value.query_cache[self.layer_idx], query_states), dim=2
                )  # [batch, n_q_heads, seq_len, head_dim]

                # Keep only window_size most recent queries
                window_size = past_key_value.method_config["window_size"]
                if past_key_value.query_cache[self.layer_idx].shape[-2] > window_size:
                    past_key_value.query_cache[self.layer_idx] = (
                        past_key_value.query_cache[self.layer_idx][
                            :, :, -window_size:, :
                        ]
                    )
            cached_queries = past_key_value.query_cache[self.layer_idx]
        else:
            cached_queries = None

        if past_key_value.method in ["fastkvzip"]:
            past_key_value.compressor._update_hidden_cache(
                hidden_states, self.layer_idx
            )
            if past_key_value.do_compress:
                past_key_value.compressor._update_score(self.layer_idx)
        # =============== Enable Cache end ===============

        key_states, value_states = past_key_value.update(
            key_states,
            value_states,
            self.layer_idx,
            cache_kwargs,
        )
        if past_key_value.do_compress and not past_key_value.flatten:
            past_key_value.compress(  # compress kv cache
                cached_queries,  # Use cached queries instead of current query
                key_states,
                value_states,
                self.layer_idx,
            )

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        if self.config._attn_implementation == "sdpa" and kwargs.get(
            "output_attentions", False
        ):
            logger.warning_once(
                "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]

    if past_key_value.flatten:  # attention with flatten KV cache
        bsz, q_len, _ = hidden_states.size()
        query_states, key_states, value_states, info = past_key_value.prepare(
            query_states, key_states, value_states, self.layer_idx
        )

        # bsz x head x seq, group, dim
        attn_output = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=info["cu_len_q"],
            cu_seqlens_k=info["cu_len_k"],
            max_seqlen_q=info["max_len_q"],
            max_seqlen_k=info["max_len_k"],
            dropout_p=0.0 if not self.training else self.attention_dropout,
            causal=True,
        )
        attn_output = attn_output.view(
            bsz,
            self.config.num_key_value_heads,
            q_len,
            self.num_key_value_groups,
            self.head_dim,
        ).transpose(1, 2)

        attn_weights = None

    else:
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=getattr(self, "sliding_window", None),
            **kwargs,
        )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    return attn_output, attn_weights


def CausalLM_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    **kwargs,
) -> Union[Tuple, CausalLMOutputWithPast]:
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    # =============== Step-level Compression logic start ===============
    past_len = past_key_values.get_seq_length()
    past_key_values.do_compress = past_len % self.config.divide_length == 0

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        **kwargs,
    )

    if past_key_values.do_compress and past_key_values.flatten:
        past_key_values.compress_post()

    hidden_states = outputs[0]
    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    slice_indices = (
        slice(-logits_to_keep, None)
        if isinstance(logits_to_keep, int)
        else logits_to_keep
    )
    logits = self.lm_head(hidden_states[:, slice_indices, :])

    loss = None
    if labels is not None:
        loss = self.loss_function(
            logits=logits,
            labels=labels,
            vocab_size=self.config.vocab_size,
            **kwargs,
        )

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
