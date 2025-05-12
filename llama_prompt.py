from typing import Callable, List, Optional, Tuple, Union

from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.processing_utils import Unpack
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.models.llama.modeling_llama import (
    KwargsForCausalLM,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    LlamaDecoderLayer,
    LlamaPreTrainedModel,
    LlamaForCausalLM
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast
)
from transformers.utils import logging
import torch
from torch.nn import CrossEntropyLoss
import torch.nn as nn

from prompt_generators import RandomPromptGenerator, HiddenStatePromptGenerator, ContextPromptGenerator

logger = logging.get_logger(__name__)


class BasePromptLlama(LlamaPreTrainedModel):

    def __init__(self, config: LlamaConfig, prompt_config: dict):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.prompt_config = prompt_config
        self.num_prompt_tokens = prompt_config["num_prompt_tokens"]
        self.at_layer = prompt_config["at_layer"]
        self.prompt_generator = None


        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()
    
    def add_prompt(self):
        """Initialize the prompt generator."""
        if self.prompt_config["type"] == "random":
            self.prompt_generator = RandomPromptGenerator(config=self.prompt_config)
            self.prompt_generator.to(self.embed_tokens.weight.device)
        elif self.prompt_config["type"] == "hidden_states":
            self.prompt_generator = HiddenStatePromptGenerator(config=self.prompt_config)
            self.prompt_generator.to(self.embed_tokens.weight.device)
        elif self.prompt_config["type"] == "context":
            self.prompt_generator = ContextPromptGenerator(config=self.prompt_config, embed=self.embed_tokens)
            self.prompt_generator.to(self.embed_tokens.weight.device)
        else:
            raise ValueError(f"Unknown prompt type: {self.prompt_config['type']}")
        return self.prompt_generator
    
    def call_prompt(self, hidden_states=None, seq_lengths=None, context_ids=None):
        """Call the prompt generator."""
        if self.prompt_generator is None:
            raise ValueError("Prompt generator not initialized. Call `add_prompt()` first.")
        if self.prompt_config["type"] == "random":
            prefix = self.prompt_generator()
            prefix = prefix.unsqueeze(0).expand(hidden_states.shape[0], -1, -1)
        elif self.prompt_config["type"] == "hidden_states":
            prefix = self.prompt_generator(hidden_states=hidden_states, seq_lengths=seq_lengths)
            # prefix = prefix.unsqueeze(1).expand(-1, hidden_states.shape[1], -1)
        elif self.prompt_config["type"] == "context":
            if context_ids is None:
                raise ValueError("context_ids must be provided for context prompts.")
            prefix = self.prompt_generator(hidden_states=hidden_states, context_ids=context_ids, seq_lengths=seq_lengths)
        return prefix


    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            seq_lengths: Optional[torch.LongTensor] = None,
            context_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
        ) -> Union[Tuple, BaseModelOutputWithPast]:
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            use_cache = use_cache if use_cache is not None else self.config.use_cache
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            if (input_ids is None) ^ (inputs_embeds is not None):
                raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

            if self.gradient_checkpointing and self.training and use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
                )
                use_cache = False

            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)

            if use_cache and past_key_values is None:
                past_key_values = DynamicCache()

            if cache_position is None:
                past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
                cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
                )

            if position_ids is None:
                position_ids = cache_position.unsqueeze(0)
            batch_size = inputs_embeds.shape[0]
            prompt_attention_mask = torch.ones(batch_size, self.num_prompt_tokens).to(attention_mask.device)
            
            causal_mask = self._update_causal_mask(
                attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
            )

            hidden_states = inputs_embeds

            # create position embeddings to be shared across the decoder layers
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

            # decoder layers
            all_hidden_states = () if output_hidden_states else None
            all_self_attns = () if output_attentions else None

            for idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):  
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                if idx == self.at_layer and self.prompt_generator is not None: # Layer where prompt is prepended.
                    prefix = self.call_prompt(hidden_states=hidden_states, seq_lengths=seq_lengths, context_ids=context_ids)
                    hidden_states = torch.cat((prefix, hidden_states), dim=1)
                    attention_mask = torch.cat([prompt_attention_mask, attention_mask], dim=-1)
                    cache_position = torch.arange(
                    0, 
                    hidden_states.shape[1], 
                    device=inputs_embeds.device
                )
                    
                    if self.training:
                        causal_mask =  self._update_causal_mask(
                            attention_mask, hidden_states, cache_position, past_key_values, output_attentions
                            )       
                        causal_mask[:, :, :, :self.num_prompt_tokens] = 0
                    position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
                    position_embeddings = self.rotary_emb(hidden_states, position_ids)

                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        hidden_states,
                        causal_mask,
                        position_ids,
                        past_key_values,
                        output_attentions,
                        use_cache,
                        cache_position,
                        position_embeddings,
                    )
                else:
                    if not self.training:
                        causal_mask = None
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        **flash_attn_kwargs,
                    )

                hidden_states = layer_outputs[0]

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

            hidden_states = self.norm(hidden_states)

            # add hidden states from the last decoder layer
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            output = BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=past_key_values if use_cache else None,
                hidden_states=all_hidden_states,
                attentions=all_self_attns,
            )
            return output if return_dict else output.to_tuple()
    
    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to plcae the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        
        return causal_mask

class GenerativePromptLlama(LlamaForCausalLM):

    def __init__(self, config, prompt_config):
        super().__init__(config)
        self.model = BasePromptLlama(config, prompt_config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.prompt_config = prompt_config
        self.num_prompt_tokens = prompt_config["num_prompt_tokens"]

        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            context_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            seq_lengths: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            logits_to_keep: Union[int, torch.Tensor] = 0,
            **kwargs: Unpack[KwargsForCausalLM],
        ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            context_ids=context_ids,
            seq_lengths=seq_lengths,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None

        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., self.num_prompt_tokens :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

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
