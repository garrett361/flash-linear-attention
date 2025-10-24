# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Template for model implementation

from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import logging

from fla.layers import TemplateAttention  # Replace with your attention layer
from fla.modules import MLP, RMSNorm
from fla.models.utils import Cache, GradientCheckpointingLayer

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast

logger = logging.get_logger(__name__)


class TemplateBlock(GradientCheckpointingLayer):
    """
    Template block implementation.
    
    This template provides a standardized structure for implementing model blocks
    in the FLA library. Modify the components to match your specific model architecture.
    """

    def __init__(self, config: TemplateConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        # Attention layer
        self.attn = TemplateAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            expand_k=config.template_param1,  # Replace with actual parameters
            expand_v=config.template_param1,
            use_short_conv=config.template_param2,
            conv_size=config.template_param3,
            use_rope=True,
            rope_theta=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
            layer_idx=layer_idx,
        )

        # MLP layer
        self.mlp = MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            fuse_swiglu=True,
        )

        # Normalization layers
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Forward pass of the template block.
        
        Args:
            hidden_states: Input hidden states
            attention_mask: Optional attention mask
            position_ids: Optional position IDs
            past_key_values: Optional cache
            output_attentions: Whether to output attention weights
            use_cache: Whether to use cache
            **kwargs: Additional keyword arguments
            
        Returns:
            Tuple containing output hidden states and optional cache
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class TemplateModel(PreTrainedModel):
    """
    Template model implementation.
    
    This template provides a standardized structure for implementing models
    in the FLA library. Modify the components to match your specific model architecture.
    """

    config_class = TemplateConfig  # Replace with your config class

    def __init__(self, config: TemplateConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([
            TemplateBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Cache]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass of the template model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Optional attention mask
            position_ids: Optional position IDs
            past_key_values: Optional cache
            inputs_embeds: Optional input embeddings
            use_cache: Whether to use cache
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return a dictionary
            **kwargs: Additional keyword arguments
            
        Returns:
            Model outputs
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.layers))

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Prepare attention mask
        if attention_mask is not None:
            if attention_mask.size() != (batch_size, seq_length):
                raise ValueError(
                    f"Attention mask should be of size {(batch_size, seq_length)}, but is {attention_mask.size()}"
                )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            hidden_states = self._gradient_checkpointing_func(
                self.layers.__call__,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
            )
        else:
            for decoder_layer in self.layers:
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    **kwargs,
                )[0]

        hidden_states = self.norm(hidden_states)

        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = ()
            for decoder_layer in self.layers:
                all_hidden_states += (hidden_states,)
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = tuple([layer[2] for layer in hidden_states])

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states] if v is not None)
        return CausalLMOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states if output_hidden_states else None,
            attentions=None,
        )


class TemplateForCausalLM(PreTrainedModel):
    """
    Template model for causal language modeling.
    
    This template provides a standardized structure for implementing causal LM models
    in the FLA library. Modify the components to match your specific model architecture.
    """

    config_class = TemplateConfig  # Replace with your config class
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: TemplateConfig):
        super().__init__(config)
        self.model = TemplateModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Cache]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass of the template causal LM model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Optional attention mask
            position_ids: Optional position IDs
            past_key_values: Optional cache
            inputs_embeds: Optional input embeddings
            labels: Optional labels for language modeling
            use_cache: Whether to use cache
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return a dictionary
            **kwargs: Additional keyword arguments
            
        Returns:
            Model outputs
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Decoder outputs
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
            **kwargs,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

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

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        """
        Prepare inputs for generation.
        
        Args:
            input_ids: Input token IDs
            past_key_values: Optional cache
            attention_mask: Optional attention mask
            inputs_embeds: Optional input embeddings
            **kwargs: Additional keyword arguments
            
        Returns:
            Prepared inputs
        """
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]
            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]
        # If `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs
