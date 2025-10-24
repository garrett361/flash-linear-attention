# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Template for model configuration

from typing import Optional, Union, List
from transformers import PretrainedConfig


class TemplateConfig(PretrainedConfig):
    """
    Configuration class for Template model.
    
    This template provides a standardized configuration structure for new models
    in the FLA library. Modify the parameters to match your specific model requirements.
    
    Args:
        vocab_size (int, optional):
            Vocabulary size of the model. Default: 32000.
        hidden_size (int, optional):
            Hidden size of the model. Default: 2048.
        intermediate_size (int, optional):
            Intermediate size of the MLP. Default: 5632.
        num_hidden_layers (int, optional):
            Number of hidden layers. Default: 24.
        num_attention_heads (int, optional):
            Number of attention heads. Default: 32.
        num_key_value_heads (int, optional):
            Number of key-value heads for GQA. Default: None.
        hidden_act (str, optional):
            Hidden activation function. Default: "silu".
        max_position_embeddings (int, optional):
            Maximum position embeddings. Default: 32768.
        rms_norm_eps (float, optional):
            RMS norm epsilon. Default: 1e-6.
        attention_dropout (float, optional):
            Attention dropout rate. Default: 0.0.
        use_cache (bool, optional):
            Whether to use cache. Default: True.
        pad_token_id (int, optional):
            Padding token ID. Default: None.
        bos_token_id (int, optional):
            Beginning of sequence token ID. Default: 1.
        eos_token_id (int, optional):
            End of sequence token ID. Default: 2.
        tie_word_embeddings (bool, optional):
            Whether to tie word embeddings. Default: False.
        rope_theta (float, optional):
            RoPE theta parameter. Default: 10000.0.
        rope_scaling (dict, optional):
            RoPE scaling configuration. Default: None.
        attention_bias (bool, optional):
            Whether to use attention bias. Default: False.
        attention_dropout (float, optional):
            Attention dropout rate. Default: 0.0.
        # Template-specific parameters
        template_param1 (float, optional):
            Template-specific parameter 1. Default: 1.0.
        template_param2 (bool, optional):
            Template-specific parameter 2. Default: True.
        template_param3 (int, optional):
            Template-specific parameter 3. Default: 64.
        **kwargs:
            Additional keyword arguments.
    """
    
    model_type = "template"  # Replace with your model type
    keys_to_ignore_at_inference = ["past_key_values"]
    
    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 2048,
        intermediate_size: int = 5632,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 32,
        num_key_value_heads: Optional[int] = None,
        hidden_act: str = "silu",
        max_position_embeddings: int = 32768,
        rms_norm_eps: float = 1e-6,
        attention_dropout: float = 0.0,
        use_cache: bool = True,
        pad_token_id: Optional[int] = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[dict] = None,
        attention_bias: bool = False,
        # Template-specific parameters
        template_param1: float = 1.0,
        template_param2: bool = True,
        template_param3: int = 64,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.attention_dropout = attention_dropout
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        
        # Template-specific parameters
        self.template_param1 = template_param1
        self.template_param2 = template_param2
        self.template_param3 = template_param3
        
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
