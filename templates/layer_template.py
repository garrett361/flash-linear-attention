# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Template for implementing new attention layers

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from transformers.utils import logging

from fla.modules import RMSNorm, RotaryEmbedding, ShortConvolution
from fla.modules.activations import swiglu, swish
from fla.ops.template import chunk_template  # Replace with actual operation
from fla.ops.utils.index import prepare_lens_from_mask

if TYPE_CHECKING:
    from fla.models.utils import Cache

logger = logging.get_logger(__name__)


class TemplateAttention(nn.Module):
    """
    Template for implementing new attention mechanisms.
    
    This template provides a standardized structure for implementing attention layers
    in the FLA library. Follow these steps to implement a new attention mechanism:
    
    1. Replace 'TemplateAttention' with your attention mechanism name
    2. Update the docstring with your mechanism's description and paper reference
    3. Modify the __init__ parameters to match your mechanism's requirements
    4. Implement the forward method with your specific attention computation
    5. Create corresponding operations in fla/ops/template/
    6. Add proper imports and exports
    
    Args:
        hidden_size (int, Optional):
            The hidden size of the input. Default: 2048.
        num_heads (int, Optional):
            The number of attention heads. Default: 32.
        num_kv_heads (int, Optional):
            The number of key/value heads for grouped-query attention. If None, equals `num_heads`.
            Default: None.
        expand_k (float, Optional):
            Expansion factor for key dimension. Default: 1.0.
        expand_v (float, Optional):
            Expansion factor for value dimension. Default: 1.0.
        qkv_bias (bool, Optional):
            Whether to use bias for Q/K/V projections. Default: False.
        qk_norm (bool, Optional):
            Whether to apply per-head RMSNorm to Q and K before attention. Default: False.
        use_short_conv (bool, Optional):
            Whether to use short convolution. Default: False.
        conv_size (int, Optional):
            Size of the convolution kernel. Default: 4.
        conv_bias (bool, Optional):
            Whether to use bias in convolution. Default: False.
        use_rope (bool, Optional):
            Whether to use rotary position embedding. Default: True.
        rope_theta (float, Optional):
            The base frequency for rotary position embedding. Default: 10000.
        use_input_gate (bool, Optional):
            Whether to use input gating. Default: False.
        use_output_gate (bool, Optional):
            Whether to use output gating. Default: False.
        use_norm (bool, Optional):
            Whether to use normalization. Default: True.
        norm_eps (float, Optional):
            Epsilon for normalization. Default: 1e-5.
        max_position_embeddings (int, Optional):
            The maximum position embeddings. Default: None.
        layer_idx (int, Optional):
            The index of the layer (used for cache compatibility). Default: None.
        **kwargs:
            Additional keyword arguments for specific attention mechanisms.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 32,
        num_kv_heads: Optional[int] = None,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        use_short_conv: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        use_rope: bool = True,
        rope_theta: float = 10000.,
        use_input_gate: bool = False,
        use_output_gate: bool = False,
        use_norm: bool = True,
        norm_eps: float = 1e-5,
        max_position_embeddings: Optional[int] = None,
        layer_idx: int | None = None,
        **kwargs
    ) -> TemplateAttention:
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.use_rope = use_rope
        self.rope_theta = rope_theta
        self.use_input_gate = use_input_gate
        self.use_output_gate = use_output_gate
        self.use_norm = use_norm
        self.norm_eps = norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.layer_idx = layer_idx

        # Calculate dimensions
        self.key_dim = int(self.hidden_size * self.expand_k)
        self.value_dim = int(self.hidden_size * self.expand_v)
        self.head_k_dim = self.key_dim // self.num_heads
        self.head_v_dim = self.value_dim // self.num_heads
        self.head_kv_dim = self.key_dim // self.num_kv_heads

        if layer_idx is None:
            warnings.warn(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "lead to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        # Projection layers
        self.q_proj = nn.Linear(self.hidden_size, self.key_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.key_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.value_dim, bias=qkv_bias)
        self.o_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        # Optional gating
        if use_output_gate:
            self.g_proj = nn.Linear(self.hidden_size, self.value_dim, bias=False)

        # Optional short convolution
        if use_short_conv:
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu'
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu'
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu'
            )

        # Optional normalization
        if use_norm:
            if use_output_gate:
                from fla.modules import FusedRMSNormGated
                self.g_norm = FusedRMSNormGated(
                    hidden_size=self.head_v_dim,
                    eps=norm_eps
                )
            else:
                self.g_norm = RMSNorm(
                    hidden_size=self.head_v_dim,
                    eps=norm_eps
                )

        # Optional rotary position embedding
        if use_rope:
            self.rotary = RotaryEmbedding(self.head_k_dim, theta=rope_theta)

        # Optional QK normalization
        if qk_norm:
            self.q_norm = RMSNorm(self.head_k_dim, eps=norm_eps)
            self.k_norm = RMSNorm(self.head_k_dim, eps=norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        """
        Forward pass for the attention mechanism.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask of shape [batch_size, seq_len]
            past_key_values: Optional cache for key-value pairs
            use_cache: Whether to use caching
            output_attentions: Whether to output attention weights
            **kwargs: Additional keyword arguments
            
        Returns:
            Tuple containing:
            - Output tensor of shape [batch_size, seq_len, hidden_size]
            - Optional attention weights
            - Optional updated cache
        """
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        # Get cached state if available
        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        # Handle cu_seqlens for variable length sequences
        cu_seqlens = kwargs.get('cu_seqlens', None)
        
        # Project inputs to Q, K, V
        if self.use_short_conv:
            # Handle convolution states
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state.get('conv_state', (None, None, None))
            
            conv_mask = attention_mask[:, -hidden_states.shape[1]:] if attention_mask is not None else None
            
            q, conv_state_q = self.q_conv1d(
                x=self.q_proj(hidden_states),
                mask=conv_mask,
                cache=conv_state_q,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens
            )
            k, conv_state_k = self.k_conv1d(
                x=self.k_proj(hidden_states),
                mask=conv_mask,
                cache=conv_state_k,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens
            )
            v, conv_state_v = self.v_conv1d(
                x=self.v_proj(hidden_states),
                mask=conv_mask,
                cache=conv_state_v,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens
            )
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)

        # Apply input gating if enabled
        if self.use_input_gate:
            q, k, v = map(lambda x: swish(x), (q, k, v))

        # Handle padding mask
        if attention_mask is not None:
            v = v.mul_(attention_mask[:, -v.shape[-2]:, None])

        # Reshape for multi-head attention
        q = rearrange(q, '... (h d) -> ... h d', d=self.head_k_dim)
        k = rearrange(k, '... (h d) -> ... h d', d=self.head_k_dim)
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_v_dim)

        # Apply QK normalization if enabled
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Apply rotary position embedding if enabled
        if self.use_rope:
            seqlen_offset = 0
            if past_key_values is not None:
                seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
            q, k = self.rotary(q, k, seqlen_offset=seqlen_offset)

        # Get recurrent state for attention computation
        recurrent_state = last_state.get('recurrent_state', None) if last_state is not None else None

        # Apply attention mechanism (replace with your specific implementation)
        o, recurrent_state = chunk_template(
            q=q,
            k=k,
            v=v,
            initial_state=recurrent_state,
            output_final_state=use_cache,
            **kwargs
        )

        # Update cache if using caching
        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=q.shape[1]
            )

        # Apply output gating and normalization
        if self.use_norm and not self.use_output_gate:
            o = self.g_norm(o)
        elif self.use_output_gate:
            g = rearrange(self.g_proj(hidden_states), '... (h d) -> ... h d', d=self.head_v_dim)
            o = self.g_norm(o, g) if self.use_norm else swiglu(g, o)

        # Reshape output and project
        o = rearrange(o, '... h d -> ... (h d)')
        o = self.o_proj(o)

        return o, None, past_key_values

    def state_size(self, seq_len: int = 2048) -> int:
        """
        Calculate the state size for caching.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            State size in elements
        """
        # Replace with actual state size calculation for your mechanism
        return 2 * self.head_k_dim * self.head_v_dim
