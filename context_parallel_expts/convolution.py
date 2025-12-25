import os
import torch
import torch.distributed as dist
from typing import Optional, Tuple
import math
import warnings
from fla.modules.convolution import causal_conv1d_fwd, causal_conv1d_bwd
from fla.utils import autotune_cache_kwargs, get_multiprocessor_count, input_guard, is_amd

class CausalConv1dFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(
        ctx,
        x: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        residual: Optional[torch.Tensor] = None,
        initial_state: Optional[torch.Tensor] = None,
        output_final_state: Optional[bool] = False,
        activation: Optional[str] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        cp_rank: int = 0,
        cp_size: int = 1,
        # cp_group = None, # Multi-node
    ):

        # assert bias is None and residual is None and initial_state is None
        assert activation is None and cu_seqlens is None

        ctx.activation = activation
        ctx.cu_seqlens = cu_seqlens
        ctx.save_for_backward(x, weight, bias, residual, initial_state)

        # Save rank's details 
        ctx.cp_rank = cp_rank 
        ctx.cp_size = cp_size 
        # ctx.cp_group = cp_group

        if cp_size == 1:

            # No Context Parallelism

            y, final_state = causal_conv1d_fwd(
                x=x,
                weight=weight,
                bias=bias,
                residual=residual,
                initial_state=initial_state,
                output_final_state=output_final_state,
                activation=activation,
                cu_seqlens=cu_seqlens,
            )

        else: 

            # Context Parallel Implementation

            B, Ts, D = x.size(0), x.size(1), x.size(2) 
            W = weight.size(1) # Assuming weight is of shape: (D, W)

            x_halo_to_send = x[:, -(W-1):, :]

            halo_shape = (B, W-1, D)

            # Initiate non blocking halo communication
            halo = None 
            recv_req = send_req = None 

            if cp_rank > 0:
                halo = torch.empty(*halo_shape, dtype=x.dtype, device=x.device)
                recv_req = dist.irecv(halo, src=cp_rank-1)
            
            if cp_rank < cp_size - 1 and x_halo_to_send is not None:
                send_req = dist.isend(x_halo_to_send, dst=cp_rank+1)

            
            # Perform local convolution on shard (overlapped with communication)
            
            y, final_state = causal_conv1d_fwd(
                x=x,
                weight=weight,
                bias=bias,
                residual=residual,
                initial_state=initial_state,
                output_final_state=output_final_state,
                activation=activation,
                cu_seqlens=cu_seqlens,
            )

            # Wait for halo exchange to complete
            
            if recv_req is not None:
                recv_req.wait()

            if send_req is not None:
                send_req.wait()

            if cp_rank > 0:

                # Perform boundary correction using received halo

                x_bndr = torch.cat([halo, x[:, :W - 1, :]], dim=1)

                y_bndr, _ = causal_conv1d_fwd(
                    x=x_bndr,
                    weight=weight,
                    bias=bias,
                    residual=residual,
                    # initial_state=initial_state,
                    # output_final_state=output_final_state,
                    activation=activation,
                    # cu_seqlens=cu_seqlens,
                )

                y[:, :W-1, :] = y_bndr[:, W-1:, :]


        # return y, final_state

        return y, None

    @staticmethod
    @input_guard
    def backward(ctx, dy: torch.Tensor, dht: Optional[torch.Tensor] = None):
        x, weight, bias, residual, initial_state = ctx.saved_tensors
        cp_rank, cp_size = ctx.cp_rank, ctx.cp_size

        # assert bias is None and residual is None and initial_state is None

        B, Ts, D = x.shape
        W = weight.size(1)
        halo_shape = (B, W-1, D)

        if cp_size == 1:

            dx, dw, db, dr, dh0 = causal_conv1d_bwd(
                x=x,
                dy=dy,
                dht=dht,
                weight=weight,
                bias=bias,
                residual=residual,
                initial_state=initial_state,
                activation=ctx.activation,
                cu_seqlens=ctx.cu_seqlens,
            )

        else:

            # Context Parallel Implementation

            # Start the exchange of dy halo
            dy_halo_right = torch.zeros(*halo_shape, dtype=x.dtype, device=x.device)
            recv_req = send_req = None

            if cp_rank < cp_size - 1:
                recv_req = dist.irecv(dy_halo_right, src=cp_rank + 1)

            if cp_rank > 0:
                send_req = dist.isend(dy[:, :W - 1, :], dst=cp_rank - 1)

            # Backward Kernel on Local shard while communication happening

            dx, dw, db, _, _ = causal_conv1d_bwd(
                x=x,
                dy=dy,
                dht=dht,
                weight=weight,
                bias=bias,
                residual=residual,
                initial_state=initial_state,
                activation=ctx.activation,
                cu_seqlens=ctx.cu_seqlens,
            )

            # Wait for communication to finish

            if recv_req is not None: recv_req.wait()
            if send_req is not None: send_req.wait()

            # Backward kernel on boundary tokens with RIGHT x zero

            x_pad = torch.zeros(*halo_shape, dtype=x.dtype, device=x.device)
            x_bndr = torch.cat([x[:, -(W-1):, :], x_pad], dim=1)
            dy_bndr = torch.cat([dy[:, -(W-1):, :], dy_halo_right], dim=1)

            dx_bndr, dw_bndr, _, _, _ = causal_conv1d_bwd(
                x=x_bndr,
                dy=dy_bndr,
                dht=None,
                weight=weight,
                bias=None,
                residual=None,
                initial_state=None,
                activation=ctx.activation,
                cu_seqlens=None,
            )

            dx[:, -(W-1):, :] = dx_bndr[:, :W-1, :]

            # Backward kernel on boundary tokens with LEFT dy zero and RIGHT x zero

            dy_pad = torch.zeros(*halo_shape, dtype=dy.dtype, device=dy.device)
            dy_bndr = torch.cat([dy_pad, dy_halo_right], dim=1)  # [B, 2*(W-1), D]


            _, dw_crt, _, _, _ = causal_conv1d_bwd(
                x=x_bndr,
                dy=dy_bndr,
                dht=None,
                weight=weight,
                bias=None,
                residual=None,
                initial_state=None,
                activation=ctx.activation,
                cu_seqlens=None,
            )

            dw += dw_crt

            # Reduce dw across all ranks
            dist.all_reduce(dw, op=dist.ReduceOp.SUM)

            dist.all_reduce(db, op=dist.ReduceOp.SUM)

        # return dx, dw, db, dr, dh0, None, None, None

        return dx, dw, db, None, None, None, None, None, None, None 