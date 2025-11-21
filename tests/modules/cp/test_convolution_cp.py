import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from dtest import DTest

from fla.ops.gated_delta_rule.chunk import (
    chunk_gated_delta_rule_bwd,
    chunk_gated_delta_rule_fwd,
)

from fla.modules.convolution import causal_conv1d_fwd, causal_conv1d_bwd
from fla.utils import assert_close


def _exchange_halo(rank, world_size, device, halo_to_send, halo_shape, dtype):
    halo = None
    recv_req = send_req = None

    if rank > 0:
        halo = torch.empty(*halo_shape, dtype=dtype, device=device)
        recv_req = dist.irecv(halo, src=rank - 1)

    if rank < world_size - 1 and halo_to_send is not None:
        send_req = dist.isend(halo_to_send.contiguous(), dst=rank + 1)

    if recv_req is not None:
        recv_req.wait()
    if send_req is not None:
        send_req.wait()

    return halo

def _exchange_grad_halo(
    rank: int,
    world_size: int,
    device: torch.device,
    grad_to_send: torch.Tensor | None,
    halo_shape,
) -> torch.Tensor | None:
    """
    Exchange gradient halos in the *reverse* direction:
    - grad flows from rank r -> r-1
    - rank < world_size-1 receives from r+1
    """
    recv_req = None
    send_req = None
    halo_grad = None

    # Post non-blocking receive first (if not last rank)
    if rank < world_size - 1:
        halo_grad = torch.empty(*halo_shape, dtype=torch.float32, device=device)
        recv_req = dist.irecv(halo_grad, src=rank + 1)

    # Post non-blocking send (if not rank 0)
    if rank > 0 and grad_to_send is not None:
        send_req = dist.isend(grad_to_send.contiguous(), dst=rank - 1)

    # Wait for operations to complete
    if recv_req is not None:
        recv_req.wait()
    if send_req is not None:
        send_req.wait()

    return halo_grad




class TestCPGDN(DTest):
    B: int = 1
    T: int = 256
    D: int = 64
    W: int = 4
    device: torch.device = torch.device("cuda")

    
    dtype: torch.dtype = torch.bfloat16

    def cp_shard(
        self,
        tensor: torch.Tensor,
        cp_degree: int | None = None,
        shard_dim: int | None = 1,
    ) -> torch.Tensor:
        return tensor.tensor_split(cp_degree or self.world_size, dim=shard_dim)

    def cp_unshard(
        self,
        tensor_list: list[torch.Tensor],
        shard_dim: int | None = 1,
    ) -> torch.Tensor:
        return torch.cat(tensor_list, dim=shard_dim)


    @pytest.mark.world_size([2])
    def test_fwd(self, world_size: int) -> None:
        x = torch.randn(self.B, self.T, self.D, device=self.device, dtype=self.dtype)
        weight = torch.randn(self.D, self.W, device=self.device, dtype=self.dtype)

        ######## reference fwd ########
        ref_out, _ = causal_conv1d_fwd(x, weight, residual=None, bias=None)

        ######## cp fwd ########
        halo_shape = (self.B, self.W - 1, self.D)
        
        x_shard = self.cp_shard(x)[self.rank]
        halo_to_send = x_shard[:, -(self.W - 1):, :]

        # Perform local fwd convolution
        out_shard, _ = causal_conv1d_fwd(x_shard, weight, residual=None, bias=None)
        print(f"Rank {self.rank} - out_shard shape: {out_shard.shape}")
        
        # Exchange halos using non-blocking operations
        halo = _exchange_halo(
            self.rank, world_size, self.device,
            halo_to_send, halo_shape, self.dtype
        )        

        print(f"Rank {self.rank} - received halo of shape: {halo.shape if halo is not None else None}")

        # Correct boundary using halo
        if self.rank > 0:
            print(f"Rank {self.rank} - correcting boundary")
            x_bndr = torch.cat([halo, x_shard[:, :self.W - 1, :]], dim=1)
            print(f"Rank {self.rank} - x_bndr shape: {x_bndr.shape}")
            halo_out, _ = causal_conv1d_fwd(x_bndr, weight, residual=None, bias=None)
            print(f"Rank {self.rank} - halo_out shape: {halo_out.shape}")
            out_shard[:, :self.W - 1] = halo_out[:, self.W - 1:]

        print(f"Rank {self.rank} - corrected out_shard shape: {out_shard.shape}")

        ####### compare #####
        o_cp_ref = self.cp_shard(ref_out)[self.rank]
        assert_close("o", o_cp_ref, out_shard, 0.002)










