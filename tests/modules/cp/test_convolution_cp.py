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

        ######## cp fwd with overlapped communication ########
        halo_shape = (self.B, self.W - 1, self.D)
        
        x_shard = self.cp_shard(x)[self.rank]
        halo_to_send = x_shard[:, -(self.W - 1):, :]

        # Step 1: Initiate non-blocking halo exchange (communication starts)
        halo = None
        recv_req = send_req = None
        
        if self.rank > 0:
            halo = torch.empty(*halo_shape, dtype=self.dtype, device=self.device)
            recv_req = dist.irecv(halo, src=self.rank - 1)
        
        if self.rank < world_size - 1 and halo_to_send is not None:
            send_req = dist.isend(halo_to_send.contiguous(), dst=self.rank + 1)
        
        print(f"Rank {self.rank} - halo exchange initiated")

        # Step 2: Perform local convolution on shard (overlapped with communication)
        out_shard, _ = causal_conv1d_fwd(x_shard, weight, residual=None, bias=None)
        print(f"Rank {self.rank} - out_shard computed (shape: {out_shard.shape})")
        
        # Step 3: Wait for halo exchange to complete
        if recv_req is not None:
            recv_req.wait()
            print(f"Rank {self.rank} - received halo (shape: {halo.shape})")
        if send_req is not None:
            send_req.wait()
            print(f"Rank {self.rank} - halo sent")

        # Step 4: Perform boundary correction using received halo
        if self.rank > 0 and halo is not None:
            print(f"Rank {self.rank} - correcting boundary")
            x_bndr = torch.cat([halo, x_shard[:, :self.W - 1, :]], dim=1)
            print(f"Rank {self.rank} - x_bndr shape: {x_bndr.shape}")
            halo_out, _ = causal_conv1d_fwd(x_bndr, weight, residual=None, bias=None)
            print(f"Rank {self.rank} - halo_out shape: {halo_out.shape}")
            out_shard[:, :self.W - 1] = halo_out[:, self.W - 1:]

        print(f"Rank {self.rank} - final out_shard shape: {out_shard.shape}")

        ####### compare #####
        o_cp_ref = self.cp_shard(ref_out)[self.rank]
        assert_close("o", o_cp_ref, out_shard, 0.002)










