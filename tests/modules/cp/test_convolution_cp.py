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


    @pytest.mark.world_size([2, 4, 8])
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




    @pytest.mark.world_size([2, 4, 8])
    def test_bwd(self, world_size: int) -> None:

        x = torch.randn(self.B, self.T, self.D, device=self.device, dtype=self.dtype)
        weight = torch.randn(self.D, self.W, device=self.device, dtype=self.dtype)

        # Reference
        ref_out, _ = causal_conv1d_fwd(x, weight, residual=None, bias=None)
        dy_global = torch.ones_like(ref_out)
        ref_dx, ref_dw, _, _, _ = causal_conv1d_bwd(
            x=x,
            dy=dy_global,
            dht=None,
            weight=weight,
            bias=None,
            residual=None,
            initial_state=None,
            activation=None,
            cu_seqlens=None,
        )

        # CP shards
        x_shard = self.cp_shard(x)[self.rank]            # [B, Ts, D]
        dy_shard = self.cp_shard(dy_global)[self.rank]   # [B, Ts, D]
        Ts = x_shard.shape[1]
        halo_shape = (self.B, self.W - 1, self.D)

        # -------------------------
        # CP Forward (unchanged)
        # needs LEFT x-halo
        # -------------------------
        x_halo_left = None
        recv_req_fwd = None
        send_req_fwd = None

        if self.rank > 0:
            x_halo_left = torch.empty(*halo_shape, dtype=self.dtype, device=self.device)
            recv_req_fwd = dist.irecv(x_halo_left, src=self.rank - 1)

        if self.rank < world_size - 1:
            x_to_send = x_shard[:, -(self.W - 1):, :].contiguous()
            send_req_fwd = dist.isend(x_to_send, dst=self.rank + 1)

        out_shard, _ = causal_conv1d_fwd(x_shard, weight, residual=None, bias=None)

        if recv_req_fwd is not None:
            recv_req_fwd.wait()
            x_bndr = torch.cat([x_halo_left, x_shard[:, :self.W - 1, :]], dim=1)
            halo_out, _ = causal_conv1d_fwd(x_bndr, weight, residual=None, bias=None)
            out_shard[:, :self.W - 1] = halo_out[:, self.W - 1:]

        if send_req_fwd is not None:
            send_req_fwd.wait()

        # -------------------------
        # CP Backward (FIXED)
        # needs RIGHT dy-halo only
        # -------------------------
        # 1) Exchange RIGHT dy halo
        # -------------------------
        dy_halo_right = torch.zeros(*halo_shape, dtype=self.dtype, device=self.device)
        recv_req = send_req = None

        if self.rank < world_size - 1:
            recv_req = dist.irecv(dy_halo_right, src=self.rank + 1)

        if self.rank > 0:
            send_req = dist.isend(dy_shard[:, :self.W - 1, :].contiguous(), dst=self.rank - 1)

        if recv_req is not None: recv_req.wait()
        if send_req is not None: send_req.wait()

        # -------------------------
        # 2) One bwd kernel call
        #    - pad x on the RIGHT so fake positions don't create dw terms
        #    - append right dy halo so last tokens get future dy
        # -------------------------
        x_pad = torch.zeros(*halo_shape, dtype=self.dtype, device=self.device)
        x_ext = torch.cat([x_shard, x_pad], dim=1)              # [B, Ts+W-1, D]
        dy_ext = torch.cat([dy_shard, dy_halo_right], dim=1)    # [B, Ts+W-1, D]

        dx_ext, dw_local, _, _, _ = causal_conv1d_bwd(
            x=x_ext,
            dy=dy_ext,
            dht=None,
            weight=weight,
            bias=None,
            residual=None,
            initial_state=None,
            activation=None,
            cu_seqlens=None,
        )

        dx_shard = dx_ext[:, :Ts, :].contiguous()

        # -------------------------
        # 3) Reduce dw in fp32
        # -------------------------
        dw_local_fp32 = dw_local.float()
        dist.all_reduce(dw_local_fp32, op=dist.ReduceOp.SUM)

        # -------------------------
        # 4) Extract reference shards for comparison
        # -------------------------
        ref_dx_shard = self.cp_shard(ref_dx)[self.rank]

        assert_close("d_x", ref_dx_shard, dx_shard, 0.005)
        assert_close("d_w", ref_dw, dw_local_fp32, 0.005)


