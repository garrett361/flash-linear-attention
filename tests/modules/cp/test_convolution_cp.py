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


    @pytest.mark.world_size([2])
    def test_bwd(self, world_size: int) -> None:
        # Use float32 to make debugging & matching easier
        x = torch.randn(self.B, self.T, self.D, device=self.device, dtype=self.dtype)
        weight = torch.randn(self.D, self.W, device=self.device, dtype=self.dtype)

        # ----- reference full backward (single-GPU) -----
        # Forward just to get the shape for dy
        ref_out, _ = causal_conv1d_fwd(x, weight, residual=None, bias=None)
        dy = torch.randn_like(ref_out)

        dx_ref, dw_ref, _, _, _ = causal_conv1d_bwd(
            x=x,
            dy=dy,
            dht=None,
            weight=weight,
            bias=None,
            residual=None,
            initial_state=None,
            activation=None,
            cu_seqlens=None,
        )

        print("Reference backward computed.")

        # ----- CP backward with halo-aware correction -----
        halo_shape = (self.B, self.W - 1, self.D)

        # Shard x and dy along the time dimension
        x_shard = self.cp_shard(x)[self.rank]
        dy_shard = self.cp_shard(dy)[self.rank]

        # Local backward (ignores cross-shard dependencies)
        dx_local, dw_local, _, _, _ = causal_conv1d_bwd(
            x=x_shard,
            dy=dy_shard,
            dht=None,
            weight=weight,
            bias=None,
            residual=None,
            initial_state=None,
            activation=None,
            cu_seqlens=None,
        )

        # ---- Forward-style halo of x (from left neighbor) ----
        halo_to_send_x = x_shard[:, -(self.W - 1):, :]
        halo_x = _exchange_halo(
            rank=self.rank,
            world_size=world_size,
            device=self.device,
            halo_to_send=halo_to_send_x,
            halo_shape=halo_shape,
            dtype=self.dtype,
        )

        # ---- Boundary backward on tiny [halo_x, local_prefix] window ----
        dw_bndr = None
        grad_halo_to_prev = None

        if self.rank > 0:
            # Build boundary input: [tail of prev rank, first W-1 of this rank]
            x_bndr = torch.cat([halo_x, x_shard[:, : self.W - 1, :]], dim=1)
            # dy for boundary: only outputs corresponding to this rank's first W-1 positions
            # In forward we used halo_out[:, W-1:, :] -> out_shard[:, :W-1, :]
            # So here we place dy_shard[:, :W-1, :] into those same positions.
            B, L_bndr, D = x_bndr.shape  # L_bndr = 2*(W-1)
            dy_bndr = torch.zeros(B, L_bndr, D, device=self.device, dtype=self.dtype)
            dy_bndr[:, self.W - 1 :, :] = dy_shard[:, : self.W - 1, :]

            dx_bndr, dw_bndr, _, _, _ = causal_conv1d_bwd(
                x=x_bndr,
                dy=dy_bndr,
                dht=None,
                weight=weight,
                bias=None,
                residual=None,
                initial_state=None,
                activation=None,
                cu_seqlens=None,
            )

            # Split boundary gradients:
            # - first W-1 positions: grad wrt halo_x (goes to previous rank)
            # - last  W-1 positions: extra grad wrt our local first W-1 inputs
            grad_halo_to_prev = dx_bndr[:, : self.W - 1, :]
            extra_grad_local_prefix = dx_bndr[:, self.W - 1 :, :]

            # Add extra gradient to our own first W-1 timesteps
            dx_local[:, : self.W - 1, :] += extra_grad_local_prefix

        # ---- Send gradient halo back to previous rank; receive from right ----
        halo_grad = _exchange_grad_halo(
            rank=self.rank,
            world_size=world_size,
            device=self.device,
            grad_to_send=grad_halo_to_prev,
            halo_shape=halo_shape,
        )

        # If we received gradient from the right neighbor, add to our tail
        if halo_grad is not None:
            dx_local[:, -(self.W - 1):, :] += halo_grad

        # ---- Combine weight gradients: local + boundary, then allreduce ----
        if dw_bndr is not None:
            dw_local = dw_local + dw_bndr

        # All-reduce over CP group so every rank sees the full dw
        dist.all_reduce(dw_local)

        # ----- Compare with reference -----
        dx_ref_shard = self.cp_shard(dx_ref)[self.rank]

        assert_close("dx", dx_ref_shard, dx_local, 0.002)

        # Check dw on (say) rank 0; all ranks have the same due to all_reduce
        if self.rank == 0:
            assert_close("dw", dw_ref, dw_local, 0.002)

    







