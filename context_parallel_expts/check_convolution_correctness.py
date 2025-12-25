import os
import torch
import torch.distributed as dist

from fla.modules.convolution import causal_conv1d_fwd, causal_conv1d_bwd, causal_conv1d
from convolution import CausalConv1dFunction

# -----------------------------
# Init distributed
# -----------------------------
def init_dist():
    if not dist.is_initialized():
        dist.init_process_group("nccl")
        torch.cuda.set_device(dist.get_rank())


# -----------------------------
# Helper: run reference
# -----------------------------
def run_reference(x, weight, bias, activation):
    x = x.detach().clone().requires_grad_(True)
    weight = weight.detach().clone().requires_grad_(True)
    bias = bias.detach().clone().requires_grad_(True)

    y_ref, _ = causal_conv1d(
        x=x,
        weight=weight,
        bias=bias,
        residual=None,
        initial_state=None,
        output_final_state=False,
        activation=activation,
        backend="triton",
        cu_seqlens=None,
    )

    dy = torch.ones_like(y_ref)
    y_ref.backward(dy)

    return y_ref.detach(), x.grad.detach(), weight.grad.detach(), bias.grad.detach()




# -----------------------------
# Helper: run CP version
# -----------------------------
def run_cp(x, weight, bias, activation, cp_rank, cp_size):
    x = x.detach().clone().requires_grad_(True)
    weight = weight.detach().clone().requires_grad_(True)
    bias = bias.detach().clone().requires_grad_(True)

    y_cp, _ = CausalConv1dFunction.apply(
        x,
        weight,
        bias,   # bias
        None,   # residual
        None,   # initial_state
        False,  # output_final_state
        activation,   # activation
        None,   # cu_seqlens
        cp_rank,
        cp_size,
    )

    dy = torch.ones_like(y_cp)
    y_cp.backward(dy)

    return y_cp.detach(), x.grad.detach(), weight.grad.detach(), bias.grad.detach()


# -----------------------------
# Main test
# -----------------------------
def main():
    init_dist()

    rank = dist.get_rank()
    world = dist.get_world_size()

    torch.manual_seed(0)

    # ---- config ----
    B = 1
    T = 256
    D = 64
    W = 4
    dtype = torch.bfloat16
    device = torch.device("cuda")

    # ---- inputs ----
    x_global = torch.randn(B, T, D, device=device, dtype=dtype)
    weight = torch.randn(D, W, device=device, dtype=dtype)
    bias = torch.randn(D, device=device, dtype=dtype)

    # ---- shard x ----
    x_shards = torch.chunk(x_global, world, dim=1)
    x = x_shards[rank].contiguous()

    # ---- reference (only rank 0 runs it) ----
    if rank == 0:
        y_ref, dx_ref, dw_ref, db_ref = run_reference(x_global, weight, bias, None)

    # ---- CP run ----
    y_cp, dx_cp, dw_cp, db_cp = run_cp(x, weight, bias, None, rank, world)

    # ---- gather CP outputs for comparison ----
    y_cp_full = [torch.empty_like(y_cp) for _ in range(world)]
    dx_cp_full = [torch.empty_like(dx_cp) for _ in range(world)]

    dist.all_gather(y_cp_full, y_cp)
    dist.all_gather(dx_cp_full, dx_cp)

    y_cp_full = torch.cat(y_cp_full, dim=1)
    dx_cp_full = torch.cat(dx_cp_full, dim=1)


    # ---- compare (rank 0 only) ----
    if rank == 0:
        def check(name, a, b, atol=5e-1, rtol=5e-2):
            ok = torch.allclose(a, b, atol=atol, rtol=rtol)
            max_err = (a - b).abs().max().item()
            print(f"{name:>6}: {'OK' if ok else 'FAIL'} | max_err = {max_err:.3e}")
            return ok

        print("\n=== CP CausalConv1d Correctness ===")
        check("y", y_cp_full, y_ref)
        check("dx", dx_cp_full, dx_ref)
        check("dw", dw_cp, dw_ref)
        check("db", db_cp, db_ref)


        # print("To understand the magnitude of errors: ")

        # eps = 1e-2  # or 1e-1 depending on scale
        # mask = dw_ref.abs() > eps
        # rel = ((dw_cp - dw_ref).abs() / dw_ref.abs().clamp_min(eps))
        # print("rel_max(masked):", rel[mask].max())
        # print("rel_mean(masked):", rel[mask].mean())
        # print("masked_frac:", mask.float().mean())

        # diff = (dw_cp - dw_ref).float()
        # ref  = dw_ref.float()
        # print("Linf:", diff.abs().max().item())
        # print("L2_rel:", (diff.norm() / ref.norm()).item())
        # print("cosine:", torch.nn.functional.cosine_similarity(dw_cp.flatten().float(),
        #                                                     dw_ref.flatten().float(), dim=0).item())


    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()