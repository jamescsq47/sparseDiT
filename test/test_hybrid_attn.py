import os
from yunchang import LongContextAttention, set_seq_parallel_pg, EXTRACT_FUNC_DICT
import torch
import torch.distributed as dist
from torch.profiler import profile, record_function, ProfilerActivity

try:
    from flash_attn import flash_attn_func
except ImportError:
    raise RuntimeError("flash_attn is necessary for this test!")
from yunchang.kernels import AttnType
from test_utils import attention_ref
import argparse
from flash_attn.utils.benchmark import benchmark_forward
import time

def greedy_partition_and_rearrange(sparse: torch.Tensor, num_groups: int = 8, group_size: int = 5):
    """
    将 [B, H, W] 的 sparse（B=40）按贪心法分到 num_groups 组，并重排为
    [B, H, W]，使得前 group_size 个属于组0，接着 group_size 个属于组1，依此类推。
    每组强制恰好 group_size 个元素。
    """
    B = sparse.shape[0]
    assert num_groups * group_size == B, "总数必须能被 num_groups * group_size 整除"

    # 权重：对后两个维度求和 -> [B]
    weights = sparse.sum(dim=(1, 2))

    # 按权重降序遍历
    order = torch.argsort(weights, descending=True)
    w_list = weights[order].detach().cpu().tolist()
    idx_list = order.detach().cpu().tolist()

    groups = [[] for _ in range(num_groups)]
    group_sums = [0.0] * num_groups
    group_counts = [0] * num_groups

    for idx, w in zip(idx_list, w_list):
        # 只在还有容量的组里选当前 sum 最小的
        gid = min(
            (g for g in range(num_groups) if group_counts[g] < group_size),
            key=lambda g: group_sums[g]
        )
        groups[gid].append(idx)
        group_sums[gid] += float(w)
        group_counts[gid] += 1

    # 拉平成新的排列顺序：组0的5个，组1的5个，...
    new_order = [i for g in groups for i in g]
    perm_idx = torch.tensor(new_order, device=sparse.device, dtype=torch.long)

    # 重排（index_select 避免 Python 列表高级索引的额外开销）
    sparse_reordered = sparse.index_select(0, perm_idx)

    return sparse_reordered, groups, group_sums, perm_idx

def parse_args():
    parser = argparse.ArgumentParser(
        description="Test hybrid attention with configurable sequence length"
    )
    parser.add_argument(
        "--seqlen", type=int, default=75648, help="sequence length (default: 1024)"
    )
    parser.add_argument(
        "--use_bwd",
        action="store_true",
        help="whether to test backward pass (default: False)",
    )
    parser.add_argument(
        "--sp_ulysses_degree",
        type=int,
        default=None,
        help="sp_ulysses_degree (default: world_size)",
    )
    parser.add_argument(
        "--ring_impl_type",
        type=str,
        default="basic",
        choices=["basic", "zigzag", "basic_flashinfer"],
        help="ring implementation type (default: basic)",
    )
    parser.add_argument(
        "--causal",
        action="store_true",
        help="whether to use causal attention (default: False)",
    )
    parser.add_argument(
        "--attn_impl",
        type=str,
        default="torch",
        choices=[
            "torch",
            "fa",
            "fa3",
            "flashinfer",
            "sage_fp16",
            "sage_fp8",
            "sparse_sage",
            "sage_fp8_sm90",
            "sage_fp16_triton",
            "sage_auto",
            "paro",
        ],
        help="attention implementation type (default: torch)",
    )
    parser.add_argument(
        "--sparse_sage_l1",
        type=float,
        default=0.07,
        help="l1 for sparse sage attention (default: 0.07)",
    )
    parser.add_argument(
        "--sparse_sage_pv_l1",
        type=float,
        default=0.08,
        help="pv_l1 for sparse sage attention (default: 0.08)",
    )
    parser.add_argument(
        "--sparse_sage_tune_mode",
        action="store_true",
        default=False,
        help="enable tune mode for sparse sage attention (default: False)",
    )
    parser.add_argument(
        "--sparse_sage_tune_path",
        type=str,
        default="./sparsesage_autotune.pt",
        help="path to the sparse sage autotune results (default: ./sparsesage_autotune.pt)",
    )
    return parser.parse_args()


def log(msg, a, rank0_only=False):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if rank0_only:
        if rank == 0:
            print(
                f"[Rank#0] {msg}: "
                f"max {a.abs().max().item()}, "
                f"mean {a.abs().mean().item()}",
                flush=True,
            )
        return

    for i in range(world_size):
        if i == rank:
            if rank == 0:
                print(f"{msg}:")
            print(
                f"[Rank#{rank}] "
                f"max {a.abs().max().item()}, "
                f"mean {a.abs().mean().item()}",
                flush=True,
            )
        dist.barrier()


# test it with:
# torchrun --nproc_per_node=4  test/test_hybrid_attn.py
if __name__ == "__main__":
    args = parse_args()

    torch.random.manual_seed(0)

    dist.init_process_group("nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Inference mainly uses fp16; ROCM flash attention with bf16 precision is slightly larger, will be fixed soon
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{rank}")

    batch_size = 1
    seqlen = args.seqlen
    nheads = 40
    d = 128
    dropout_p = 0
    causal = args.causal
    deterministic = False

    use_bwd = args.use_bwd

    assert seqlen % world_size == 0
    assert d % 8 == 0

    ring_impl_type = args.ring_impl_type

    # Prepare inputs
    q = torch.randn(
        batch_size,
        seqlen,
        nheads,
        d,
        device=device,
        dtype=dtype,
        requires_grad=True if use_bwd else False,
    )
    k = torch.randn(
        batch_size,
        seqlen,
        nheads,
        d,
        device=device,
        dtype=dtype,
        requires_grad=True if use_bwd else False,
    )
    v = torch.randn(
        batch_size,
        seqlen,
        nheads,
        d,
        device=device,
        dtype=dtype,
        requires_grad=True if use_bwd else False,
    )

    # random sparse mask with given sparse ratio
    sparse_ratio = 1
    shape = (batch_size, seqlen // 64, nheads, seqlen // 64)
    sparse = (torch.rand(shape, device=device) < sparse_ratio)
    # half = shape[1] // 2
    # sparse[:,:half,:,:] = True
    # sparse[:,half:,:,:] = False
    # # ulysses 8/4/2 cut the third dimension
    # # ring 8 cut the second dimension

    sparse_data = torch.load("/mnt/public/ns-t-te-b905754427352261-427-bk/fs/home/xieruiqi/diffuser-dev520/examples/wan/logs/calib_data/720p/sparse_plan_expanded.pth", map_location='cpu', weights_only=True)
    sparse = sparse_data['sparse'][0, 2, :, :, :]
    sparse_reordered, groups, group_sums, perm_idx = greedy_partition_and_rearrange(sparse)
    sparse_reordered = sparse_reordered.unsqueeze(0)  
    sparse_reordered = sparse_reordered.transpose(1, 2) 
    sparse_reordered = sparse_reordered.to(device) 
    sparse_reordered = sparse_reordered.contiguous()

    dout = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype)

    dist.broadcast(q, src=0)
    dist.broadcast(k, src=0)
    dist.broadcast(v, src=0)
    dist.broadcast(sparse_reordered, src=0)
    dist.broadcast(dout, src=0)

    # prepare process group for hybrid sequence parallelism
    use_ring_low_dim = True

    sp_ulysses_degree = (
        args.sp_ulysses_degree if args.sp_ulysses_degree is not None else world_size
    )
    sp_ring_degree = world_size // sp_ulysses_degree

    print(
        f"rank {rank}, sp_ulysses_degree: {sp_ulysses_degree}, sp_ring_degree: {sp_ring_degree}"
    )

    set_seq_parallel_pg(sp_ulysses_degree, sp_ring_degree, rank, world_size)

    # Use EXTRACT_FUNC_DICT to shard the tensors
    local_q = (
        EXTRACT_FUNC_DICT[ring_impl_type](
            q, rank, world_size=world_size, rd=sp_ring_degree, ud=sp_ulysses_degree
        )
        .detach()
        .clone()
    )

    local_k = (
        EXTRACT_FUNC_DICT[ring_impl_type](
            k, rank, world_size=world_size, rd=sp_ring_degree, ud=sp_ulysses_degree
        )
        .detach()
        .clone()
    )

    local_v = (
        EXTRACT_FUNC_DICT[ring_impl_type](
            v, rank, world_size=world_size, rd=sp_ring_degree, ud=sp_ulysses_degree
        )
        .detach()
        .clone()
    )

    local_sparse = (
        EXTRACT_FUNC_DICT[ring_impl_type](
            sparse_reordered, rank, world_size=world_size, rd=sp_ring_degree, ud=sp_ulysses_degree
        )
        .detach()
        .clone()
    )
    # print(f"rank:{rank},sum of local sparse: {local_sparse.sum()}")


    if use_bwd:
        local_q.requires_grad = True
        local_k.requires_grad = True
        local_v.requires_grad = True

    # Map argument to AttnType enum
    attn_impl_map = {
        "torch": AttnType.TORCH,
        "fa": AttnType.FA,
        "fa3": AttnType.FA3,
        "flashinfer": AttnType.FLASHINFER,
        "sage_fp16": AttnType.SAGE_FP16,
        "sage_fp8": AttnType.SAGE_FP8,
        "sage_fp8_sm90": AttnType.SAGE_FP8_SM90,
        "sage_fp16_triton": AttnType.SAGE_FP16_TRITON,
        "sage_auto": AttnType.SAGE_AUTO,
        "sparse_sage": AttnType.SPARSE_SAGE,
        "paro": AttnType.PARO,
    }

    if args.attn_impl == "sparse_sage":
        if use_bwd:
            raise RuntimeError("Sparse Sage attention does not support backward pass")
        from spas_sage_attn.autotune import (
            SparseAttentionMeansim,
            load_sparse_attention_state_dict,
        )

        attn_processor = SparseAttentionMeansim(
            l1=args.sparse_sage_l1, pv_l1=args.sparse_sage_pv_l1, tune_pv=True
        )
    else:
        attn_processor = None

    usp_attn = LongContextAttention(
        ring_impl_type=ring_impl_type,
        attn_type=attn_impl_map[args.attn_impl],
        attn_processor=attn_processor,
    )

    if args.attn_impl == "sparse_sage":
        from spas_sage_attn.autotune import (
            SparseAttentionMeansim,
            extract_sparse_attention_state_dict,
        )
        if not args.sparse_sage_tune_mode:
            saved_state_dict = torch.load(
                args.sparse_sage_tune_path + f".rank{dist.get_rank()}"
            )
            load_sparse_attention_state_dict(
                usp_attn, saved_state_dict, multigpu=True, verbose=True
            )
        else:
            # os.environ["sparse_sage_tune_mode"] = "1"
            saved_state_dict = extract_sparse_attention_state_dict(
                usp_attn, verbose=True
            )
            torch.save(
                saved_state_dict, args.sparse_sage_tune_path + f".rank{dist.get_rank()}"
            )

    if rank == 0:
        print("#" * 30)
        print("# ds-ulysses forward:")
        print("#" * 30)

    # common test parameters
    window_size = (-1, -1)
    alibi_slopes, attn_bias = None, None
    dropout_mask = None

    print(f"before usp attn forward: {local_q.shape} {local_k.shape} {local_v.shape}")

    # usp attn forward
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for _ in range(3):
            _ = usp_attn(
                local_q,
                local_k,
                local_v,
                dropout_p=dropout_p,
                causal=causal,
                window_size=window_size,
                softcap=0.0,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_attn_probs=True,
                sparse=local_sparse,
            )
        torch.cuda.synchronize()
        with record_function("usp_attn_forward"):
            local_out = usp_attn(
                local_q,
                local_k,
                local_v,
                dropout_p=dropout_p,
                causal=causal,
                window_size=window_size,
                softcap=0.0,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_attn_probs=True,
                sparse=local_sparse,
            )
        torch.cuda.synchronize()
        for _ in range(3):
            out_ref, _, _ = flash_attn_func(
                q,
                k,
                v,
                dropout_p=dropout_p,
                causal=causal,
                window_size=window_size,
                softcap=0.0,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_attn_probs=True,
            )
        torch.cuda.synchronize()
        with record_function("flash_attn_func"):
            out_ref, _, _ = flash_attn_func(
                q,
                k,
                v,
                dropout_p=dropout_p,
                causal=causal,
                window_size=window_size,
                softcap=0.0,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_attn_probs=True,
            )
        torch.cuda.synchronize()
                

    torch.cuda.synchronize()
    torch.distributed.barrier()  # 同步所有卡
    start = time.perf_counter()

    _ = usp_attn(
                local_q,
                local_k,
                local_v,
                dropout_p=dropout_p,
                causal=causal,
                window_size=window_size,
                softcap=0.0,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_attn_probs=True,
                sparse=local_sparse,
            )

    torch.cuda.synchronize()
    torch.distributed.barrier()  # 确保所有卡都完成
    elapsed = time.perf_counter() - start

    flops = 4 * nheads * batch_size * d * seqlen * seqlen / (2 if causal else 1)
    print(f"[Rank#{rank}] usp attn forward throughout: {1/elapsed}iters/sec, time: {elapsed*1e3}ms, flops: {flops/elapsed*1e-12}TFLOPs/s\n")

    # extract local dout
    local_dout = (
        EXTRACT_FUNC_DICT[ring_impl_type](
            dout, rank, world_size=world_size, rd=sp_ring_degree, ud=sp_ulysses_degree
        )
        .detach()
        .clone()
    )

    max_memory = torch.cuda.max_memory_allocated(device) / (
        1024 * 1024
    )  # Convert to MB
    print(f"[Rank#{rank}] Maximum GPU memory used: {max_memory:.2f} MB")
    torch.cuda.reset_peak_memory_stats(device)  # Reset stats

    if rank == 0:
        print("#" * 30)
        print("# ds-ulysses backward:")
        print("#" * 30)

    # usp attn backward
    if use_bwd:
        local_out.backward(local_dout)

    dist.barrier()

    if rank == 0:
        print("#" * 30)
        print("# local forward:")
        print("#" * 30)
    # reference, a local flash attn
    out_ref, _, _ = flash_attn_func(
        q,
        k,
        v,
        dropout_p=dropout_p,
        causal=causal,
        window_size=window_size,
        softcap=0.0,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=True,
    )

    # out_pt_ref, attn_pt_ref = attention_ref(
    #     q,
    #     k,
    #     v,
    #     None,
    #     None,
    #     attn_bias,
    #     dropout_p,
    #     dropout_mask,
    #     causal=causal,
    #     window_size=window_size,
    # )

    if rank == 0:
        print("#" * 30)
        print("# local forward:")
        print("#" * 30)

    if use_bwd:
        out_ref.backward(dout)

    dist.barrier()

    # check correctness
    # When checking correctness, use EXTRACT_FUNC_DICT for reference outputs
    local_out_ref = EXTRACT_FUNC_DICT[ring_impl_type](
        out_ref, rank, world_size=world_size, rd=sp_ring_degree, ud=sp_ulysses_degree
    )
    # local_out_pt_ref = EXTRACT_FUNC_DICT[ring_impl_type](
    #     out_pt_ref, rank, world_size=world_size, rd=sp_ring_degree, ud=sp_ulysses_degree
    # )

    log("local (rank) out", local_out, rank0_only=True)
    log("out (distributed) - out_ref (non-distributed) diff", local_out_ref - local_out)

    # log("out_ref (non-distributed) - out_pt_ref (gpu) diff", local_out_ref - local_out_pt_ref)

    # torch.testing.assert_close(local_out, local_out_ref, atol=1e-1, rtol=0)
    # torch.testing.assert_close(out_ref, out_pt_ref, atol=1e-2, rtol=0)


    if use_bwd:
        local_dq_ref = EXTRACT_FUNC_DICT[ring_impl_type](
            q.grad, rank, world_size=world_size, rd=sp_ring_degree, ud=sp_ulysses_degree
        )
        log("load_dq", local_q.grad)
        log("dq diff", local_dq_ref - local_q.grad)

        local_dk_ref = EXTRACT_FUNC_DICT[ring_impl_type](
            k.grad, rank, world_size=world_size, rd=sp_ring_degree, ud=sp_ulysses_degree
        )
        log("load_dk", local_k.grad)
        log("dk diff", local_dk_ref - local_k.grad)

        local_dv_ref = EXTRACT_FUNC_DICT[ring_impl_type](
            v.grad, rank, world_size=world_size, rd=sp_ring_degree, ud=sp_ulysses_degree
        )
        log("load_dv", local_v.grad)
        log("dv diff", local_dv_ref - local_v.grad)

    if dist.is_initialized():
        dist.destroy_process_group()
    
    if rank == 0 or rank == 7:
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
        prof.export_chrome_trace(f"profile/profile_rank{rank}_ulysses{sp_ulysses_degree}ring{sp_ring_degree}_real.json")  # 可选：导出火焰图
