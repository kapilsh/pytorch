#!/usr/bin/env python3
"""Benchmark the CuTeDSL barrier-free all-reduce kernels against the built-in
symmetric-memory ops and NCCL. Times CUDA graph replay (us/op) and checks the
replayed outputs.

Usage:
torchrun --nproc-per-node 8 benchmarks/distributed/bench_symm_mem_cute_all_reduce.py \
    --backend CUDA --dtype bfloat16 --min-bytes 1024 --max-bytes 16777216

Add --allocator nccl4py to put the inbox in an nccl4py-backed MemPool. For multicast
variants on NCCL-allocated inboxes, also set NCCL_NVLS_ENABLE=1.
"""

import argparse
import functools
import os

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.distributed._symmetric_memory import _nccl4py_pool
from torch.distributed._symmetric_memory._cute_all_reduce import (
    _release_workspaces,
    one_shot_all_reduce,
    two_shot_all_reduce,
)


def time_graph(fn, ops_per_graph: int, replays: int) -> float:
    for _ in range(3):
        fn()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        for _ in range(ops_per_graph):
            fn()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / (replays * ops_per_graph)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="CUDA", choices=["CUDA", "NCCL"])
    parser.add_argument(
        "--allocator", default="symm_mem", choices=["symm_mem", "nccl4py"]
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--min-bytes", type=int, default=1024)
    parser.add_argument("--max-bytes", type=int, default=16 << 20)
    parser.add_argument("--ops-per-graph", type=int, default=50)
    parser.add_argument("--replays", type=int, default=20)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    if args.backend != symm_mem.get_backend(device):
        symm_mem.set_backend(args.backend)
    group = dist.group.WORLD.group_name
    world_size = dist.get_world_size()
    dtype = getattr(torch, args.dtype)
    has_mc = (
        symm_mem.rendezvous(symm_mem.empty(16, device=device), group).multicast_ptr != 0
    )
    if args.allocator == "nccl4py":
        window = _nccl4py_pool.NcclWindow(
            _nccl4py_pool.empty(16, device=device), dist.group.WORLD
        )
        has_mc = window.multicast_ptr != 0
        window.close()

    sizes = []
    nbytes = args.min_bytes
    while nbytes <= args.max_bytes:
        sizes.append(nbytes)
        nbytes *= 2
    numels = [n // dtype.itemsize for n in sizes]
    expected = world_size * (world_size + 1) / 2

    cute_ops = {}
    for algo, fn in (
        ("one_shot", one_shot_all_reduce),
        ("two_shot", two_shot_all_reduce),
    ):
        for proto in ("ll", "lamport"):
            for mc in (False, True) if has_mc else (False,):
                name = f"cute_{proto}_{algo}" + ("_mc" if mc else "")
                cute_ops[name] = functools.partial(
                    fn,
                    group=group,
                    protocol=proto,
                    use_multimem=mc,
                    allocator=args.allocator,
                )
    torch_ops = {
        "torch_one_shot": lambda t: torch.ops.symm_mem.one_shot_all_reduce(
            t, "sum", group
        ),
        "torch_two_shot": lambda t: torch.ops.symm_mem.two_shot_all_reduce_(
            t, "sum", group
        ),
    }
    if has_mc:
        torch_ops["torch_mm_one_shot"] = (
            lambda t: torch.ops.symm_mem.multimem_one_shot_all_reduce(t, "sum", group)
        )
        torch_ops["torch_mm_all_reduce"] = (
            lambda t: torch.ops.symm_mem.multimem_all_reduce_(t, "sum", group)
        )

    # grow every cute inbox to the max size eagerly: growth cannot be captured
    for op in cute_ops.values():
        op(torch.ones(numels[-1], dtype=dtype, device=device))
    torch.cuda.synchronize()

    results: dict[str, list[str]] = {}
    for numel in numels:
        x = torch.full((numel,), rank + 1, dtype=dtype, device=device)
        for name, op in cute_ops.items():
            out = torch.empty_like(x)
            us = time_graph(
                lambda op=op, out=out: op(x, out=out),
                args.ops_per_graph,
                args.replays,
            )
            ok = bool((out == expected).all())
            results.setdefault(name, []).append(f"{us:.2f}" if ok else "WRONG")
        for name, op in torch_ops.items():
            t = symm_mem.empty(numel, dtype=dtype, device=device)
            symm_mem.rendezvous(t, group)
            try:
                us = time_graph(
                    lambda op=op, t=t: op(t), args.ops_per_graph, args.replays
                )
                results.setdefault(name, []).append(f"{us:.2f}")
            except RuntimeError:
                results.setdefault(name, []).append("n/a")
        xn = x.clone()
        us = time_graph(lambda: dist.all_reduce(xn), args.ops_per_graph, args.replays)
        results.setdefault("nccl", []).append(f"{us:.2f}")

    if rank == 0:
        print(
            f"\nall_reduce us/op (CUDA graph), backend={args.backend}, "
            f"allocator={args.allocator}, "
            f"world_size={world_size}, dtype={args.dtype}, multicast={has_mc}"
        )
        width = max(len(n) for n in results) + 2
        print("bytes".ljust(width) + "".join(f"{s:>10}" for s in sizes))
        for name, row in results.items():
            print(name.ljust(width) + "".join(f"{v:>10}" for v in row))
    # symmetric tensors must be freed while the communicator is still alive
    x = xn = t = None
    _release_workspaces()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
