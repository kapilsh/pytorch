# Owner(s): ["oncall: distributed"]

import importlib.util
import random
from unittest import skipUnless

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.testing._internal.common_distributed import (
    MultiProcContinuousTest,
    requires_nccl_version,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import requires_cuda_p2p_access, run_tests


HAS_CUTE = importlib.util.find_spec("cutlass") is not None
if HAS_CUTE:
    from cuda.bindings import runtime as cudart

    from torch.distributed._symmetric_memory import _nccl4py_pool
    from torch.distributed._symmetric_memory._cute_all_reduce import (
        _release_workspaces,
        one_shot_all_reduce,
        two_shot_all_reduce,
    )

    ALGOS = {"one_shot": one_shot_all_reduce, "two_shot": two_shot_all_reduce}

PROTOCOLS = ("ll", "lamport")
DTYPES = (torch.bfloat16, torch.float16, torch.float32)


class _CuteAllReduceTests:
    """Shared tests; subclasses pick the symmetric-memory backend and inbox allocator."""

    BACKEND = "CUDA"
    ALLOCATOR = "symm_mem"

    @classmethod
    def _init_pg(cls, rank, world_size, rdvz_file):
        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)
        dist.init_process_group(
            backend="nccl",
            world_size=world_size,
            rank=rank,
            store=dist.FileStore(rdvz_file, world_size),
            timeout=cls.timeout,
            device_id=device,
        )
        if symm_mem.get_backend(device) != cls.BACKEND:
            symm_mem.set_backend(cls.BACKEND)
        cls.pg = dist.distributed_c10d._get_default_group()

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.rank)

    def _multimem_options(self) -> tuple[bool, ...]:
        if self.ALLOCATOR == "nccl4py":
            probe = _nccl4py_pool.empty(16, device=self.device)
            window = _nccl4py_pool.NcclWindow(probe, self.pg)
            has_mc = window.multicast_ptr != 0
            window.close()
        else:
            probe = symm_mem.empty(16, device=self.device)
            has_mc = symm_mem.rendezvous(probe, self.pg.group_name).multicast_ptr != 0
        return (False, True) if has_mc else (False,)

    def _rank_inputs(self, numel: int, dtype: torch.dtype, seed: int):
        xs = []
        for r in range(self.world_size):
            gen = torch.Generator(device=self.device).manual_seed(seed * 100 + r)
            x = torch.randn(numel, generator=gen, device=self.device).to(dtype)
            x[::7] = -0.0  # must not be mistaken for the lamport sentinel
            xs.append(x)
        return xs

    @staticmethod
    def _expected(xs, dtype: torch.dtype) -> torch.Tensor:
        acc = xs[0].float()
        for x in xs[1:]:
            acc = acc + x.float()
        return acc.to(dtype)

    @skip_if_lt_x_gpu(2)
    def test_all_reduce(self) -> None:
        group = self.pg.group_name
        for proto in PROTOCOLS:
            for algo, fn in ALGOS.items():
                for mc in self._multimem_options():
                    for dtype in DTYPES:
                        for seed, numel in enumerate(
                            (8, 1000 * 8, 4096 * 8 + 24, 1 << 20)
                        ):
                            xs = self._rank_inputs(numel, dtype, seed)
                            out = fn(
                                xs[self.rank],
                                group,
                                protocol=proto,
                                allocator=self.ALLOCATOR,
                                use_multimem=mc,
                            )
                            self.assertEqual(
                                out,
                                self._expected(xs, dtype),
                                atol=0,
                                rtol=0,
                                msg=f"{proto} {algo} mc={mc} {dtype} numel={numel}",
                            )
        _release_workspaces()

    @skip_if_lt_x_gpu(2)
    def test_in_place(self) -> None:
        group = self.pg.group_name
        for proto in PROTOCOLS:
            for fn in ALGOS.values():
                xs = self._rank_inputs(4096, torch.bfloat16, 0)
                x = xs[self.rank].clone()
                out = fn(x, group, protocol=proto, allocator=self.ALLOCATOR, out=x)
                self.assertEqual(out.data_ptr(), x.data_ptr())
                self.assertEqual(x, self._expected(xs, torch.bfloat16), atol=0, rtol=0)
        _release_workspaces()

    @skip_if_lt_x_gpu(2)
    def test_reuse_under_skew(self) -> None:
        # Many back-to-back calls with no host sync and per-rank device delays, so
        # fast ranks run ahead into the next call while peers are still reading.
        group = self.pg.group_name
        sizes = (8, 4096, 40000, 1 << 18)
        mcs = self._multimem_options()
        for proto in PROTOCOLS:
            for fn in ALGOS.values():
                for mc in mcs:
                    fn(
                        torch.zeros(
                            max(sizes), device=self.device, dtype=torch.bfloat16
                        ),
                        group,
                        protocol=proto,
                        allocator=self.ALLOCATOR,
                        use_multimem=mc,
                    )
        schedule = random.Random(0)
        skew = random.Random(1 + self.rank)
        results = []
        for it in range(200):
            proto = schedule.choice(PROTOCOLS)
            algo = schedule.choice(list(ALGOS))
            mc = schedule.choice(mcs)
            numel = schedule.choice(sizes)
            base = torch.arange(numel, device=self.device) % 13 - 6.0
            x = (base * (self.rank + 1) + it % 5).to(torch.bfloat16)
            if skew.random() < 0.5:
                torch.cuda._sleep(skew.randint(0, 1_000_000))
            out = ALGOS[algo](
                x, group, protocol=proto, allocator=self.ALLOCATOR, use_multimem=mc
            )
            ws = self.world_size
            expected = (base * (ws * (ws + 1) // 2) + ws * (it % 5)).to(torch.bfloat16)
            results.append((it, proto, algo, mc, out, expected))
        for it, proto, algo, mc, out, expected in results:
            self.assertEqual(
                out, expected, atol=0, rtol=0, msg=f"iter {it}: {proto} {algo} mc={mc}"
            )
        _release_workspaces()

    @skip_if_lt_x_gpu(2)
    def test_cuda_graph(self) -> None:
        group = self.pg.group_name
        numel = 1 << 16
        ws = self.world_size
        for proto in PROTOCOLS:
            for fn in ALGOS.values():
                x = torch.full((numel,), self.rank + 1.0, device=self.device)
                out = torch.empty_like(x)
                fn(
                    x, group, protocol=proto, allocator=self.ALLOCATOR, out=out
                )  # grow the inbox eagerly
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
                    for _ in range(4):
                        fn(x, group, protocol=proto, allocator=self.ALLOCATOR, out=out)
                for _ in range(5):
                    out.zero_()
                    graph.replay()
                    torch.cuda.synchronize()
                    self.assertTrue(bool((out == ws * (ws + 1) / 2).all()))
        _release_workspaces()

    @skip_if_lt_x_gpu(2)
    def test_invalid_size(self) -> None:
        x = torch.ones(3, device=self.device, dtype=torch.bfloat16)
        for proto in PROTOCOLS:
            with self.assertRaisesRegex(ValueError, "multiple of"):
                one_shot_all_reduce(
                    x, self.pg.group_name, protocol=proto, allocator=self.ALLOCATOR
                )


@skipUnless(HAS_CUTE, "requires CuTeDSL (nvidia-cutlass-dsl)")
@requires_cuda_p2p_access()
class CuteAllReduceCUDABackendTest(_CuteAllReduceTests, MultiProcContinuousTest):
    BACKEND = "CUDA"


@skipUnless(HAS_CUTE, "requires CuTeDSL (nvidia-cutlass-dsl)")
@requires_nccl_version((2, 28), "NCCL symmetric memory peer pointers need 2.28+")
@requires_cuda_p2p_access()
class CuteAllReduceNCCLBackendTest(_CuteAllReduceTests, MultiProcContinuousTest):
    BACKEND = "NCCL"


@skipUnless(HAS_CUTE, "requires CuTeDSL (nvidia-cutlass-dsl)")
@requires_nccl_version((2, 28), "NCCL symmetric windows need 2.28+")
@requires_cuda_p2p_access()
class CuteAllReduceNccl4pyAllocatorTest(_CuteAllReduceTests, MultiProcContinuousTest):
    ALLOCATOR = "nccl4py"

    @skip_if_lt_x_gpu(2)
    def test_window_peer_pointers(self) -> None:
        t = _nccl4py_pool.empty(1024, dtype=torch.int32, device=self.device)
        window = _nccl4py_pool.NcclWindow(t, self.pg)
        self.assertEqual(len(window.buffer_ptrs), self.world_size)
        t.fill_(self.rank + 1)
        torch.cuda.synchronize()
        dist.barrier()
        for peer, ptr in enumerate(window.buffer_ptrs):
            got = torch.empty_like(t)
            (err,) = cudart.cudaMemcpy(
                got.data_ptr(),
                ptr,
                t.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
            )
            self.assertEqual(err, cudart.cudaError_t.cudaSuccess)
            self.assertTrue(bool((got == peer + 1).all()), f"peer {peer}")
        dist.barrier()
        window.close()
        del t
        _nccl4py_pool.release_pool()

    @skip_if_lt_x_gpu(2)
    def test_window_rejects_foreign_tensor(self) -> None:
        with self.assertRaisesRegex(ValueError, "_nccl4py_pool.empty"):
            _nccl4py_pool.NcclWindow(torch.empty(16, device=self.device), self.pg)


if __name__ == "__main__":
    run_tests()
