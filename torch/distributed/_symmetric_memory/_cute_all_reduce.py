"""Barrier-free one-shot / two-shot all-reduce kernels in CuTeDSL over symmetric memory.

Works on any symmetric-memory backend that exposes peer and multicast pointers
(``CUDA`` and ``NCCL``). See Note [Barrier-free all-reduce protocols] for the design.
"""

from __future__ import annotations

import math
from typing import Literal

import cuda.bindings.driver as cudadrv
import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass.cute.runtime import make_ptr

import torch
import torch.distributed as dist
from torch.distributed._symmetric_memory import _resolve_group_name, empty, rendezvous
from torch.distributed.distributed_c10d import _resolve_process_group, ProcessGroup


__all__ = ["one_shot_all_reduce", "two_shot_all_reduce"]

# Note [Barrier-free all-reduce protocols]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Every kernel here is push-based: a rank stores its data straight into its peers'
# symmetric "inbox", and a reader learns that a line has landed from the line itself.
# There is no barrier and no separate signal pad. The input and output are regular
# tensors; the inbox is the only symmetric allocation.
#
# Two ways to tell that a 16 B line has landed:
#
# * LL (NCCL's low-latency protocol): line = {d0, flag, d1, flag}, i.e. 8 B of payload
#   with a 32-bit epoch flag stored twice, written as one 16 B store. A reader accepts
#   the line once both flags equal the current epoch; checking both flags guards
#   against a torn write. Payload efficiency is 50%. The inbox is double-buffered on
#   (epoch & 1). Stale lines carry an older epoch, so they never match and no cleanup
#   is needed.
#
# * Lamport (negative-zero sentinel): line = 16 B of payload. The inbox is pre-filled
#   with -0.0, and every element a writer pushes has -0.0 replaced by +0.0. That
#   change cannot affect a sum, and it means a pushed value never equals the
#   sentinel, so "every element != -0.0" means the line has landed. Payload
#   efficiency is 100%, but the inbox must be put back to -0.0 before it is reused.
#   It is triple-buffered: call k writes slot k % 3 and clears slot (k + 2) % 3, the
#   one call k - 1 used.
#   - Clearing that slot is safe because this rank has already read every line peers
#     wrote into it.
#   - No peer can write into it again before call k + 2, which needs this rank's
#     call k + 1 data, which is only pushed after call k's kernel, clear included,
#     has finished.
#   The state tensor records how many lines each slot used, so the clear only touches
#   lines that were actually written.
#
# The algorithms:
#
# * one_shot: each rank pushes its whole input into region `rank` of every peer's
#   inbox, then reads all `world_size` regions of its own inbox and reduces locally.
#   One cross-rank hop.
#
# * two_shot: reduce-scatter, then all-gather, with no barrier between them. Rank r
#   pushes chunk d to rank d's reduce-scatter region `r`. Rank r reduces chunk r and
#   pushes the result to region `r` of every peer's all-gather area. Each rank then
#   reads all chunks out of its all-gather area. Thread t handles the same lines in
#   every phase on every rank, and the waits only depend on earlier phases, so this
#   cannot deadlock as long as the whole grid is resident (grids are capped at
#   _MAX_BLOCKS). The last chunk is padded with zeros so every inbox line gets
#   written.
#
# With use_multimem=True, the all-gather style broadcast (one_shot's push and
# two_shot's all-gather push) becomes a single multimem.st to the multicast address
# instead of one store per peer.
#
# Reduction runs in fp32, summing sources in rank order, so every rank ends up with
# bitwise identical output. The epoch/step counter lives on the device, so it keeps
# working under CUDA graph replay. Each block's thread 0 increments an arrival
# counter when the block finishes, and the last block to arrive advances the counter.
# That way no block can see the next call's epoch while other blocks are still
# running. Ranks stay in lockstep because every rank makes the same calls.

I32, I64, U32, F32 = cutlass.Int32, cutlass.Int64, cutlass.Uint32, cutlass.Float32
_GMEM = cute.AddressSpace.gmem
_THREADS = 256
_MAX_BLOCKS = 24
_LINE_BYTES = 16

_FP32, _BF16, _FP16 = 0, 1, 2
_DTYPE_CODE = {torch.float32: _FP32, torch.bfloat16: _BF16, torch.float16: _FP16}
_ELEM_TYPE = {_FP32: F32, _BF16: cutlass.BFloat16, _FP16: cutlass.Float16}
# bit pattern of -0.0, compared on an unsigned view of the element width
_NEG_ZERO_BITS = {
    _FP32: (U32, 0x80000000),
    _BF16: (cutlass.Uint16, 0x8000),
    _FP16: (cutlass.Uint16, 0x8000),
}

_LL_ONE_SHOT, _LL_TWO_SHOT, _LAMPORT_ONE_SHOT, _LAMPORT_TWO_SHOT = range(4)
_KERNEL_KIND = {
    ("ll", "one_shot"): _LL_ONE_SHOT,
    ("ll", "two_shot"): _LL_TWO_SHOT,
    ("lamport", "one_shot"): _LAMPORT_ONE_SHOT,
    ("lamport", "two_shot"): _LAMPORT_TWO_SHOT,
}

# state tensor layout (int32): [epoch_or_step, arrivals, used[0], used[1], used[2]]
_STATE_NUMEL = 8


# ---- memory ops -----------------------------------------------------------------
# A 16 B line is a (4,) vector of u32 words. Inbox traffic uses relaxed system-scope
# vector loads/stores so a spinning reader always sees fresh data; the input goes
# through the read-only cache and the output uses plain stores.


def _ptr(addr, align):
    return cute.make_ptr(U32, addr, _GMEM, assumed_align=align)


def _line_addr(base, line):
    return base + I64(line) * _LINE_BYTES


def _load(addr, n, **order):
    """Load `n` u32 words from a 4 * n B aligned address as a vector."""
    vec = ir.VectorType.get([n], U32.mlir_type)
    return cute.TensorSSA(cute.arch.load(_ptr(addr, 4 * n), vec, **order), (n,), U32)


def _ld_input(addr, n):
    """Load `n` input words through the read-only (non-coherent) cache."""
    # inline PTX: cute.arch.load has no .nc variant
    regs = ", ".join(f"{{$w{k}}}" for k in range(n))
    vals = cute.arch.inline_ptx(
        f"ld.global.nc.v{n}.u32 {{{regs}}}, [{{$r0}}];",
        write_only_types=[U32] * n,
        read_only_args=[_ptr(addr, 4 * n)],
    )
    return _words(*vals)


def _store(addr, words, **order):
    cute.arch.store(_ptr(addr, 4 * words.shape[0]), words.ir_value(), **order)


def _ld_line(base, line):
    return _load(_line_addr(base, line), 4, sem="relaxed", scope="sys")


def _words(*vals):
    """Pack scalar u32 words into a vector."""
    frag = cute.make_rmem_tensor((len(vals),), U32)
    for k, v in enumerate(vals):
        frag[k] = v
    return frag.load()


def _push(peers, mc_base, line, words, use_mc):
    """Store one 16 B line at `line` of every peer's inbox."""
    if use_mc:
        # inline PTX: CuTeDSL has no relaxed.sys multimem store
        cute.arch.inline_ptx(
            "multimem.st.relaxed.sys.global.v4.f32 [{$r0}], {{$r1}, {$r2}, {$r3}, {$r4}};",
            read_only_args=[
                _ptr(_line_addr(mc_base, line), _LINE_BYTES),
                *(words[k] for k in range(4)),
            ],
        )
    else:
        for peer in peers:
            _store(_line_addr(peer, line), words, sem="relaxed", scope="sys")


def _reduce(payloads, dt):
    """Sum per-source payload vectors elementwise in fp32, rounding once at the end."""
    elem = _ELEM_TYPE[dt]
    acc = payloads[0].bitcast(elem).to(F32)
    for p in payloads[1:]:
        acc = acc + p.bitcast(elem).to(F32)
    return acc.to(elem).bitcast(U32)


# ---- LL protocol ----------------------------------------------------------------


@cute.jit
def _ll_recv(base: I64, line0: I32, stride: I32, eu: U32, nsrc: cutlass.Constexpr[int]):
    """Payload (2 words) of line `line0 + s * stride` for every source s, once published.

    All loads are issued before the first spin so cross-rank latency overlaps."""
    lines = [_ld_line(base, line0 + s * stride) for s in range(nsrc)]
    payloads = []
    for s in cutlass.range_constexpr(nsrc):
        line = lines[s]
        while line[1] != eu or line[3] != eu:
            line = _ld_line(base, line0 + s * stride)
        payloads.append(_words(line[0], line[2]))
    return payloads


# ---- Lamport (negative-zero sentinel) protocol ----------------------------------


def _sentinel_word(dt) -> int:
    return 0x80000000 if dt == _FP32 else 0x80008000


def _is_neg_zero(words, dt):
    """Elementwise: is the element -0.0? Compares bits on a 16- or 32-bit view."""
    bits_type, neg_zero = _NEG_ZERO_BITS[dt]
    return words.bitcast(bits_type) == bits_type(neg_zero)


def _scrub(words, dt):
    """Replace every -0.0 element with +0.0."""
    bits_type, _ = _NEG_ZERO_BITS[dt]
    bits = cute.where(_is_neg_zero(words, dt), bits_type(0), words.bitcast(bits_type))
    return bits.bitcast(U32)


@cute.jit
def _lamport_recv(
    base: I64,
    line0: I32,
    stride: I32,
    nsrc: cutlass.Constexpr[int],
    dt: cutlass.Constexpr[int],
):
    """Line `line0 + s * stride` for every source s, once no element is -0.0."""
    lines = [_ld_line(base, line0 + s * stride) for s in range(nsrc)]
    out = []
    for s in cutlass.range_constexpr(nsrc):
        line = lines[s]
        while cute.any_(_is_neg_zero(line, dt)):
            line = _ld_line(base, line0 + s * stride)
        out.append(line)
    return out


@cute.jit
def _lamport_clear(
    my_base: I64,
    line0: I32,
    stride: I32,
    used: I32,
    tid: I32,
    nthreads: I32,
    nregions: cutlass.Constexpr[int],
    dt: cutlass.Constexpr[int],
):
    sentinel = cute.full((4,), _sentinel_word(dt), U32)
    for r in cutlass.range_constexpr(nregions):
        j = tid
        while j < used:
            _store(_line_addr(my_base, line0 + r * stride + j), sentinel)
            j = j + nthreads


# ---- shared prologue / epilogue -------------------------------------------------


def _grid_ids():
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    tidx, _, _ = cute.arch.thread_idx()
    gdim, _, _ = cute.arch.grid_dim()
    return tidx, bidx * bdim + tidx, gdim * bdim, gdim


@cute.jit
def _finish(
    state: cute.Pointer,
    tidx: I32,
    gdim: I32,
    next_epoch: I32,
    used_idx: I32,
    used_val: I32,
    track_used: cutlass.Constexpr[bool],
):
    """The last block to finish advances the epoch; it is the only writer of state."""
    cute.arch.sync_threads()
    if tidx == I32(0):
        old = cute.arch.atomic_add(state + 1, I32(1), sem="relaxed", scope="gpu")
        if old == gdim - I32(1):
            cute.arch.store(state + 1, I32(0))
            if cutlass.const_expr(track_used):
                cute.arch.store(state + 2 + used_idx, used_val)
            cute.arch.store(state, next_epoch)


# ---- kernels --------------------------------------------------------------------
# Line/region units are 16 B lines. `stride` is the region capacity (lines) of the
# workspace, fixed for its lifetime; `lines` is this call's line count and `chunk`
# the two-shot per-rank chunk (ceil(lines / ws)).


@cute.kernel
def _ll_one_shot_kernel(
    peer_arr: cute.Tensor,
    mc_base: I64,
    my_base: I64,
    x_base: I64,
    out_base: I64,
    state: cute.Pointer,
    rank: I32,
    lines: I32,
    stride: I32,
    ws: cutlass.Constexpr[int],
    dt: cutlass.Constexpr[int],
    use_mc: cutlass.Constexpr[bool],
):
    tidx, tid, nthreads, gdim = _grid_ids()
    epoch = cute.arch.load(state, I32)
    eu = U32(epoch)
    slot = (epoch % I32(2)) * ws * stride
    peers = [peer_arr[s] for s in range(ws)]
    i = tid
    while i < lines:
        d = _ld_input(x_base + I64(i) * 8, 2)
        _push(
            peers, mc_base, slot + rank * stride + i, _words(d[0], eu, d[1], eu), use_mc
        )
        i = i + nthreads
    i = tid
    while i < lines:
        _store(
            out_base + I64(i) * 8,
            _reduce(_ll_recv(my_base, slot + i, stride, eu, ws), dt),
        )
        i = i + nthreads
    _finish(state, tidx, gdim, epoch + I32(1), I32(0), I32(0), False)


@cute.kernel
def _ll_two_shot_kernel(
    peer_arr: cute.Tensor,
    mc_base: I64,
    my_base: I64,
    x_base: I64,
    out_base: I64,
    state: cute.Pointer,
    rank: I32,
    lines: I32,
    chunk: I32,
    stride: I32,
    ws: cutlass.Constexpr[int],
    dt: cutlass.Constexpr[int],
    use_mc: cutlass.Constexpr[bool],
):
    tidx, tid, nthreads, gdim = _grid_ids()
    epoch = cute.arch.load(state, I32)
    eu = U32(epoch)
    rs = (epoch % I32(2)) * 2 * ws * stride
    ag = rs + ws * stride
    peers = [peer_arr[s] for s in range(ws)]
    # reduce-scatter push: chunk d -> region `rank` of peer d (zero-padded tail)
    i = tid
    while i < chunk:
        for dst in cutlass.range_constexpr(ws):
            g = dst * chunk + i
            d = cute.full((2,), 0, U32)
            if g < lines:
                d = _ld_input(x_base + I64(g) * 8, 2)
            _store(
                _line_addr(peers[dst], rs + rank * stride + i),
                _words(d[0], eu, d[1], eu),
                sem="relaxed",
                scope="sys",
            )
        i = i + nthreads
    # reduce own chunk, all-gather push
    i = tid
    while i < chunk:
        r = _reduce(_ll_recv(my_base, rs + i, stride, eu, ws), dt)
        _push(
            peers, mc_base, ag + rank * stride + i, _words(r[0], eu, r[1], eu), use_mc
        )
        i = i + nthreads
    i = tid
    while i < chunk:
        payloads = _ll_recv(my_base, ag + i, stride, eu, ws)
        for j in cutlass.range_constexpr(ws):
            g = j * chunk + i
            if g < lines:
                _store(out_base + I64(g) * 8, payloads[j])
        i = i + nthreads
    _finish(state, tidx, gdim, epoch + I32(1), I32(0), I32(0), False)


@cute.kernel
def _lamport_one_shot_kernel(
    peer_arr: cute.Tensor,
    mc_base: I64,
    my_base: I64,
    x_base: I64,
    out_base: I64,
    state: cute.Pointer,
    rank: I32,
    lines: I32,
    stride: I32,
    ws: cutlass.Constexpr[int],
    dt: cutlass.Constexpr[int],
    use_mc: cutlass.Constexpr[bool],
):
    tidx, tid, nthreads, gdim = _grid_ids()
    step = cute.arch.load(state, I32)
    stale = (step + I32(2)) % I32(3)
    used = cute.arch.load(state + 2 + stale, I32)
    slot = step * ws * stride
    peers = [peer_arr[s] for s in range(ws)]
    i = tid
    while i < lines:
        w = _scrub(_ld_input(x_base + I64(i) * 16, 4), dt)
        _push(peers, mc_base, slot + rank * stride + i, w, use_mc)
        i = i + nthreads
    # restore the slot used by the previous call while peer data is in flight
    _lamport_clear(my_base, stale * ws * stride, stride, used, tid, nthreads, ws, dt)
    i = tid
    while i < lines:
        _store(
            out_base + I64(i) * 16,
            _reduce(_lamport_recv(my_base, slot + i, stride, ws, dt), dt),
        )
        i = i + nthreads
    _finish(state, tidx, gdim, (step + I32(1)) % I32(3), step, lines, True)


@cute.kernel
def _lamport_two_shot_kernel(
    peer_arr: cute.Tensor,
    mc_base: I64,
    my_base: I64,
    x_base: I64,
    out_base: I64,
    state: cute.Pointer,
    rank: I32,
    lines: I32,
    chunk: I32,
    stride: I32,
    ws: cutlass.Constexpr[int],
    dt: cutlass.Constexpr[int],
    use_mc: cutlass.Constexpr[bool],
):
    tidx, tid, nthreads, gdim = _grid_ids()
    step = cute.arch.load(state, I32)
    stale = (step + I32(2)) % I32(3)
    used = cute.arch.load(state + 2 + stale, I32)
    rs = step * 2 * ws * stride
    ag = rs + ws * stride
    peers = [peer_arr[s] for s in range(ws)]
    i = tid
    while i < chunk:
        for dst in cutlass.range_constexpr(ws):
            g = dst * chunk + i
            w = cute.full((4,), 0, U32)
            if g < lines:
                w = _ld_input(x_base + I64(g) * 16, 4)
            _store(
                _line_addr(peers[dst], rs + rank * stride + i),
                _scrub(w, dt),
                sem="relaxed",
                scope="sys",
            )
        i = i + nthreads
    _lamport_clear(
        my_base, stale * 2 * ws * stride, stride, used, tid, nthreads, 2 * ws, dt
    )
    i = tid
    while i < chunk:
        r = _reduce(_lamport_recv(my_base, rs + i, stride, ws, dt), dt)
        _push(peers, mc_base, ag + rank * stride + i, _scrub(r, dt), use_mc)
        i = i + nthreads
    i = tid
    while i < chunk:
        lines_in = _lamport_recv(my_base, ag + i, stride, ws, dt)
        for j in cutlass.range_constexpr(ws):
            g = j * chunk + i
            if g < lines:
                _store(out_base + I64(g) * 16, lines_in[j])
        i = i + nthreads
    _finish(state, tidx, gdim, (step + I32(1)) % I32(3), step, chunk, True)


@cute.jit
def _launch(
    peers_ptr: cute.Pointer,
    mc_base: I64,
    my_base: I64,
    x_base: I64,
    out_base: I64,
    state: cute.Pointer,
    rank: I32,
    lines: I32,
    chunk: I32,
    stride: I32,
    nblocks: I32,
    kind: cutlass.Constexpr[int],
    ws: cutlass.Constexpr[int],
    dt: cutlass.Constexpr[int],
    use_mc: cutlass.Constexpr[bool],
    stream: cudadrv.CUstream,
):
    peer_arr = cute.make_tensor(peers_ptr, cute.make_layout(ws))
    head = (peer_arr, mc_base, my_base, x_base, out_base, state, rank)
    if cutlass.const_expr(kind == _LL_ONE_SHOT):
        k = _ll_one_shot_kernel(*head, lines, stride, ws, dt, use_mc)
    elif cutlass.const_expr(kind == _LL_TWO_SHOT):
        k = _ll_two_shot_kernel(*head, lines, chunk, stride, ws, dt, use_mc)
    elif cutlass.const_expr(kind == _LAMPORT_ONE_SHOT):
        k = _lamport_one_shot_kernel(*head, lines, stride, ws, dt, use_mc)
    else:
        k = _lamport_two_shot_kernel(*head, lines, chunk, stride, ws, dt, use_mc)
    k.launch(grid=[nblocks, 1, 1], block=[_THREADS, 1, 1], stream=stream)


# ---- host side ------------------------------------------------------------------


class _Workspace:
    """Per (group, device, protocol, algo, dtype) inbox + device-side epoch state.

    The inbox only grows; growing it is collective and must happen outside CUDA graph
    capture (warm up eagerly at the largest size first)."""

    def __init__(
        self, group_name: str, device: torch.device, protocol: str, algo: str, dt: int
    ) -> None:
        self.group_name = group_name
        self.pg = _resolve_process_group(group_name)
        self.device = device
        self.protocol = protocol
        self.dt = dt
        self.rank = self.pg.rank()
        self.world_size = self.pg.size()
        self.nslots = 2 if protocol == "ll" else 3
        self.nregions = self.world_size if algo == "one_shot" else 2 * self.world_size
        self.capacity = 0
        self.buf: torch.Tensor | None = None
        self.hdl = None
        self.state = torch.zeros(_STATE_NUMEL, dtype=torch.int32, device=device)

    def _sync(self) -> None:
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.pg, device_ids=[self.device.index])
        torch.cuda.synchronize(self.device)

    def reserve(self, region_lines: int) -> None:
        if region_lines <= self.capacity:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "cute all_reduce: the symmetric inbox must grow, which cannot happen "
                "during CUDA graph capture; run the largest size once eagerly first."
            )
        capacity = 1 << max(10, math.ceil(math.log2(region_lines)))
        if self.buf is not None:
            # peers may still be pushing into the old inbox
            self._sync()
            self.buf = self.hdl = None
        nbytes = self.nslots * self.nregions * capacity * _LINE_BYTES
        buf = empty(nbytes, dtype=torch.uint8, device=self.device)
        hdl = rendezvous(buf, self.group_name)
        if self.protocol == "ll":
            buf.zero_()
            self.state.zero_()[0] = 1  # epoch 0 would match the zeroed flags
        else:
            sentinel = _sentinel_word(self.dt)
            buf.view(torch.int32).fill_(sentinel - (1 << 32))
            self.state.zero_()
        # every peer's inbox must be initialized before anyone pushes into it
        self._sync()
        self.buf, self.hdl, self.capacity = buf, hdl, capacity
        self.peers_ptr = make_ptr(I64, hdl.buffer_ptrs_dev, _GMEM)
        self.my_base = buf.data_ptr()
        self.mc_base = hdl.multicast_ptr
        self.state_ptr = make_ptr(I32, self.state.data_ptr(), _GMEM)


_workspaces: dict[tuple[object, ...], _Workspace] = {}
_compiled: dict[tuple[int, int, int, bool], object] = {}


def _release_workspaces() -> None:
    """Free all inboxes. Call before destroying the process group: the NCCL backend
    cannot deregister a window once its communicator is gone."""
    _workspaces.clear()


def _all_reduce(
    x: torch.Tensor,
    group: str | ProcessGroup,
    algo: Literal["one_shot", "two_shot"],
    protocol: Literal["ll", "lamport"],
    use_multimem: bool,
    out: torch.Tensor | None,
) -> torch.Tensor:
    if (protocol, algo) not in _KERNEL_KIND:
        raise ValueError(f"unknown protocol {protocol!r}")
    if x.dtype not in _DTYPE_CODE:
        raise ValueError(f"unsupported dtype {x.dtype}; expected fp32, bf16 or fp16")
    x = x.contiguous()
    if out is None:
        out = torch.empty_like(x)
    elif out.shape != x.shape or out.dtype != x.dtype or not out.is_contiguous():
        raise ValueError("out must be a contiguous tensor matching x")
    nbytes = x.numel() * x.element_size()
    if nbytes == 0:
        return out
    payload = 8 if protocol == "ll" else _LINE_BYTES
    if nbytes % payload or x.data_ptr() % _LINE_BYTES or out.data_ptr() % _LINE_BYTES:
        raise ValueError(
            f"{protocol} all_reduce needs 16 B aligned tensors whose size is a "
            f"multiple of {payload} B, got {nbytes} B"
        )

    dt = _DTYPE_CODE[x.dtype]
    group_name = _resolve_group_name(group)
    key = (group_name, x.device, protocol, algo, dt)
    ws_obj = _workspaces.get(key)
    if ws_obj is None:
        ws_obj = _workspaces[key] = _Workspace(group_name, x.device, protocol, algo, dt)
    world = ws_obj.world_size
    lines = nbytes // payload
    chunk = -(-lines // world)
    region_lines = lines if algo == "one_shot" else chunk
    ws_obj.reserve(region_lines)
    if use_multimem and not ws_obj.mc_base:
        raise RuntimeError(
            "use_multimem=True but the symmetric inbox has no multicast address "
            "(no NVLS support, or NCCL backend with NCCL_NVLS_ENABLE=0)"
        )

    kind = _KERNEL_KIND[(protocol, algo)]
    nblocks = min(_MAX_BLOCKS, -(-region_lines // _THREADS))
    args = (
        ws_obj.peers_ptr,
        ws_obj.mc_base,
        ws_obj.my_base,
        x.data_ptr(),
        out.data_ptr(),
        ws_obj.state_ptr,
        ws_obj.rank,
        lines,
        chunk,
        ws_obj.capacity,
        nblocks,
    )
    # re-fetched per call so launches land on the capture stream under graph capture
    stream = cudadrv.CUstream(torch.cuda.current_stream().cuda_stream)
    ckey = (kind, world, dt, use_multimem)
    fn = _compiled.get(ckey)
    if fn is None:
        fn = _compiled[ckey] = cute.compile(
            _launch, *args, kind, world, dt, use_multimem, stream
        )
    fn(*args, stream)
    return out


def one_shot_all_reduce(
    x: torch.Tensor,
    group: str | ProcessGroup,
    *,
    protocol: Literal["ll", "lamport"] = "ll",
    use_multimem: bool = False,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Sum-all-reduce ``x`` across ``group`` with a single barrier-free hop.

    Every rank pushes ``x`` into all peers' symmetric inboxes and reduces locally.
    ``x`` and ``out`` are regular tensors (``out=x`` is allowed). Latency-optimal for
    small messages. ``protocol`` selects the arrival signal (``"ll"``: epoch flags,
    ``"lamport"``: negative-zero sentinel); ``use_multimem`` broadcasts the push with
    one ``multimem.st``. Collective: all ranks must call it with the same arguments,
    in the same order. See Note [Barrier-free all-reduce protocols].
    """
    return _all_reduce(x, group, "one_shot", protocol, use_multimem, out)


def two_shot_all_reduce(
    x: torch.Tensor,
    group: str | ProcessGroup,
    *,
    protocol: Literal["ll", "lamport"] = "ll",
    use_multimem: bool = False,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Sum-all-reduce ``x`` across ``group`` as a barrier-free reduce-scatter then
    all-gather. Moves ``1 / world_size`` of the one-shot traffic per hop at the cost of
    a second hop. Same arguments and contract as :func:`one_shot_all_reduce`;
    ``use_multimem`` applies to the all-gather push.
    """
    return _all_reduce(x, group, "two_shot", protocol, use_multimem, out)
