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

_LL_ONE_SHOT, _LL_TWO_SHOT, _LAMPORT_ONE_SHOT, _LAMPORT_TWO_SHOT = range(4)
_KERNEL_KIND = {
    ("ll", "one_shot"): _LL_ONE_SHOT,
    ("ll", "two_shot"): _LL_TWO_SHOT,
    ("lamport", "one_shot"): _LAMPORT_ONE_SHOT,
    ("lamport", "two_shot"): _LAMPORT_TWO_SHOT,
}

# state tensor layout (int32): [epoch_or_step, arrivals, used[0], used[1], used[2]]
_STATE_NUMEL = 8


# ---- PTX primitives -------------------------------------------------------------


def _ptr(addr):
    return cute.make_ptr(U32, addr, _GMEM)


def _line_ptr(base, line):
    return _ptr(base + I64(line) * _LINE_BYTES)


def _st_v4(ptr, w0, w1, w2, w3):
    cute.arch.inline_ptx(
        "st.relaxed.sys.global.v4.u32 [{$r0}], {{$r1}, {$r2}, {$r3}, {$r4}};",
        read_only_args=[ptr, w0, w1, w2, w3],
    )


def _st_v4_mc(ptr, w0, w1, w2, w3):
    cute.arch.inline_ptx(
        "multimem.st.relaxed.sys.global.v4.f32 [{$r0}], {{$r1}, {$r2}, {$r3}, {$r4}};",
        read_only_args=[ptr, w0, w1, w2, w3],
    )


def _ld_v4(ptr):
    return cute.arch.inline_ptx(
        "ld.relaxed.sys.global.v4.u32 {{$w0}, {$w1}, {$w2}, {$w3}}, [{$r0}];",
        write_only_types=[U32, U32, U32, U32],
        read_only_args=[ptr],
    )


def _ld_nc_v2(ptr):
    return cute.arch.inline_ptx(
        "ld.global.nc.v2.u32 {{$w0}, {$w1}}, [{$r0}];",
        write_only_types=[U32, U32],
        read_only_args=[ptr],
    )


def _ld_nc_v4(ptr):
    return cute.arch.inline_ptx(
        "ld.global.nc.v4.u32 {{$w0}, {$w1}, {$w2}, {$w3}}, [{$r0}];",
        write_only_types=[U32, U32, U32, U32],
        read_only_args=[ptr],
    )


def _st_out_v2(ptr, w0, w1):
    cute.arch.inline_ptx(
        "st.global.v2.u32 [{$r0}], {{$r1}, {$r2}};", read_only_args=[ptr, w0, w1]
    )


def _st_out_v4(ptr, w0, w1, w2, w3):
    cute.arch.inline_ptx(
        "st.global.v4.u32 [{$r0}], {{$r1}, {$r2}, {$r3}, {$r4}};",
        read_only_args=[ptr, w0, w1, w2, w3],
    )


def _push(peers, mc_base, line, words, use_mc):
    """Store one 16 B line at `line` of every peer's inbox."""
    if use_mc:
        _st_v4_mc(_line_ptr(mc_base, line), *words)
    else:
        for peer in peers:
            _st_v4(_line_ptr(peer, line), *words)


# ---- element codec: u32 words of 1x fp32 / 2x bf16 / 2x fp16 <-> fp32 -----------


def _unpack(w, dt):
    if dt == _FP32:
        return [
            cute.arch.inline_ptx(
                "mov.b32 {$w0}, {$r0};", write_only_types=[F32], read_only_args=[w]
            )
        ]
    if dt == _BF16:  # bf16 -> f32 is exact: the bf16 bits are the top half of the f32
        ptx = (
            "{ .reg .b32 a, b; shl.b32 a, {$r0}, 16; and.b32 b, {$r0}, 0xFFFF0000;"
            " mov.b32 {$w0}, a; mov.b32 {$w1}, b; }"
        )
    else:
        ptx = (
            "{ .reg .b16 lo, hi; mov.b32 {lo, hi}, {$r0};"
            " cvt.f32.f16 {$w0}, lo; cvt.f32.f16 {$w1}, hi; }"
        )
    return list(
        cute.arch.inline_ptx(ptx, write_only_types=[F32, F32], read_only_args=[w])
    )


def _pack(fs, dt):
    if dt == _FP32:
        return [
            cute.arch.inline_ptx(
                "mov.b32 {$w0}, {$r0};", write_only_types=[U32], read_only_args=[f]
            )
            for f in fs
        ]
    cvt = "cvt.rn.bf16x2.f32" if dt == _BF16 else "cvt.rn.f16x2.f32"
    # the second source operand lands in the low half (element 0)
    return [
        cute.arch.inline_ptx(
            f"{cvt} {{$w0}}, {{$r1}}, {{$r0}};",
            write_only_types=[U32],
            read_only_args=[fs[k], fs[k + 1]],
        )
        for k in range(0, len(fs), 2)
    ]


def _reduce(words, nsrc, dt):
    """Sum `nsrc` source-major groups of u32 words elementwise in fp32."""
    per = len(words) // nsrc
    acc = [f for w in words[:per] for f in _unpack(w, dt)]
    for s in range(1, nsrc):
        vals = [f for w in words[s * per : (s + 1) * per] for f in _unpack(w, dt)]
        acc = [a + v for a, v in zip(acc, vals)]
    return _pack(acc, dt)


# ---- LL protocol ----------------------------------------------------------------


def _ll_ready(f0, f1, eu):
    return cute.arch.inline_ptx(
        "{ .reg .pred p; setp.eq.u32 p, {$r0}, {$r2}; setp.eq.and.u32 p, {$r1}, {$r2}, p;"
        " selp.u32 {$w0}, 1, 0, p; }",
        write_only_types=[U32],
        read_only_args=[f0, f1, eu],
    )


@cute.jit
def _ll_recv(base: I64, line0: I32, stride: I32, eu: U32, nsrc: cutlass.Constexpr[int]):
    """Payload words of line `line0 + s * stride` for every source s, once published.

    All loads are issued before the first spin so cross-rank latency overlaps."""
    ptrs = [_line_ptr(base, line0 + s * stride) for s in range(nsrc)]
    vals = [_ld_v4(p) for p in ptrs]
    words = []
    for s in cutlass.range_constexpr(nsrc):
        d0, f0, d1, f1 = vals[s]
        rdy = _ll_ready(f0, f1, eu)
        while rdy == U32(0):
            d0, f0, d1, f1 = _ld_v4(ptrs[s])
            rdy = _ll_ready(f0, f1, eu)
        words.append(d0)
        words.append(d1)
    return words


# ---- Lamport (negative-zero sentinel) protocol ----------------------------------


def _sentinel_word(dt) -> int:
    return 0x80000000 if dt == _FP32 else 0x80008000


def _scrub(w, dt):
    """Replace every -0.0 element in a u32 word with +0.0."""
    if dt == _FP32:
        ptx = (
            "{ .reg .b32 t; set.eq.u32.u32 t, {$r0}, 0x80000000; not.b32 t, t;"
            " and.b32 {$w0}, {$r0}, t; }"
        )
    else:
        ptx = (
            "{ .reg .b32 m, t, r; mov.b32 r, {$r0};"
            " and.b32 m, r, 0xFFFF; set.eq.u32.u32 t, m, 0x8000;"
            " and.b32 t, t, 0xFFFF; not.b32 t, t; and.b32 r, r, t;"
            " and.b32 m, r, 0xFFFF0000; set.eq.u32.u32 t, m, 0x80000000;"
            " and.b32 t, t, 0xFFFF0000; not.b32 t, t; and.b32 {$w0}, r, t; }"
        )
    return cute.arch.inline_ptx(ptx, write_only_types=[U32], read_only_args=[w])


def _lamport_ready(w0, w1, w2, w3, dt):
    if dt == _FP32:
        checks = "".join(
            f" set.eq.u32.u32 t, {{$r{k}}}, 0x80000000; or.b32 a, a, t;"
            for k in range(4)
        )
    else:
        checks = "".join(
            f" and.b32 m, {{$r{k}}}, 0xFFFF; set.eq.u32.u32 t, m, 0x8000; or.b32 a, a, t;"
            f" and.b32 m, {{$r{k}}}, 0xFFFF0000; set.eq.u32.u32 t, m, 0x80000000;"
            f" or.b32 a, a, t;"
            for k in range(4)
        )
    return cute.arch.inline_ptx(
        "{ .reg .pred p; .reg .b32 a, m, t; mov.b32 a, 0;"
        + checks
        + " setp.eq.u32 p, a, 0; selp.u32 {$w0}, 1, 0, p; }",
        write_only_types=[U32],
        read_only_args=[w0, w1, w2, w3],
    )


def _st_sentinel(ptr, dt):
    cute.arch.inline_ptx(
        f"{{ .reg .b32 s; mov.b32 s, {_sentinel_word(dt):#x};"
        " st.global.v4.u32 [{$r0}], {s, s, s, s}; }",
        read_only_args=[ptr],
    )


@cute.jit
def _lamport_recv(
    base: I64,
    line0: I32,
    stride: I32,
    nsrc: cutlass.Constexpr[int],
    dt: cutlass.Constexpr[int],
):
    """Words of line `line0 + s * stride` for every source s, once no element is -0.0."""
    ptrs = [_line_ptr(base, line0 + s * stride) for s in range(nsrc)]
    vals = [_ld_v4(p) for p in ptrs]
    words = []
    for s in cutlass.range_constexpr(nsrc):
        w0, w1, w2, w3 = vals[s]
        rdy = _lamport_ready(w0, w1, w2, w3, dt)
        while rdy == U32(0):
            w0, w1, w2, w3 = _ld_v4(ptrs[s])
            rdy = _lamport_ready(w0, w1, w2, w3, dt)
        words += [w0, w1, w2, w3]
    return words


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
    for r in cutlass.range_constexpr(nregions):
        j = tid
        while j < used:
            _st_sentinel(_line_ptr(my_base, line0 + r * stride + j), dt)
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
        d0, d1 = _ld_nc_v2(_ptr(x_base + I64(i) * 8))
        _push(peers, mc_base, slot + rank * stride + i, [d0, eu, d1, eu], use_mc)
        i = i + nthreads
    i = tid
    while i < lines:
        r0, r1 = _reduce(_ll_recv(my_base, slot + i, stride, eu, ws), ws, dt)
        _st_out_v2(_ptr(out_base + I64(i) * 8), r0, r1)
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
        for d in cutlass.range_constexpr(ws):
            g = d * chunk + i
            d0 = U32(0)
            d1 = U32(0)
            if g < lines:
                d0, d1 = _ld_nc_v2(_ptr(x_base + I64(g) * 8))
            _st_v4(_line_ptr(peers[d], rs + rank * stride + i), d0, eu, d1, eu)
        i = i + nthreads
    # reduce own chunk, all-gather push
    i = tid
    while i < chunk:
        r0, r1 = _reduce(_ll_recv(my_base, rs + i, stride, eu, ws), ws, dt)
        _push(peers, mc_base, ag + rank * stride + i, [r0, eu, r1, eu], use_mc)
        i = i + nthreads
    i = tid
    while i < chunk:
        words = _ll_recv(my_base, ag + i, stride, eu, ws)
        for j in cutlass.range_constexpr(ws):
            g = j * chunk + i
            if g < lines:
                _st_out_v2(_ptr(out_base + I64(g) * 8), words[2 * j], words[2 * j + 1])
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
        w = [_scrub(v, dt) for v in _ld_nc_v4(_ptr(x_base + I64(i) * 16))]
        _push(peers, mc_base, slot + rank * stride + i, w, use_mc)
        i = i + nthreads
    # restore the slot used by the previous call while peer data is in flight
    _lamport_clear(my_base, stale * ws * stride, stride, used, tid, nthreads, ws, dt)
    i = tid
    while i < lines:
        r = _reduce(_lamport_recv(my_base, slot + i, stride, ws, dt), ws, dt)
        _st_out_v4(_ptr(out_base + I64(i) * 16), *r)
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
        for d in cutlass.range_constexpr(ws):
            g = d * chunk + i
            w0 = U32(0)
            w1 = U32(0)
            w2 = U32(0)
            w3 = U32(0)
            if g < lines:
                w0, w1, w2, w3 = _ld_nc_v4(_ptr(x_base + I64(g) * 16))
            w = [_scrub(v, dt) for v in (w0, w1, w2, w3)]
            _st_v4(_line_ptr(peers[d], rs + rank * stride + i), *w)
        i = i + nthreads
    _lamport_clear(
        my_base, stale * 2 * ws * stride, stride, used, tid, nthreads, 2 * ws, dt
    )
    i = tid
    while i < chunk:
        r = _reduce(_lamport_recv(my_base, rs + i, stride, ws, dt), ws, dt)
        _push(
            peers, mc_base, ag + rank * stride + i, [_scrub(v, dt) for v in r], use_mc
        )
        i = i + nthreads
    i = tid
    while i < chunk:
        words = _lamport_recv(my_base, ag + i, stride, ws, dt)
        for j in cutlass.range_constexpr(ws):
            g = j * chunk + i
            if g < lines:
                _st_out_v4(_ptr(out_base + I64(g) * 16), *words[4 * j : 4 * j + 4])
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
