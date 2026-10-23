"""Symmetric buffers from a Python-allocator ``MemPool`` backed by nccl4py.

Segments come from ``ncclMemAlloc`` through :meth:`torch.cuda.MemPool.from_py_allocator`.
:class:`NcclWindow` registers a tensor from that pool as an NCCL ``COLL_SYMMETRIC``
window on a process group's own communicator, and exposes the peer and multicast
pointers device kernels need. Because the window lives on the group's communicator,
NCCL collectives issued through the process group on that tensor (for example
``dist.all_reduce``) also run NCCL's symmetric kernels. This path does not go through
the C++ symmetric-memory allocator or ``rendezvous``.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import nccl.core as nccl
from nccl.bindings import nccl as nccl_bindings

import torch
import torch.distributed as dist


if TYPE_CHECKING:
    from torch.distributed.distributed_c10d import ProcessGroup


__all__ = ["NcclWindow", "empty", "get_mem_pool", "release_pool"]


class _NcclMemAllocator:
    """``ncclMemAlloc`` / ``ncclMemFree`` callbacks for ``MemPool.from_py_allocator``.

    The callbacks can run concurrently from several threads, so the set of live
    segment bases is guarded by a lock."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._segments: set[int] = set()

    def alloc(self, size: int, device: int, stream: int) -> int:
        # ncclMemAlloc allocates on the current device, which PyTorch does not set
        with torch.cuda.device(device):
            ptr = int(nccl_bindings.mem_alloc(size))
        with self._lock:
            self._segments.add(ptr)
        return ptr

    def free(self, ptr: int, size: int, device: int, stream: int) -> None:
        with self._lock:
            self._segments.discard(ptr)
        with torch.cuda.device(device):
            nccl_bindings.mem_free(ptr)

    def is_segment_base(self, ptr: int) -> bool:
        with self._lock:
            return ptr in self._segments


_allocator = _NcclMemAllocator()
_pool: torch.cuda.MemPool | None = None
_pool_devices: set[int] = set()


def get_mem_pool() -> torch.cuda.MemPool:
    """The nccl4py-backed pool, created on first use. One pool serves every device:
    ``use_mem_pool`` picks the device and the allocator callback receives it.

    ``no_split=True`` gives every tensor its own segment, so a tensor's
    ``data_ptr`` is the ``ncclMemAlloc`` base that window registration requires.
    ``use_on_oom`` stays off so the pool never lends its memory to other allocations.
    """
    global _pool
    if _pool is None:
        _pool = torch.cuda.MemPool.from_py_allocator(
            _allocator.alloc, _allocator.free, no_split=True
        )
    return _pool


def empty(
    *size: int, dtype: torch.dtype | None = None, device: torch.device
) -> torch.Tensor:
    """Like :func:`torch.empty`, but allocated from :func:`get_mem_pool`. Pass the
    result to :class:`NcclWindow` to make it symmetric."""
    device = torch.device(device)
    with torch.cuda.use_mem_pool(get_mem_pool(), device=device):
        t = torch.empty(size, dtype=dtype, device=device)
    _pool_devices.add(t.device.index)
    return t


def release_pool() -> None:
    """Drop the pool and return its segments with ``ncclMemFree``. Close all windows
    and free all tensors from the pool first."""
    global _pool
    _pool = None
    for idx in _pool_devices:
        with torch.cuda.device(idx):
            torch.cuda.empty_cache()
    _pool_devices.clear()


class NcclWindow:
    """A collective ``COLL_SYMMETRIC`` NCCL window over a tensor from :func:`empty`.

    The window is registered on ``group``'s own NCCL communicator, so the process
    group must already have created it (pass ``device_id`` to ``init_process_group``
    or run one collective first). Creating and closing windows is collective: every
    rank must do it in the same order. Call :meth:`close` before the tensor is freed
    and before the process group is destroyed.
    """

    def __init__(self, tensor: torch.Tensor, group: ProcessGroup) -> None:
        if not _allocator.is_segment_base(tensor.data_ptr()):
            raise ValueError(
                "NcclWindow needs a tensor allocated by _nccl4py_pool.empty"
            )
        comm_ptr = int(dist.get_backend_impl(group, tensor.device).comm_ptr)
        if comm_ptr == 0:
            raise RuntimeError(
                "the process group has no NCCL communicator yet; pass device_id to "
                "init_process_group or run a collective first"
            )
        # non-owning wrapper: torch keeps ownership of the communicator
        self._comm = nccl.Communicator(comm_ptr)
        self._tensor = tensor
        self._win = self._comm.register_window(
            tensor, flags=nccl.WindowFlag.COLL_SYMMETRIC
        )
        world_size = group.size()
        ptrs = [self._win.get_peer_device_pointer(r) for r in range(world_size)]
        if not all(ptrs):
            raise RuntimeError(
                "a peer is not load/store accessible from this GPU; NcclWindow "
                "needs every rank in one NVLink domain"
            )
        self.buffer_ptrs: list[int] = [int(p) for p in ptrs]
        self.buffer_ptrs_dev = torch.tensor(
            self.buffer_ptrs, dtype=torch.int64, device=tensor.device
        )
        self.multicast_ptr = int(self._win.get_lsa_multimem_device_pointer() or 0)

    def close(self) -> None:
        """Deregister the window. Idempotent."""
        if self._win is not None:
            self._win.close()
            self._win = None
            self._tensor = None
