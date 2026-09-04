#pragma once

#include <ATen/ATen.h>
#include <c10/cuda/CUDAAllocatorConfig.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryTypes.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>

#include <mutex>
#include <optional>
#include <shared_mutex>

namespace c10d::symmetric_memory {

// NOTE [per-process-group signal pad]
// One allocation can be rendezvous'd into several process groups, and each
// gets its own `CUDAPeerAllocInfo`. They must not share signal pad bytes: the
// barrier slot address is `world_size * channel + rank`, computed with the
// *group-local* world size and rank, so two groups over one pad compute
// overlapping addresses out of unrelated numbering and silently corrupt each
// other.
//
// So the pad region reserved at the front of every allocation holds
// `kSignalPadGroupSlots` slices of `get_signal_pad_size()` bytes, and each
// group takes one slice. `get_signal_pad_size()` keeps meaning the size of a
// single group's pad, so the channel count per group is unchanged.
//
// The slice index has to be agreed across the group -- peers address my pad as
// `signal_pads[me] + slot * slice`, so a rank that picked a different slot
// would write into the wrong slice. Groups can also be registered in different
// orders on different ranks, so it cannot be a local counter. It is chosen
// during rendezvous by all-gathering each member's locally-taken slots and
// taking the lowest one free everywhere.
constexpr size_t kSignalPadGroupSlots = 4;

// Resource wrapper that owns a (vaddr, allocation handle) pair. Upon
// destruction, it unmaps the vaddr and releases the allocation handle.
struct AllocationRef : public c10::intrusive_ptr_target {
  void* ptr;
  HandleType handle;
  size_t block_size;
  int device_idx;
  bool is_multicast;

  AllocationRef(
      void* ptr,
      HandleType handle,
      size_t block_size,
      int device_idx,
      bool is_multicast = false);

  ~AllocationRef();
};

// Forward declaration of CUDAPeerAllocInfo
class CUDAPeerAllocInfo;

class CUDASymmetricMemory : public SymmetricMemory {
 public:
  // This is mostly a shallow copy that shares the pointer to
  // `CUDAPeerAllocInfo` which corresponds to the base Block. The
  // CUDASymmetricMemory handle is specified by the offset to the base ptr.
  CUDASymmetricMemory(
      const c10::intrusive_ptr<CUDAPeerAllocInfo>& pai,
      size_t offset);

  ~CUDASymmetricMemory() override {};

  std::vector<void*> get_buffer_ptrs() override;
  std::vector<void*> get_signal_pad_ptrs() override;
  void** get_buffer_ptrs_dev() override;
  void** get_signal_pad_ptrs_dev() override;
  size_t get_buffer_size() override;
  size_t get_offset() override;

  bool has_multicast_support() override;
  void* get_multicast_ptr() override;

  void barrier(int channel, size_t timeout_ms) override;
  void put_signal(int dst_rank, int channel, size_t timeout_ms) override;
  void wait_signal(int src_rank, int channel, size_t timeout_ms) override;

  // See NOTE [signal pad stream serialization]. Hold the returned lock across
  // the kernel launch.
  [[nodiscard]] std::unique_lock<std::mutex> guard_stream();

  int get_rank() override;
  int get_world_size() override;
  c10::Device get_device() override;
  bool world_within_direct_access() override;

 private:
  int local_device_idx_;
  int rank_;
  int world_size_;
  c10::intrusive_ptr<CUDAPeerAllocInfo> pai_;
  size_t offset_{0}; // in byte
};

// A class to hold the base pointers and signal pad pointers for a group of
// peers. One `CUDAPeerAllocInfo` object can be shared by multiple
// `CUDASymmetricMemory` objects when latter reside on the same allocation
// and rendezvous over the same group. (The `CUDASymmetricMemory` objects may
// have different offsets compared to the base address.)
class CUDAPeerAllocInfo : public c10::intrusive_ptr_target {
 public:
  CUDAPeerAllocInfo(
      std::vector<c10::intrusive_ptr<AllocationRef>> alloc_refs,
      std::vector<void*> buffers,
      std::vector<void*> signal_pads,
      void* mc_signal_pad_addr,
      HandleType mc_handle,
      void* mc_addr,
      size_t buffer_size,
      int local_device_idx,
      int rank,
      int world_size,
      std::string group_name,
      size_t pad_slot);

  // See NOTE [signal pad stream serialization]. Lives here, not on
  // `CUDASymmetricMemory`, because one `CUDAPeerAllocInfo` owns exactly one
  // signal pad slice: every handle sharing this object shares that slice and
  // must share one serialization point, while other allocations -- and, per
  // NOTE [per-process-group signal pad], other groups over this allocation --
  // own different slices and must stay independent of each other.
  [[nodiscard]] std::unique_lock<std::mutex> guard_stream();

  // Which slice of the allocation's pad region this group holds.
  size_t pad_slot() const {
    return pad_slot_;
  }

 private:
  std::mutex stream_mu_;
  std::optional<c10::cuda::CUDAStream> last_stream_;
  size_t pad_slot_;

  std::vector<c10::intrusive_ptr<AllocationRef>> alloc_refs_;
  std::vector<void*> buffers_;
  std::vector<void*> signal_pads_;
  void* mc_signal_pad_addr_;
  HandleType mc_handle_;
  void* mc_addr_;
  size_t buffer_size_;
  int local_device_idx_;
  int rank_;
  int world_size_;
  void** buffers_dev_;
  void** signal_pads_dev_;
  std::string group_name_;

  friend class CUDASymmetricMemory;
};

// Serializes signal-pad access for the collective entry points, which hold a
// base-class pointer. Returns an empty lock (a no-op) for backends other than
// the CUDA one, whose primitives do not use this pad protocol.
[[nodiscard]] std::unique_lock<std::mutex> guard_pad_stream(
    const c10::intrusive_ptr<SymmetricMemory>& mem);

// Metadata associated with each allocation performed by
// `CUDASymmetricMemoryAllocator`.
struct Block : public c10::intrusive_ptr_target {
  c10::intrusive_ptr<AllocationRef> alloc_ref;
  int device_idx;
  size_t block_size;
  size_t buffer_size;
  // Byte offset from the allocation base (alloc_ref->ptr) to the start of the
  // user buffer; the signal pad occupies [0, buffer_offset).
  size_t buffer_offset;
  std::optional<std::string> default_group_name;
  std::map<std::string, c10::intrusive_ptr<CUDAPeerAllocInfo>> symm_mems;

  Block(
      c10::intrusive_ptr<AllocationRef> alloc_ref,
      int device_idx,
      size_t block_size,
      size_t buffer_size,
      size_t buffer_offset,
      const std::optional<std::string>& group_name);
};

class CUDASymmetricMemoryAllocator : public SymmetricMemoryAllocator {
 public:
  void* alloc(
      size_t size,
      int device_idx,
      const std::optional<std::string>& group_name) override;

  void free(void* ptr) override;
  size_t get_alloc_size(void* ptr) override;
  c10::intrusive_ptr<SymmetricMemory> rendezvous(
      void* ptr,
      const std::optional<std::string>& group_name) override;
  bool has_multicast_support(int device_idx) override;
  bool has_allocation(void* ptr) override;
  c10::DeviceType supported_device_type() override;
  std::string name() override;

 private:
  c10::intrusive_ptr<Block> find_block(void* ptr);
  c10::intrusive_ptr<Block> find_block_covering(void* ptr, size_t& offset);

  std::shared_mutex mutex_;
  std::unordered_map<void*, c10::intrusive_ptr<Block>> ptr_to_block_;
  c10::cuda::CUDACachingAllocator::Expandable_Segments_Handle_Type
      handle_type_ = c10::cuda::CUDACachingAllocator::
          Expandable_Segments_Handle_Type::UNSPECIFIED;
};

} // namespace c10d::symmetric_memory
