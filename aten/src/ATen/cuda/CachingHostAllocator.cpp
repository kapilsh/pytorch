#include <ATen/cuda/CachingHostAllocator.h>

#include <ATen/cuda/CUDAEvent.h>
#include <c10/core/thread_pool.h>
#include <c10/cuda/CUDAAllocatorConfig.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <c10/util/Gauge.h>

#include <cuda_runtime_api.h>
#include <array>
#include <future>

#ifdef __linux__
#include <c10/util/error.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <unistd.h>

// Older kernel headers (e.g. manylinux) predate io_uring; the numbers are the
// same on every architecture.
#ifndef __NR_io_uring_setup
#define __NR_io_uring_setup 425
#endif
#ifndef __NR_io_uring_register
#define __NR_io_uring_register 427
#endif
#endif

namespace at::cuda {
namespace {

using Block = HostBlock<CUDAStream>;

#ifdef __linux__
// pinned_unmovable: pages that are only cudaHostRegister'd (or mlocked) can
// still be migrated by the kernel, e.g. by compaction when a NUMA node runs
// low on free memory, and a GPU copy that touches a page under migration
// stalls for up to hundreds of milliseconds. A long-term pin
// (pin_user_pages(FOLL_LONGTERM), the pin taken by an RDMA memory
// registration) makes compaction, reclaim and NUMA balancing skip the pages.
// io_uring buffer registration takes that pin without needing an RDMA device;
// the ring is never used for I/O. Like an RDMA registration, it is charged to
// RLIMIT_MEMLOCK unless the process has CAP_IPC_LOCK.
//
// The memory must be a shared mapping: on fork, the kernel copies long-term
// pinned pages of a private mapping into the child instead of sharing them.

// io_uring_register limits each registered buffer to 1 GiB.
constexpr size_t kMaxIoUringBufferSize = size_t(1) << 30;
constexpr unsigned kIoUringRegisterBuffers = 0;
constexpr unsigned kIoUringUnregisterBuffers = 1;

// Long-term pins [ptr, ptr + size). Returns an io_uring fd that holds the pin.
int longTermPin(void* ptr, size_t size) {
  // struct io_uring_params is 120 bytes; all zeros asks for the defaults.
  alignas(8) std::array<char, 120> params{};
  int fd = static_cast<int>(syscall(__NR_io_uring_setup, 1, params.data()));
  TORCH_CHECK(
      fd >= 0,
      "pinned_unmovable: io_uring_setup failed: ",
      c10::utils::str_error(errno),
      ". io_uring may be disabled (sysctl kernel.io_uring_disabled) or "
      "blocked by a seccomp profile.");
  std::vector<iovec> iovs;
  for (size_t off = 0; off < size; off += kMaxIoUringBufferSize) {
    iovs.push_back(
        {static_cast<char*>(ptr) + off,
         std::min(kMaxIoUringBufferSize, size - off)});
  }
  if (syscall(
          __NR_io_uring_register,
          fd,
          kIoUringRegisterBuffers,
          iovs.data(),
          iovs.size()) != 0) {
    int err = errno;
    close(fd);
    TORCH_CHECK(
        false,
        "pinned_unmovable: pinning ",
        size,
        " bytes failed: ",
        c10::utils::str_error(err),
        err == ENOMEM ? " (check RLIMIT_MEMLOCK)" : "");
  }
  return fd;
}

void longTermUnpin(int fd) {
  // Unregister before closing: a forked child holding a copy of the fd would
  // otherwise keep the pages pinned until it exits.
  syscall(__NR_io_uring_register, fd, kIoUringUnregisterBuffers, nullptr, 0);
  close(fd);
}
#endif

struct CUDACachingHostAllocatorImpl
    : public CachingHostAllocatorImpl<CUDAStream, CUDAEventPool::Event> {
  void free(void* ctx) override {
    using Base = CachingHostAllocatorImpl<CUDAStream, CUDAEventPool::Event>;
    try {
      Base::free(ctx);
    } catch (...) {
      if (!c10::cuda::CUDACachingAllocator::CUDAAllocatorConfig::
              pinned_free_catch_all()) {
        TORCH_WARN("Exception in pinned allocator free(), rethrowing");
        throw;
      }
      // pinned_free_catch_all is enabled: suppress the exception to prevent
      // it from escaping through ~StorageImpl() (implicitly noexcept), which
      // would cause std::terminate. Allows graceful shutdown to proceed.
      STATIC_GAUGE(pytorch.CUDACachingHostAllocator.free_fail_catch_all)
          .record(1);
      TORCH_WARN("Suppressed exception in pinned allocator free()");
    }
  }

 private:
  ska::flat_hash_map<void*, bool> use_host_register;
  // pinned_unmovable allocations: ptr -> (size, fd holding the long-term pin).
  ska::flat_hash_map<void*, std::pair<size_t, int>> unmovable_pins;

  void allocate_host_memory(size_t size, void** ptr) override {
    // try allocating from reserve segment first before calling into expensive APIs
    if (get_reserve_segment().initialized()) {
      *ptr = get_reserve_segment().allocate(size);
      if (*ptr != nullptr) {
        return;
      }
    }
    allocate_host_memory_slowpath(size, ptr);
  }

  void allocate_host_memory_slowpath(size_t size, void** ptr) {
    // Pinned memory pointers allocated by any device can be directly used by
    // any other device, regardless of the current device at the time of
    // allocation, since we assume unified addressing. So we grab any existing
    // primary context, if available. See pytorch/pytorch#21081.
    // This can be a large performance hit if we cross NUMA nodes by allocating
    // and pinning memory on one side of the NUMA node and then using it on the
    // other side. Thankfully, we use one process per GPU, so we don't run into
    // this issue.
    at::OptionalDeviceGuard device_guard;
    auto primary_ctx_device_index =
        c10::cuda::getDeviceIndexWithPrimaryContext();
    if (primary_ctx_device_index.has_value()) {
      device_guard.reset_device(
          at::Device(at::DeviceType::CUDA, *primary_ctx_device_index));
    }

    auto start = std::chrono::steady_clock::now();
    bool unmovable = c10::cuda::CUDACachingAllocator::CUDAAllocatorConfig::pinned_unmovable();
    bool use_register = unmovable || c10::cuda::CUDACachingAllocator::CUDAAllocatorConfig::pinned_use_cuda_host_register();
    if (use_register) {
      allocWithCudaHostRegister(ptr, size, unmovable);
    } else {
      // Use cudaHostAlloc for allocating pinned memory (global lock in driver)
      if (current_stream_is_capturing_fast_path()) {
          at::cuda::CUDAStreamCaptureModeGuard g{cudaStreamCaptureModeRelaxed};
          C10_CUDA_CHECK(cudaHostAlloc(ptr, size, cudaHostAllocDefault));
      } else {
          C10_CUDA_CHECK(cudaHostAlloc(ptr, size, cudaHostAllocDefault));
      }
    }

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Update the statistics on the time spent on cudaHostAlloc/hostRegister
    {
      std::lock_guard<std::mutex> g(stats_.timing_mutex_);
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(use_host_register.count(*ptr) == 0);
      use_host_register[*ptr] = use_register;
      stats_.host_alloc_time.increase(duration.count());
    }
  }

  void free_block(Block* block) override {
    // We never free blocks from the reserve segment
    if (get_reserve_segment().initialized()) {
      // Check if the block is from the reserve segment
      if (get_reserve_segment().owns(block->ptr_)) {
        return;
      }
    }

    free_block_slowpath(block);
  }

  void free_block_slowpath(Block* block) {
    auto start = std::chrono::steady_clock::now();
    // Users may change the allocator config at will. torch unit tests do this.
    // However, allocations using cudaHostRegister should use corresponding
    // cudaHostUnregister and similarly for cudaHostAlloc / cudaFreeHost.
    void* ptr = block->ptr_;
    bool use_register = false;
    std::optional<std::pair<size_t, int>> unmovable_pin;
    {
      std::lock_guard<std::mutex> g(stats_.timing_mutex_);
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(use_host_register.count(ptr) == 1);
      use_register = use_host_register[ptr];
      auto it = unmovable_pins.find(ptr);
      if (it != unmovable_pins.end()) {
        unmovable_pin = it->second;
        unmovable_pins.erase(it);
      }
    }
    if (unmovable_pin) {
#ifdef __linux__
      AT_CUDA_CHECK(cudaHostUnregister(ptr));
      longTermUnpin(unmovable_pin->second);
      munmap(ptr, unmovable_pin->first);
#endif
    } else if (use_register) {
      AT_CUDA_CHECK(cudaHostUnregister(ptr));
      // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
      std::free(ptr);
    } else {
      AT_CUDA_CHECK(cudaFreeHost(ptr));
    }
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Update the statistics on the time spent on cudaFreeHost/hostUnregister
    {
      std::lock_guard<std::mutex> g(stats_.timing_mutex_);
      use_host_register.erase(ptr);
      stats_.host_free_time.increase(duration.count());
    }
  }

  void record_stream(
      std::optional<std::vector<CUDAEventPool::Event>>& events,
      CUDAStream stream) override {
    auto event = create_event_internal(stream.device_index());
    event->record(stream);
    events->push_back(std::move(event));
  }

  bool query_event(CUDAEventPool::Event& event) override {
    return event->query();
  }

  CUDAEventPool::Event create_event_internal(DeviceIndex idx) {
    // Leak the event pool to avoid shutdown issue.
    static auto* event_pool = new CUDAEventPool();
    return event_pool->get(idx);
  }

  PinnedReserveSegment& get_reserve_segment() {
    static auto reserve_segment = [&]() {
      if (c10::cuda::CUDACachingAllocator::CUDAAllocatorConfig::pinned_reserve_segment_size_mb() > 0) {
        void *ptr;
        size_t sz = c10::cuda::CUDACachingAllocator::CUDAAllocatorConfig::pinned_reserve_segment_size_mb() * 1024 * 1024;
        allocate_host_memory_slowpath(sz, &ptr);
        return PinnedReserveSegment(ptr, sz);
      } else {
        return PinnedReserveSegment();
      }
    } ();
    return reserve_segment;
  }

  TaskThreadPool* getThreadPool() {
    static TaskThreadPool* pool = new TaskThreadPool(
        static_cast<int>(c10::cuda::CUDACachingAllocator::CUDAAllocatorConfig::
            pinned_max_register_threads()));
    return pool;
  }

  void mapPagesForRegister(
      const void* ptr,
      size_t size,
      size_t i,
      size_t numThreads,
      size_t pageSize) {
    uintptr_t start = (uintptr_t)ptr + (size * i / numThreads);
    uintptr_t end = start + (size / numThreads);
    if (i == (numThreads - 1)) {
      end = (uintptr_t)ptr + size;
    }

    // pre-fault/map the pages by setting the first byte of the page
    uintptr_t alignedStart =
        ((start + pageSize - 1) & ~(pageSize - 1));
    for (uintptr_t p = alignedStart; p < end; p += pageSize) {
      // NOLINTNEXTLINE(performance-no-int-to-ptr)
      memset((void*)p, 0, 1);
    }
  }

  void allocWithCudaHostRegister(void** ptr, size_t roundSize, bool unmovable) {
    // Here we do regular allocation, pre-fault/map the pages, and then do
    // cudaHostRegister with GPU mapping flags to lock the pages, so we
    // can minimize the cost for the cuda global lock.
    if (unmovable) {
#ifdef __linux__
      // A shared mapping, so that forks do not copy the long-term pinned pages.
      *ptr = mmap(
          nullptr,
          roundSize,
          PROT_READ | PROT_WRITE,
          MAP_SHARED | MAP_ANONYMOUS,
          -1,
          0);
      TORCH_CHECK(
          *ptr != MAP_FAILED,
          "pinned_unmovable: mmap of ",
          roundSize,
          " bytes failed: ",
          c10::utils::str_error(errno));
#else
      TORCH_CHECK(false, "pinned_unmovable is only supported on Linux");
#endif
    } else {
      // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
      *ptr = std::malloc(roundSize);
    }

    // Parallelize the mapping/registering of pages to reduce wall time
    size_t pageSize = (1 << 12); // 4kB pages
    size_t numMapThreads = c10::cuda::CUDACachingAllocator::
        CUDAAllocatorConfig::pinned_num_register_threads();
    if ((numMapThreads > 1) && (roundSize >= (pageSize * numMapThreads))) {
      // parallelize the mapping of pages with a threadpool
      auto* pool = getThreadPool();
      std::vector<std::promise<void>> promises;
      std::vector<std::future<void>> futures;
      promises.reserve(numMapThreads);
      futures.reserve(numMapThreads);

      for (size_t i = 0; i < numMapThreads; i++) {
        promises.emplace_back();
        futures.push_back(promises[i].get_future());
        auto task = [this,
                     i,
                     ptr,
                     roundSize,
                     numMapThreads,
                     pageSize,
                     &promises]() mutable {
          mapPagesForRegister(
              *ptr,
              roundSize,
              i, // thread task-id
              numMapThreads,
              pageSize);
          // set the promise when mapping pages are done
          promises[i].set_value();
        };
        pool->run(task);
      }
      for (auto& future : futures) {
        future.wait();
      }
    } else {
      // Map pages in the same thread
      mapPagesForRegister(*ptr, roundSize, 0, 1, pageSize);
    }

#ifdef __linux__
    if (unmovable) {
      // Pin before cudaHostRegister: a long-term pin first migrates pages out
      // of ZONE_MOVABLE / CMA, which must happen before the driver maps them.
      int fd = -1;
      try {
        fd = longTermPin(*ptr, roundSize);
      } catch (...) {
        munmap(*ptr, roundSize);
        throw;
      }
      std::lock_guard<std::mutex> g(stats_.timing_mutex_);
      unmovable_pins[*ptr] = {roundSize, fd};
    }
#endif

    // Register the mapped pages using cudaHostRegister
    if (current_stream_is_capturing_fast_path()) {
      at::cuda::CUDAStreamCaptureModeGuard g{cudaStreamCaptureModeRelaxed};
      AT_CUDA_CHECK(cudaHostRegister(*ptr, roundSize, cudaHostRegisterDefault));
    } else {
      AT_CUDA_CHECK(cudaHostRegister(*ptr, roundSize, cudaHostRegisterDefault));
    }
  }

  CUDAStream get_current_stream() const override {
    // get_current_stream() is called in contexts (such as allocation)
    // where a device may not already be set. Set it before
    // continuing.
    at::OptionalDeviceGuard device_guard;
    auto primary_ctx_device_index =
        c10::cuda::getDeviceIndexWithPrimaryContext();
    if (primary_ctx_device_index.has_value()) {
      device_guard.reset_device(
          at::Device(at::DeviceType::CUDA, *primary_ctx_device_index));
    }
    return at::cuda::getCurrentCUDAStream();
  }

  bool stream_is_capturing(CUDAStream s) const override {
    return s.is_capturing();
  }
};

DECLARE_HOST_ALLOCATOR(
    CUDACachingHostAllocator,
    CUDACachingHostAllocatorImpl,
    raw_local_deleter,
    caching_host_allocator)

REGISTER_HOST_ALLOCATOR(at::kCUDA, &caching_host_allocator)

} // anonymous namespace
} // namespace at::cuda
