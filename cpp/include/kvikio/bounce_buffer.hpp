/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <stack>

#include <kvikio/defaults.hpp>

namespace kvikio {

/**
 * @brief Allocator for page-aligned host memory
 *
 * Uses std::aligned_alloc to allocate host memory aligned to page boundaries (typically 4096
 * bytes). This allocator is suitable for Direct I/O operations that require page-aligned buffers
 * but do not need CUDA context (i.e., host-to-host transfers only).
 */
class PageAlignedAllocator {
 public:
  /**
   * @brief Allocate page-aligned host memory
   *
   * @param size Requested size in bytes (will be rounded up to page boundary)
   * @return Pointer to allocated memory
   */
  void* allocate(std::size_t size);

  /**
   * @brief Deallocate memory previously allocated by this allocator
   *
   * @param buffer Pointer to memory to deallocate
   * @param size Size of the allocation (unused, for interface consistency)
   */
  void deallocate(void* buffer, std::size_t size);
};

/**
 * @brief Allocator for CUDA pinned host memory
 *
 * Uses cudaMemHostAlloc to allocate pinned (page-locked) host memory that can be efficiently
 * transferred to/from GPU device memory. The allocation is only guaranteed to be aligned to "at
 * least 256 bytes". It is NOT guaranteed to be page aligned.
 *
 * @note Do NOT use with Direct I/O - lacks page alignment guarantee
 */
class CudaPinnedAllocator {
 public:
  /**
   * @brief Allocate CUDA pinned host memory
   *
   * @param size Requested size in bytes
   * @return Pointer to allocated pinned memory
   */
  void* allocate(std::size_t size);

  /**
   * @brief Deallocate memory previously allocated by this allocator
   *
   * @param buffer Pointer to memory to deallocate
   * @param size Size of the allocation (unused, for interface consistency)
   */
  void deallocate(void* buffer, std::size_t size);
};

/**
 * @brief Allocator for page-aligned AND CUDA-registered pinned host memory
 *
 * Combines the benefits of both page alignment (for Direct I/O) and CUDA registration
 * (for efficient host-device transfers). Uses std::aligned_alloc followed by
 * cudaMemHostRegister to achieve both properties.
 *
 * @note This is the required allocator for Direct I/O with device memory. Requires a valid CUDA
 * context when allocating.
 */
class CudaPageAlignedPinnedAllocator {
 public:
  /**
   * @brief Allocate page-aligned CUDA-registered pinned host memory
   *
   * @param size Requested size in bytes (will be rounded up to page boundary)
   * @return Pointer to allocated memory
   */
  void* allocate(std::size_t size);

  /**
   * @brief Deallocate memory previously allocated by this allocator
   *
   * @param buffer Pointer to memory to deallocate
   * @param size Size of the allocation (unused, for interface consistency)
   */
  void deallocate(void* buffer, std::size_t size);
};

/**
 * @brief Thread-safe singleton pool for reusable bounce buffers
 *
 * Manages a pool of host memory buffers used for staging data during I/O operations.
 * Buffers are retained and reused across calls to minimize allocation overhead.
 * The pool uses a LIFO (stack) allocation strategy optimized for cache locality.
 *
 * All buffers in the pool have the same size, controlled by `defaults::bounce_buffer_size()`. If
 * the buffer size changes, all cached buffers are cleared and reallocated at the new size.
 *
 * Call `BounceBufferPool::get` to get an allocation that will be retained when it
 * goes out of scope (RAII). The size of all retained allocations are the same.
 *
 * @tparam Allocator The allocator policy that determines buffer properties:
 * - PageAlignedAllocator: For host-only Direct I/O
 * - CudaPinnedAllocator: For device I/O without Direct I/O
 * - CudaPageAlignedPinnedAllocator: For device I/O with Direct I/O
 *
 * @note The destructor intentionally leaks allocations to avoid CUDA cleanup issues when static
 * destructors run after CUDA context destruction
 */
template <typename Allocator = CudaPinnedAllocator>
class BounceBufferPool {
 private:
  mutable std::mutex _mutex{};
  // Stack of free allocations (LIFO for cache locality)
  std::stack<void*> _free_buffers{};
  // The size of each allocation in `_free_buffers`
  std::size_t _buffer_size{defaults::bounce_buffer_size()};
  Allocator _allocator{};

 public:
  /**
   * @brief RAII wrapper for a host bounce buffer allocation
   *
   * Automatically returns the buffer to the pool when destroyed (RAII pattern). Provides access to
   * the underlying memory and its size.
   *
   * @note Non-copyable but movable to allow transfer of ownership while maintaining RAII
   */
  class Buffer {
   private:
    BounceBufferPool* _pool;
    void* _buffer;
    std::size_t _size;

   public:
    Buffer(BounceBufferPool<Allocator>* pool, void* buffer, std::size_t size);
    Buffer(Buffer const&)            = delete;
    Buffer& operator=(Buffer const&) = delete;
    Buffer(Buffer&& o) noexcept;
    Buffer& operator=(Buffer&& o) noexcept;
    ~Buffer() noexcept;
    void* get() const noexcept;
    void* get(std::ptrdiff_t offset) const noexcept;
    std::size_t size() const noexcept;
  };

  BounceBufferPool() = default;

  // Notice, we do not clear the allocations at destruction thus the allocations leaks
  // at exit. We do this because `BounceBufferPool::instance()` stores the allocations in a
  // static stack that are destructed below main, which is not allowed in CUDA:
  // <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#initialization>
  ~BounceBufferPool() noexcept = default;

 private:
  /**
   * @brief Free all retained allocations
   *
   * NB: The `_mutex` must be taken prior to calling this function.
   *
   * @return The number of bytes cleared
   */
  std::size_t _clear();

  /**
   * @brief Ensure the sizes of the retained allocations match `defaults::bounce_buffer_size()`
   *
   * If the configured bounce buffer size has changed, clears all cached buffers so new allocations
   * will use the updated size.
   *
   * NB: `_mutex` must be taken prior to calling this function.
   */
  void _ensure_buffer_size();

 public:
  /**
   * @brief Acquire a bounce buffer from the pool
   *
   * Returns a cached buffer if available, otherwise allocates a new one. The returned Buffer object
   * will automatically return the buffer to the pool when it goes out of scope.
   *
   * @return RAII Buffer object wrapping the allocated memory
   * @exception CudaError if allocation fails (e.g., invalid CUDA context for pinned allocators)
   */
  [[nodiscard]] Buffer get();

  /**
   * @brief Return a buffer to the pool for reuse
   *
   * Typically called automatically by Buffer's destructor. Only adds the buffer to the pool if its
   * size matches the current pool buffer size; otherwise the buffer is deallocated immediately.
   *
   * @param buffer Pointer to memory to return
   * @param size Size of the buffer in bytes
   */
  void put(void* buffer, std::size_t size);

  /**
   * @brief Free all retained allocations in the pool
   *
   * Clears the pool and deallocates all cached buffers. Useful for reclaiming memory when bounce
   * buffers are no longer needed.
   *
   * @return The number of bytes cleared
   */
  std::size_t clear();

  /**
   * @brief Get the number of free buffers currently available in the pool
   *
   * Returns the count of buffers that have been returned to the pool and are ready for reuse.
   *
   * @return The number of buffers available for reuse
   */
  std::size_t num_free_buffers() const;

  /**
   * @brief Get the current buffer size used by the pool
   *
   * Returns the size of buffers currently managed by the pool. This reflects the value of
   * `defaults::bounce_buffer_size()` as of the last pool operation.
   *
   * @return The size in bytes of each buffer in the pool
   */
  std::size_t buffer_size() const;

  /**
   * @brief Get the singleton instance of the pool
   *
   * Each template instantiation (different Allocator) has its own singleton instance.
   *
   * @return Reference to the singleton pool instance
   */
  KVIKIO_EXPORT static BounceBufferPool& instance();

  BounceBufferPool(BounceBufferPool const&)            = delete;
  BounceBufferPool& operator=(BounceBufferPool const&) = delete;
  BounceBufferPool(BounceBufferPool&& o)               = delete;
  BounceBufferPool& operator=(BounceBufferPool&& o)    = delete;
};

/**
 * @brief Bounce buffer pool using page-aligned host memory
 *
 * Use for: Host-only Direct I/O operations (no CUDA context involvement)
 */
using PageAlignedBounceBufferPool = BounceBufferPool<PageAlignedAllocator>;

/**
 * @brief Bounce buffer pool using CUDA pinned memory
 *
 * Use for: Device I/O operations without Direct I/O
 * Note: Not page-aligned - cannot be used with Direct I/O
 */
using CudaPinnedBounceBufferPool = BounceBufferPool<CudaPinnedAllocator>;

/**
 * @brief Bounce buffer pool using page-aligned CUDA-registered pinned memory
 *
 * Use for: Device I/O operations with Direct I/O enabled
 * Provides both page alignment (for Direct I/O) and CUDA registration (for efficient transfers)
 */
using CudaPageAlignedPinnedBounceBufferPool = BounceBufferPool<CudaPageAlignedPinnedAllocator>;

/**
 * @brief K-way bounce buffer ring for overlapping I/O with host-device transfers.
 *
 * Manages a ring of k bounce buffers to enable pipelining between file I/O and CUDA memory
 * transfers. By rotating through multiple buffers, the ring allows async H2D copies to proceed
 * while the next buffer is being filled.
 *
 * Synchronization occurs automatically when the ring wraps around to prevent overwriting buffers
 * with in-flight transfers.
 *
 * @tparam Allocator The allocator policy for bounce buffers:
 *   - CudaPinnedAllocator: For device I/O without Direct I/O
 *   - CudaPageAlignedPinnedAllocator: For device I/O with Direct I/O
 *
 * @note This class is NOT thread-safe. Use one ring per thread or per operation.
 * @note In batch mode, H2D copies are deferred until wrap-around or synchronize(), trading overlap
 * for reduced API call overhead.
 */
template <typename Allocator = CudaPinnedAllocator>
class BounceBufferRing {
 public:
  struct BatchTransferContext {
    std::vector<void*> srcs;
    std::vector<void*> dsts;
    std::vector<std::size_t> sizes;

    void add_entry(void* dst, void* src, std::size_t size);
    void clear();
  };

 private:
  std::vector<typename BounceBufferPool<Allocator>::Buffer> _buffers;
  BatchTransferContext _batch_transfer_ctx;
  std::size_t _num_buffers;
  std::size_t _cur_buf_idx{0};
  std::size_t _initial_buf_idx{0};
  std::size_t _cur_buffer_offset{0};
  bool _batch_copy;

 public:
  /**
   * @brief Construct a bounce buffer ring.
   *
   * @param num_buffers Number of bounce buffers (k) for k-way overlap. Must be >= 1. Higher values
   * allow more overlap but consume more memory.
   * @param batch_copy If true, defer H2D copies and issue them in batches. Useful for many small
   * transfers where API overhead dominates. If false (default), issue H2D copies immediately for
   * better overlap.
   */
  explicit BounceBufferRing(std::size_t num_buffers = 1, bool batch_copy = false);

  ~BounceBufferRing() noexcept = default;

  // Non-copyable, non-movable
  BounceBufferRing(BounceBufferRing const&)            = delete;
  BounceBufferRing& operator=(BounceBufferRing const&) = delete;
  BounceBufferRing(BounceBufferRing&&)                 = delete;
  BounceBufferRing& operator=(BounceBufferRing&&)      = delete;

  /**
   * @brief Get pointer to the current bounce buffer.
   *
   * Use this to fill the buffer directly (e.g., via pread), then call submit_h2d() to transfer the
   * data to device.
   *
   * @return Pointer to the start of the current buffer.
   */
  [[nodiscard]] void* cur_buffer() const noexcept;

  /**
   * @brief Get pointer to the current bounce buffer at a specific offset.
   *
   * Useful for partial buffer fills or when accumulating data incrementally.
   *
   * @param offset Byte offset from the start of the current buffer.
   * @return Pointer to cur_buffer() + offset.
   */
  [[nodiscard]] void* cur_buffer(std::ptrdiff_t offset) const noexcept;

  /**
   * @brief Get the size of each bounce buffer in the ring.
   *
   * All buffers in the ring have the same size, determined by defaults::bounce_buffer_size() at
   * ring construction time.
   *
   * @return Size in bytes of each buffer.
   */
  [[nodiscard]] std::size_t buffer_size() const noexcept;

  /**
   * @brief Get the number of buffers in the ring (k for k-way overlap).
   *
   * @return Number of bounce buffers.
   */
  [[nodiscard]] std::size_t num_buffers() const noexcept;

  /**
   * @brief Get the current fill level of the active buffer.
   *
   * Indicates how many bytes have been accumulated in the current buffer via
   * accumulate_and_submit_h2d(). Reset to 0 after each advance().
   *
   * @return Number of bytes currently in the buffer.
   */
  [[nodiscard]] std::size_t cur_buffer_offset() const noexcept;

  /**
   * @brief Get remaining number of bytes in current buffer for accumulation.
   */
  [[nodiscard]] std::size_t cur_buffer_remaining_capacity() const noexcept;

  /**
   * @brief Advance to next buffer in the ring.
   *
   * Resets current buffer offset and moves to next buffer index. If wrapping around to the oldest
   * in-flight buffer, synchronizes the stream first to ensure that buffer's transfer is complete.
   *
   * @param stream CUDA stream to synchronize on wrap-around.
   */
  void advance(CUstream stream);

  /**
   * @brief Queue async copy from current bounce buffer to device memory.
   *
   * In non-batch mode, issues cuMemcpyHtoDAsync immediately. In batch mode, defers the copy until
   * wrap-around or synchronize().
   *
   * @param device_dst Device memory destination.
   * @param size Bytes to copy from cur_buffer().
   * @param stream CUDA stream for the async transfer.
   *
   * @note Does NOT advance to next buffer. Call submit_h2d() for copy + advance.
   */
  void enqueue_h2d(void* device_dst, std::size_t size, CUstream stream);

  /**
   * @brief Async copy from device memory to current bounce buffer.
   *
   * @param device_src Device memory source.
   * @param size Bytes to copy.
   * @param stream CUDA stream for the async transfer.
   */
  void enqueue_d2h(void* device_src, std::size_t size, CUstream stream);

  /**
   * @brief Accumulate data into bounce buffer, auto-submit when full.
   *
   * Copies host data into the internal buffer. When the buffer fills, issues an async H2D copy and
   * advances to the next buffer. Handles data larger than buffer_size() by splitting across
   * multiple buffers.
   *
   * Typical usage for streaming host data to device:
   * @code
   *   while (has_more_data()) {
   *     ring.accumulate_and_submit_h2d(device_ptr, host_data, chunk_size, stream);
   *     device_ptr += chunk_size;  // Note: only advance by submitted amount
   *   }
   *   ring.flush_h2d(device_ptr, stream);
   *   ring.synchronize(stream);
   * @endcode
   *
   * @param device_dst Device memory destination (should track cumulative offset externally).
   * @param host_src Source data in host memory.
   * @param size Bytes to copy.
   * @param stream CUDA stream for async H2D transfers.
   *
   * @note Partial buffer contents remain until flush_h2d() is called.
   * @note Final data visibility requires flush_h2d() + synchronize().
   */
  void accumulate_and_submit_h2d(void* device_dst,
                                 void const* host_src,
                                 std::size_t size,
                                 CUstream stream);

  /**
   * @brief Submit current buffer contents to device and advance to next buffer.
   *
   * Typical usage pattern for direct-fill (e.g., pread into buffer):
   * @code
   *   ssize_t n = pread(fd, ring.cur_buffer(), ring.buffer_size(), offset);
   *   ring.submit_h2d(device_ptr, n, stream);
   *   device_ptr += n;
   * @endcode
   *
   * @param device_dst Device memory destination.
   * @param size Bytes actually written to cur_buffer().
   * @param stream CUDA stream for async H2D transfer.
   *
   * @note Synchronization may occur if this causes a wrap-around.
   * @note Final data visibility requires calling synchronize() after all submits.
   */
  void submit_h2d(void* device_dst, std::size_t size, CUstream stream);

  /**
   * @brief Flush any partially accumulated data to device.
   *
   * Call after accumulate_and_submit_h2d() to submit remaining data that didn't fill a complete
   * buffer.
   *
   * @param device_dst Device memory destination for the partial buffer.
   * @param stream CUDA stream for async H2D transfer.
   * @return Number of bytes flushed (0 if buffer was empty).
   *
   * @note Still requires synchronize() for data visibility guarantee.
   */
  std::size_t flush_h2d(void* device_dst, CUstream stream);

  /**
   * @brief Ensure all queued H2D transfers are complete.
   *
   * In batch mode, issues any pending batch copies first. Then synchronizes the stream to guarantee
   * all data is visible in device memory.
   *
   * @param stream CUDA stream to synchronize.
   *
   * @note Must be called before reading transferred data on device.
   * @note After synchronize(), the ring can be reused for new transfers.
   */
  void synchronize(CUstream stream);
};
}  // namespace kvikio
