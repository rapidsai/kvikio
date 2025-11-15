/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
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
  std::mutex _mutex{};
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
   * @note Non-copyable and non-movable to ensure single ownership
   */
  class Buffer {
   private:
    BounceBufferPool* _pool;
    void* _buffer;
    std::size_t const _size;

   public:
    Buffer(BounceBufferPool<Allocator>* pool, void* buffer, std::size_t size);
    Buffer(Buffer const&)            = delete;
    Buffer& operator=(Buffer const&) = delete;
    Buffer(Buffer&& o)               = delete;
    Buffer& operator=(Buffer&& o)    = delete;
    ~Buffer() noexcept;
    void* get() noexcept;
    void* get(std::ptrdiff_t offset) noexcept;
    std::size_t size() noexcept;
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
}  // namespace kvikio
