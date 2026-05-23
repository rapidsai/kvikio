/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stack>
#include <thread>
#include <utility>
#include <vector>

#include <kvikio/defaults.hpp>
#include <kvikio/shim/cuda.hpp>

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
   *
   * @note noexcept. Any failure during the underlying push or deallocation (e.g., allocator
   * out-of-memory in `std::stack::push`, or a CUDA error in the deallocate path on size mismatch)
   * is caught and logged. This guarantee is load-bearing for `Buffer::~Buffer`, which would
   * otherwise `std::terminate` on a thrown exception.
   */
  void put(void* buffer, std::size_t size) noexcept;

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
 * @brief Per-(thread, CUDA context) cache of bounce buffers with async recycling.
 *
 * Wraps `BounceBufferPool<Allocator>` with a fan-out free-list keyed by
 * `(std::thread::id, CUcontext)`. Each key tracks three counts that together are capped at
 * `cap`: a free list of recyclable `Buffer`s, a count of buffers currently checked out to
 * callers (Phase A), and a count of pending CUDA-side recycle callbacks (Phase B).
 *
 * Typical lifecycle of a buffer:
 *   1. Caller invokes `try_get(ctx)` to acquire a `Buffer` (Phase A begins). The buffer is
 *      filled with host data.
 *   2. Caller submits a `cuMemcpyAsync` to a `stream` and calls `recycle_after(ctx, buf,
 *      stream)`. The cache schedules a `cuLaunchHostFunc` on the stream that will recycle the
 *      buffer to the free list when the H2D completes (Phase A ends, Phase B begins).
 *   3. The CUDA driver invokes the recycle callback on a driver thread. The callback moves the
 *      Buffer from "in flight" to "free" (Phase B ends).
 *
 * Failure paths (no H2D submitted) use `recycle_now(ctx, buf)` which returns the buffer to the
 * free list immediately.
 *
 * `try_get` is non-blocking: it returns `std::nullopt` when the cap is reached. Backpressure on
 * the caller side decides whether to retry, defer, or block.
 *
 * @tparam Allocator The allocator policy used by the underlying `BounceBufferPool`. Common
 * choices are `CudaPinnedAllocator` (for device I/O) and `CudaPageAlignedPinnedAllocator` (for
 * Direct I/O with device memory).
 *
 * @note The class itself is decoupled from kvikio's defaults subsystem; callers pass an
 * explicit `cap` to the constructor. The singleton accessor `instance()` is the bridge that
 * reads `defaults::num_bounce_buffers_per_cache()` once and constructs the process-wide
 * instance. Tests can construct local instances with arbitrary caps.
 *
 * @note Each in-flight `Buffer` must be checked out and returned by the same `(thread, ctx)`
 * key. Mixing threads or contexts within one buffer's lifecycle breaks the per-key accounting.
 */
template <typename Allocator = CudaPinnedAllocator>
class BounceBufferCachePerThreadAndContext {
 public:
  using Buffer = typename BounceBufferPool<Allocator>::Buffer;

  /**
   * @brief Construct a cache with the given per-(thread, ctx) buffer cap.
   *
   * @param cap Maximum number of buffers (free + checked-out + in-flight) per
   * `(thread, CUcontext)` key. A value of 0 is a sentinel for "no cap": `try_get` never returns
   * `nullopt`, the cache grows on demand, and backpressure becomes the caller's responsibility.
   */
  explicit BounceBufferCachePerThreadAndContext(std::size_t cap);

  // Non-copyable, non-movable (holds a mutex; per-key state has pointer identity exposed to
  // the recycle callback).
  BounceBufferCachePerThreadAndContext(BounceBufferCachePerThreadAndContext const&) = delete;
  BounceBufferCachePerThreadAndContext& operator=(BounceBufferCachePerThreadAndContext const&) =
    delete;
  BounceBufferCachePerThreadAndContext(BounceBufferCachePerThreadAndContext&&)            = delete;
  BounceBufferCachePerThreadAndContext& operator=(BounceBufferCachePerThreadAndContext&&) = delete;

  ~BounceBufferCachePerThreadAndContext() = default;

  /**
   * @brief Get the process-wide singleton instance.
   *
   * The instance is constructed lazily on first call with the cap from
   * `defaults::num_bounce_buffers_per_cache()`. The singleton is intentionally heap-allocated
   * and never deleted; its destructor never runs at process exit, which avoids the
   * CUDA-driver-API-in-static-destructor undefined behavior that would otherwise occur when
   * cached `Buffer`s destruct (and route through `BounceBufferPool::put`'s size-mismatch
   * deallocation path, which can call `cuMemFreeHost`).
   *
   * Each template instantiation (different `Allocator`) has its own singleton.
   *
   * @return Reference to the singleton instance.
   */
  KVIKIO_EXPORT static BounceBufferCachePerThreadAndContext& instance();

  /**
   * @brief Get the per-(thread, ctx) buffer cap this instance was constructed with.
   *
   * @return The cap. A value of 0 means "no cap".
   */
  [[nodiscard]] std::size_t cap() const noexcept;

  /**
   * @brief Try to acquire a buffer for the given CUDA context (non-blocking).
   *
   * Pops from the per-(this thread, ctx) free list if non-empty. Otherwise allocates a fresh
   * Buffer from the underlying `BounceBufferPool` if the per-key count
   * (`free + checked_out + in_flight`) is below `cap()` (or `cap() == 0`). When the cap is
   * reached, returns `std::nullopt`.
   *
   * @param ctx The CUDA context to associate the buffer with. The caller is expected to have
   * `ctx` current on the calling thread before calling.
   * @return A `Buffer` on success, `std::nullopt` if the cap is reached.
   *
   * @exception kvikio::CUfileException if the underlying allocation fails.
   */
  [[nodiscard]] std::optional<Buffer> try_get(CUcontext ctx);

  /**
   * @brief Schedule the buffer to be recycled when the stream's prior work completes.
   *
   * Records a `cuLaunchHostFunc` on `stream` so that, once `stream` has finished the
   * `cuMemcpyAsync` (or whatever CUDA work) that consumes `buf`, the buffer is returned to the
   * free list on a CUDA driver thread. Non-blocking on the calling thread.
   *
   * @param ctx The CUDA context the buffer was acquired under (must match the original
   * `try_get` call).
   * @param buf The buffer (ownership transferred into the cache).
   * @param stream The CUDA stream whose completion signals that `buf` is safe to recycle.
   *
   * @exception kvikio::CUfileException if `cuLaunchHostFunc` fails. In that case the recovery
   * path synchronizes `stream` to ensure the buffer is safe, returns it to the free list, then
   * propagates the error.
   */
  void recycle_after(CUcontext ctx, Buffer&& buf, CUstream stream);

  /**
   * @brief Recycle the buffer immediately, without involving CUDA.
   *
   * Used on failure paths where no CUDA work was submitted with this buffer.
   *
   * @param ctx The CUDA context the buffer was acquired under.
   * @param buf The buffer (ownership transferred into the cache).
   */
  void recycle_now(CUcontext ctx, Buffer&& buf);

 private:
  struct PerKey {
    std::mutex mutex;
    std::vector<Buffer> free;
    std::size_t checked_out{0};
    std::size_t in_flight{0};
    // Invariant: cap == 0 || free.size() + checked_out + in_flight <= cap
  };

  // Heap-allocated payload passed to `cuLaunchHostFunc` as user data. The callback deletes it
  // after moving the buffer into the free list.
  struct RecycleCallbackData {
    PerKey* per_key;
    Buffer buffer;
  };

  static void CUDA_CB recycle_callback(void* user_data);

  // Locate or lazily create the PerKey state for (this thread, ctx).
  PerKey& get_per_key(CUcontext ctx);

  std::size_t _cap;
  std::mutex _map_mutex;  // first-touch insertion into _per_key
  std::map<std::pair<std::thread::id, CUcontext>, std::unique_ptr<PerKey>> _per_key;
};

}  // namespace kvikio
