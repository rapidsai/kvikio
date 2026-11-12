/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <sys/mman.h>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <future>
#include <memory>
#include <mutex>
#include <stack>
#include <tuple>
#include <utility>
#include <vector>

#include <kvikio/bounce_buffer.hpp>
#include <kvikio/defaults.hpp>
#include <kvikio/detail/nvtx.hpp>
#include <kvikio/detail/utils.hpp>
#include <kvikio/error.hpp>
#include <kvikio/logger.hpp>
#include <kvikio/shim/cuda.hpp>
#include <kvikio/threadpool_wrapper.hpp>

namespace kvikio {
namespace {

/// Transparent huge page size on x86-64 and on the usual aarch64 configuration.
constexpr std::size_t huge_page_size = 2 * 1024 * 1024;

/**
 * @brief Alignment, and length granularity, of the mapping that backs a pinned allocation
 *
 * @param size Requested size in bytes
 * @return The huge page size for allocations of at least one huge page, the page size otherwise
 */
std::size_t pinned_alignment(std::size_t size)
{
  return size >= huge_page_size ? huge_page_size : get_page_size();
}

/**
 * @brief The dedicated pool that helps prefault pinned allocations
 *
 * Created on first use. After `defaults::bounce_buffer_prefault_nthreads()` changes, the next call
 * creates a pool of the new size, and callers still holding the old pool keep it alive until they
 * are done.
 *
 * @return The pool, or nullptr when `defaults::bounce_buffer_prefault_nthreads()` is 0
 */
std::shared_ptr<ThreadPool> prefault_thread_pool()
{
  static std::mutex mutex;
  static std::shared_ptr<ThreadPool> pool;
  auto const nthreads = defaults::bounce_buffer_prefault_nthreads();
  std::lock_guard const lock(mutex);
  if (nthreads == 0) {
    pool.reset();
  } else if (pool == nullptr || pool->get_thread_count() != nthreads) {
    pool = std::make_shared<ThreadPool>(nthreads, make_thread_pool_init_task("kvikio-pf"));
  }
  return pool;
}

/**
 * @brief Prefault a range for writing, split between the calling thread and the prefault pool
 *
 * The calling thread takes the first part. If a part cannot be handed to the pool, the calling
 * thread prefaults what is left.
 *
 * @param ptr Start of the range, page-aligned
 * @param size Length of the range in bytes
 */
void prefault_in_parallel(void* ptr, std::size_t size)
{
  // Each part is at least `min_part_size`, so a small allocation is not split.
  constexpr std::size_t min_part_size = 4 * 1024 * 1024;
  auto const pool                     = prefault_thread_pool();
  auto const max_parts = pool == nullptr ? std::size_t{1} : pool->get_thread_count() + 1;
  auto const nparts    = std::clamp<std::size_t>(size / min_part_size, 1, max_parts);
  auto const part      = detail::align_up((size + nparts - 1) / nparts, get_page_size());

  std::vector<std::future<void>> futures;
  futures.reserve(nparts - 1);
  std::size_t offset = part;
  try {
    for (; offset < size; offset += part) {
      futures.push_back(pool->submit_task([=] {
        detail::perform_prefault(
          detail::pointer_add(ptr, offset), std::min(part, size - offset), /*write=*/true);
      }));
    }
  } catch (...) {
    detail::perform_prefault(detail::pointer_add(ptr, offset), size - offset, /*write=*/true);
  }
  detail::perform_prefault(ptr, std::min(part, size), /*write=*/true);
  for (auto& future : futures) {
    future.get();
  }
}

/**
 * @brief Allocate portable pinned host memory
 *
 * Pinning costs about the same per byte however it is split into calls, and the driver serializes
 * it across threads. Most of that cost is the kernel zeroing pages as they fault in, which does run
 * in parallel. So the memory is mapped with transparent huge pages where available, faulted in from
 * several threads, and only then registered with `cuMemHostRegister`. This is several times faster
 * than `cuMemHostAlloc`.
 *
 * @param size Requested size in bytes
 * @return Page-aligned pointer to the pinned memory
 */
void* allocate_pinned(std::size_t size)
{
  auto const alignment = pinned_alignment(size);
  auto const length    = detail::align_up(size, alignment);

  // Map one alignment unit more than needed, then unmap the head and the tail so that the mapping
  // starts at an aligned address and is exactly `length` long.
  void* const mapped =
    mmap(nullptr, length + alignment, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  KVIKIO_EXPECT(mapped != MAP_FAILED, "Pinned allocation failed: mmap", std::runtime_error);
  auto* const raw = static_cast<char*>(mapped);
  auto* const ptr = static_cast<char*>(detail::align_up(mapped, alignment));
  if (ptr > raw) { munmap(raw, static_cast<std::size_t>(ptr - raw)); }
  auto const tail = static_cast<std::size_t>((raw + length + alignment) - (ptr + length));
  if (tail > 0) { munmap(ptr + length, tail); }

  // A hint. Where transparent huge pages are disabled, the range is backed by regular pages.
  if (alignment == huge_page_size) { madvise(ptr, length, MADV_HUGEPAGE); }

  try {
    prefault_in_parallel(ptr, length);
    KVIKIO_CUDA_DRIVER_TRY(
      cudaAPI::instance().MemHostRegister(ptr, length, CU_MEMHOSTREGISTER_PORTABLE));
  } catch (...) {
    munmap(ptr, length);
    throw;
  }
  return ptr;
}

/**
 * @brief Deallocate memory from `allocate_pinned`
 *
 * @param buffer Pointer returned by `allocate_pinned`
 * @param size The size passed to `allocate_pinned`
 */
void deallocate_pinned(void* buffer, std::size_t size)
{
  KVIKIO_CUDA_DRIVER_TRY(cudaAPI::instance().MemHostUnregister(buffer));
  munmap(buffer, detail::align_up(size, pinned_alignment(size)));
}

}  // namespace

void* PageAlignedAllocator::allocate(std::size_t size)
{
  void* buffer{};
  auto const page_size    = get_page_size();
  auto const aligned_size = detail::align_up(size, page_size);
  buffer                  = std::aligned_alloc(page_size, aligned_size);
  KVIKIO_EXPECT(buffer != nullptr, "Aligned allocation failed");
  return buffer;
}

void PageAlignedAllocator::deallocate(void* buffer, std::size_t /*size*/) { std::free(buffer); }

void* CudaPinnedAllocator::allocate(std::size_t size) { return allocate_pinned(size); }

void CudaPinnedAllocator::deallocate(void* buffer, std::size_t size)
{
  deallocate_pinned(buffer, size);
}

void* CudaPageAlignedPinnedAllocator::allocate(std::size_t size) { return allocate_pinned(size); }

void CudaPageAlignedPinnedAllocator::deallocate(void* buffer, std::size_t size)
{
  deallocate_pinned(buffer, size);
}

template <typename Allocator>
BounceBufferPool<Allocator>::Buffer::Buffer(BounceBufferPool<Allocator>* pool,
                                            void* buffer,
                                            std::size_t size)
  : _pool(pool), _buffer{buffer}, _size{size}
{
}

template <typename Allocator>
BounceBufferPool<Allocator>::Buffer::~Buffer() noexcept
{
  KVIKIO_NVTX_FUNC_RANGE();
  if (_buffer) {
    // Only return to the pool if not moved-from
    _pool->put(_buffer, _size);
  }
}

template <typename Allocator>
BounceBufferPool<Allocator>::Buffer::Buffer(Buffer&& other) noexcept
  : _pool(std::exchange(other._pool, nullptr)),
    _buffer(std::exchange(other._buffer, nullptr)),
    _size(std::exchange(other._size, 0))
{
}

template <typename Allocator>
BounceBufferPool<Allocator>::Buffer& BounceBufferPool<Allocator>::Buffer::operator=(
  Buffer&& other) noexcept
{
  if (this != std::addressof(other)) {
    if (_buffer != nullptr) {
      // Return current buffer to the pool
      _pool->put(_buffer, _size);
    }
    _pool   = std::exchange(other._pool, nullptr);
    _buffer = std::exchange(other._buffer, nullptr);
    _size   = std::exchange(other._size, 0);
  }

  return *this;
}

template <typename Allocator>
void* BounceBufferPool<Allocator>::Buffer::get() const noexcept
{
  KVIKIO_NVTX_FUNC_RANGE();
  return _buffer;
}

template <typename Allocator>
void* BounceBufferPool<Allocator>::Buffer::get(std::ptrdiff_t offset) const noexcept
{
  KVIKIO_NVTX_FUNC_RANGE();
  return static_cast<char*>(_buffer) + offset;
}

template <typename Allocator>
std::size_t BounceBufferPool<Allocator>::Buffer::size() const noexcept
{
  return _size;
}

template <typename Allocator>
void BounceBufferPool<Allocator>::_deallocate_buffers(std::stack<void*>& buffers,
                                                      std::size_t buffer_size)
{
  KVIKIO_NVTX_FUNC_RANGE();
  while (!buffers.empty()) {
    _allocator.deallocate(buffers.top(), buffer_size);
    buffers.pop();
  }
}

template <typename Allocator>
std::pair<std::stack<void*>, std::size_t> BounceBufferPool<Allocator>::_detach_free_buffers()
{
  auto stale_size    = _buffer_size;
  auto stale_buffers = std::exchange(_free_buffers, {});
  return {std::move(stale_buffers), stale_size};
}

template <typename Allocator>
std::pair<std::stack<void*>, std::size_t> BounceBufferPool<Allocator>::_ensure_buffer_size()
{
  KVIKIO_NVTX_FUNC_RANGE();
  auto const bounce_buffer_size = defaults::bounce_buffer_size();
  if (_buffer_size == bounce_buffer_size) { return {}; }

  auto stale_buffers = _detach_free_buffers();
  _buffer_size       = bounce_buffer_size;
  return stale_buffers;
}

template <typename Allocator>
BounceBufferPool<Allocator>::Buffer BounceBufferPool<Allocator>::get()
{
  KVIKIO_NVTX_FUNC_RANGE();
  void* reused_buffer{nullptr};
  std::size_t buffer_size{};
  std::stack<void*> stale_buffers{};
  std::size_t stale_size{};
  {
    std::lock_guard const lock(_mutex);
    std::tie(stale_buffers, stale_size) = _ensure_buffer_size();
    buffer_size                         = _buffer_size;

    // Check if we have an allocation available
    if (!_free_buffers.empty()) {
      reused_buffer = _free_buffers.top();
      _free_buffers.pop();
    }
  }

  _deallocate_buffers(stale_buffers, stale_size);

  if (reused_buffer != nullptr) { return Buffer(this, reused_buffer, buffer_size); }
  auto* buffer = _allocator.allocate(buffer_size);
  return Buffer(this, buffer, buffer_size);
}

template <typename Allocator>
void BounceBufferPool<Allocator>::put(void* buffer, std::size_t size) noexcept
{
  KVIKIO_NVTX_FUNC_RANGE();
  try {
    bool is_incoming_stale{false};
    std::stack<void*> stale_buffers{};
    std::size_t stale_size{};
    {
      std::lock_guard const lock(_mutex);
      std::tie(stale_buffers, stale_size) = _ensure_buffer_size();

      // If the size of `buffer` matches the sizes of the retained allocations,
      // it is added to the set of free allocation otherwise it is freed.
      if (size == _buffer_size) {
        _free_buffers.push(buffer);
      } else {
        is_incoming_stale = true;
      }
    }

    _deallocate_buffers(stale_buffers, stale_size);
    if (is_incoming_stale) { _allocator.deallocate(buffer, size); }
  } catch (std::exception const& e) {
    KVIKIO_LOG_ERROR(std::string("BounceBufferPool::put failed: ") + e.what());
  } catch (...) {
    KVIKIO_LOG_ERROR("BounceBufferPool::put failed: unknown exception");
  }
}

template <typename Allocator>
std::size_t BounceBufferPool<Allocator>::clear()
{
  KVIKIO_NVTX_FUNC_RANGE();
  std::stack<void*> stale_buffers{};
  std::size_t stale_size{};
  std::size_t total_stale_size{};
  {
    std::lock_guard const lock(_mutex);
    std::tie(stale_buffers, stale_size) = _detach_free_buffers();
    total_stale_size                    = stale_buffers.size() * stale_size;
  }
  _deallocate_buffers(stale_buffers, stale_size);
  return total_stale_size;
}

template <typename Allocator>
std::size_t BounceBufferPool<Allocator>::num_free_buffers() const
{
  std::lock_guard const lock(_mutex);
  return _free_buffers.size();
}

template <typename Allocator>
std::size_t BounceBufferPool<Allocator>::buffer_size() const
{
  std::lock_guard const lock(_mutex);
  return _buffer_size;
}

template <typename Allocator>
BounceBufferPool<Allocator>& BounceBufferPool<Allocator>::instance()
{
  KVIKIO_NVTX_FUNC_RANGE();
  static BounceBufferPool _instance;
  return _instance;
}

// Explicit instantiations
template class BounceBufferPool<PageAlignedAllocator>;
template class BounceBufferPool<CudaPinnedAllocator>;
template class BounceBufferPool<CudaPageAlignedPinnedAllocator>;
}  // namespace kvikio
