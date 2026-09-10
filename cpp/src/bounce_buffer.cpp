/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <exception>
#include <memory>
#include <mutex>
#include <stack>
#include <tuple>
#include <utility>

#include <kvikio/bounce_buffer.hpp>
#include <kvikio/defaults.hpp>
#include <kvikio/detail/nvtx.hpp>
#include <kvikio/detail/utils.hpp>
#include <kvikio/error.hpp>
#include <kvikio/logger.hpp>
#include <kvikio/shim/cuda.hpp>

namespace kvikio {

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

void* CudaPinnedAllocator::allocate(std::size_t size)
{
  void* buffer{};

  // If no available allocation, allocate and register a new one
  // Allocate page-locked host memory
  // Under unified addressing, host memory allocated this way is automatically portable and
  // mapped.
  KVIKIO_CUDA_DRIVER_TRY(cudaAPI::instance().MemHostAlloc(&buffer, size, CU_MEMHOSTALLOC_PORTABLE));

  return buffer;
}
void CudaPinnedAllocator::deallocate(void* buffer, std::size_t /*size*/)
{
  KVIKIO_CUDA_DRIVER_TRY(cudaAPI::instance().MemFreeHost(buffer));
}

void* CudaPageAlignedPinnedAllocator::allocate(std::size_t size)
{
  void* buffer{};
  auto const page_size    = get_page_size();
  auto const aligned_size = detail::align_up(size, page_size);
  buffer                  = std::aligned_alloc(page_size, aligned_size);
  KVIKIO_EXPECT(buffer != nullptr, "Aligned allocation failed");
  KVIKIO_CUDA_DRIVER_TRY(
    cudaAPI::instance().MemHostRegister(buffer, aligned_size, CU_MEMHOSTALLOC_PORTABLE));
  return buffer;
}

void CudaPageAlignedPinnedAllocator::deallocate(void* buffer, std::size_t /*size*/)
{
  KVIKIO_CUDA_DRIVER_TRY(cudaAPI::instance().MemHostUnregister(buffer));
  std::free(buffer);
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
