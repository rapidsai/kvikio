/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <mutex>
#include <stack>

#include <kvikio/bounce_buffer.hpp>
#include <kvikio/defaults.hpp>
#include <kvikio/detail/nvtx.hpp>
#include <kvikio/detail/utils.hpp>
#include <kvikio/error.hpp>
#include <kvikio/shim/cuda.hpp>

namespace kvikio {

void* PageAlignedAllocator::allocate(std::size_t size)
{
  void* buffer{};
  auto const page_size    = get_page_size();
  auto const aligned_size = detail::align_up(size, page_size);
  buffer                  = std::aligned_alloc(page_size, aligned_size);
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
  CUDA_DRIVER_TRY(cudaAPI::instance().MemHostAlloc(&buffer, size, CU_MEMHOSTALLOC_PORTABLE));

  return buffer;
}
void CudaPinnedAllocator::deallocate(void* buffer, std::size_t /*size*/)
{
  CUDA_DRIVER_TRY(cudaAPI::instance().MemFreeHost(buffer));
}

void* CudaPageAlignedPinnedAllocator::allocate(std::size_t size)
{
  void* buffer{};
  auto const page_size    = get_page_size();
  auto const aligned_size = detail::align_up(size, page_size);
  buffer                  = std::aligned_alloc(page_size, aligned_size);
  KVIKIO_EXPECT(buffer != nullptr, "Aligned allocation failed");
  CUDA_DRIVER_TRY(
    cudaAPI::instance().MemHostRegister(buffer, aligned_size, CU_MEMHOSTALLOC_PORTABLE));
  return buffer;
}

void CudaPageAlignedPinnedAllocator::deallocate(void* buffer, std::size_t /*size*/)
{
  CUDA_DRIVER_TRY(cudaAPI::instance().MemHostUnregister(buffer));
  std::free(buffer);
}

template <typename Allocator>
BounceBufferPool<Allocator>::Buffer::Buffer(BounceBufferPool<Allocator>* manager,
                                            void* buffer,
                                            std::size_t size)
  : _manager(manager), _buffer{buffer}, _size{size}
{
}

template <typename Allocator>
BounceBufferPool<Allocator>::Buffer::~Buffer() noexcept
{
  KVIKIO_NVTX_FUNC_RANGE();
  _manager->put(_buffer, _size);
}

template <typename Allocator>
void* BounceBufferPool<Allocator>::Buffer::get() noexcept
{
  KVIKIO_NVTX_FUNC_RANGE();
  return _buffer;
}

template <typename Allocator>
void* BounceBufferPool<Allocator>::Buffer::get(std::ptrdiff_t offset) noexcept
{
  KVIKIO_NVTX_FUNC_RANGE();
  return static_cast<char*>(_buffer) + offset;
}

template <typename Allocator>
std::size_t BounceBufferPool<Allocator>::Buffer::size() noexcept
{
  return _size;
}

template <typename Allocator>
std::size_t BounceBufferPool<Allocator>::_clear()
{
  KVIKIO_NVTX_FUNC_RANGE();
  std::size_t ret = _free_buffers.size() * _size;
  while (!_free_buffers.empty()) {
    _allocator.deallocate(_free_buffers.top(), _size);
    _free_buffers.pop();
  }
  return ret;
}

template <typename Allocator>
void BounceBufferPool<Allocator>::_ensure_buffer_size()
{
  KVIKIO_NVTX_FUNC_RANGE();
  auto const bounce_buffer_size = defaults::bounce_buffer_size();
  if (_size != bounce_buffer_size) {
    _clear();
    _size = bounce_buffer_size;
  }
}

template <typename Allocator>
BounceBufferPool<Allocator>::Buffer BounceBufferPool<Allocator>::get()
{
  KVIKIO_NVTX_FUNC_RANGE();
  std::lock_guard const lock(_mutex);
  _ensure_buffer_size();

  // Check if we have an allocation available
  if (!_free_buffers.empty()) {
    void* ret = _free_buffers.top();
    _free_buffers.pop();
    return Buffer(this, ret, _size);
  }

  auto* buffer = _allocator.allocate(_size);
  return Buffer(this, buffer, _size);
}

template <typename Allocator>
void BounceBufferPool<Allocator>::put(void* buffer, std::size_t size)
{
  KVIKIO_NVTX_FUNC_RANGE();
  std::lock_guard const lock(_mutex);
  _ensure_buffer_size();

  // If the size of `buffer` matches the sizes of the retained allocations,
  // it is added to the set of free allocation otherwise it is freed.
  if (size == _size) {
    _free_buffers.push(buffer);
  } else {
    _allocator.deallocate(buffer, size);
  }
}

template <typename Allocator>
std::size_t BounceBufferPool<Allocator>::clear()
{
  KVIKIO_NVTX_FUNC_RANGE();
  std::lock_guard const lock(_mutex);
  return _clear();
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
