/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <memory>
#include <mutex>
#include <stack>

#include <kvikio/bounce_buffer.hpp>
#include <kvikio/defaults.hpp>
#include <kvikio/detail/event.hpp>
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
  KVIKIO_EXPECT(buffer != nullptr, "Aligned allocation failed");
  return buffer;
}

void PageAlignedAllocator::deallocate(void* buffer, std::size_t /*size*/) { std::free(buffer); }

void* CudaPinnedAllocator::allocate(std::size_t size)
{
  void* buffer{};

  // Allocate page-locked (pinned) host memory with CU_MEMHOSTALLOC_PORTABLE. The PORTABLE flag
  // ensures this memory is accessible from all CUDA contexts, which is essential for the singleton
  // BounceBufferPool that may serve multiple contexts and devices.
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
  // Register the page-aligned allocation as pinned memory with CU_MEMHOSTREGISTER_PORTABLE. The
  // PORTABLE flag ensures this memory is accessible from all CUDA contexts, which is essential for
  // the singleton BounceBufferPool that may serve multiple contexts and devices.
  CUDA_DRIVER_TRY(
    cudaAPI::instance().MemHostRegister(buffer, aligned_size, CU_MEMHOSTREGISTER_PORTABLE));
  return buffer;
}

void CudaPageAlignedPinnedAllocator::deallocate(void* buffer, std::size_t /*size*/)
{
  CUDA_DRIVER_TRY(cudaAPI::instance().MemHostUnregister(buffer));
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
BounceBufferPool<Allocator>::Buffer::Buffer(Buffer&& o) noexcept
  : _pool(std::exchange(o._pool, nullptr)),
    _buffer(std::exchange(o._buffer, nullptr)),
    _size(std::exchange(o._size, 0))
{
}

template <typename Allocator>
BounceBufferPool<Allocator>::Buffer& BounceBufferPool<Allocator>::Buffer::operator=(
  Buffer&& o) noexcept
{
  if (this != std::addressof(o)) {
    if (_buffer != nullptr) {
      // Return current buffer to the pool
      _pool->put(_buffer, _size);
    }
    _pool   = std::exchange(o._pool, nullptr);
    _buffer = std::exchange(o._buffer, nullptr);
    _size   = std::exchange(o._size, 0);
  }

  return *this;
}

template <typename Allocator>
void* BounceBufferPool<Allocator>::Buffer::get() const noexcept
{
  return _buffer;
}

template <typename Allocator>
void* BounceBufferPool<Allocator>::Buffer::get(std::ptrdiff_t offset) const noexcept
{
  return static_cast<std::byte*>(_buffer) + offset;
}

template <typename Allocator>
std::size_t BounceBufferPool<Allocator>::Buffer::size() const noexcept
{
  return _size;
}

template <typename Allocator>
std::size_t BounceBufferPool<Allocator>::_clear()
{
  KVIKIO_NVTX_FUNC_RANGE();
  std::size_t ret = _free_buffers.size() * _buffer_size;
  while (!_free_buffers.empty()) {
    _allocator.deallocate(_free_buffers.top(), _buffer_size);
    _free_buffers.pop();
  }
  return ret;
}

template <typename Allocator>
void BounceBufferPool<Allocator>::_ensure_buffer_size()
{
  KVIKIO_NVTX_FUNC_RANGE();
  auto const bounce_buffer_size = defaults::bounce_buffer_size();
  if (_buffer_size != bounce_buffer_size) {
    _clear();
    _buffer_size = bounce_buffer_size;
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
    return Buffer(this, ret, _buffer_size);
  }

  auto* buffer = _allocator.allocate(_buffer_size);
  return Buffer(this, buffer, _buffer_size);
}

template <typename Allocator>
void BounceBufferPool<Allocator>::put(void* buffer, std::size_t size)
{
  KVIKIO_NVTX_FUNC_RANGE();
  std::lock_guard const lock(_mutex);
  _ensure_buffer_size();

  // If the size of `buffer` matches the sizes of the retained allocations,
  // it is added to the set of free allocation otherwise it is freed.
  if (size == _buffer_size) {
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

template <typename Allocator>
BounceBufferRing<Allocator>::BounceBufferRing(std::size_t num_buffers)
{
  KVIKIO_NVTX_FUNC_RANGE();
  KVIKIO_EXPECT(num_buffers >= 1, "BounceBufferRing requires at least 1 buffer");

  _buffers.reserve(num_buffers);
  _events.reserve(num_buffers);
  for (std::size_t i = 0; i < num_buffers; ++i) {
    _buffers.emplace_back(BounceBufferPool<Allocator>::instance().get());
    _events.emplace_back(detail::EventPool::instance().get());
  }
}

template <typename Allocator>
void BounceBufferRing<Allocator>::enqueue_h2d(void* device_dst, std::size_t size, CUstream stream)
{
  CUDA_DRIVER_TRY(cudaAPI::instance().MemcpyHtoDAsync(
    convert_void2deviceptr(device_dst), cur_buffer(), size, stream));
}

template <typename Allocator>
void BounceBufferRing<Allocator>::enqueue_d2h(void* device_src, std::size_t size, CUstream stream)
{
  CUDA_DRIVER_TRY(cudaAPI::instance().MemcpyDtoHAsync(
    cur_buffer(), convert_void2deviceptr(device_src), size, stream));
}

template <typename Allocator>
void BounceBufferRing<Allocator>::advance(CUstream stream)
{
  KVIKIO_NVTX_FUNC_RANGE();
  _cur_buffer_offset = 0;
  ++_cur_buf_idx;
  if (_cur_buf_idx >= _buffers.size()) { _cur_buf_idx = 0; }

  // Ensure the buffer we are advancing into is not still being read by a prior H2D
  auto event        = _events[_cur_buf_idx].get();
  auto event_status = cudaAPI::instance().EventQuery(event);
  if (event_status == CUDA_ERROR_NOT_READY) {
    CUDA_DRIVER_TRY(cudaAPI::instance().EventSynchronize(event));
  } else if (event_status != CUDA_SUCCESS) {
    CUDA_DRIVER_TRY(event_status);
  }
}

template <typename Allocator>
void* BounceBufferRing<Allocator>::cur_buffer() const noexcept
{
  return _buffers[_cur_buf_idx].get();
}

template <typename Allocator>
void* BounceBufferRing<Allocator>::cur_buffer(std::ptrdiff_t off) const noexcept
{
  return _buffers[_cur_buf_idx].get(off);
}

template <typename Allocator>
std::size_t BounceBufferRing<Allocator>::buffer_size() const noexcept
{
  return _buffers[_cur_buf_idx].size();
}

template <typename Allocator>
std::size_t BounceBufferRing<Allocator>::num_buffers() const noexcept
{
  return _buffers.size();
}

template <typename Allocator>
std::size_t BounceBufferRing<Allocator>::cur_buffer_offset() const noexcept
{
  return _cur_buffer_offset;
}

template <typename Allocator>
std::size_t BounceBufferRing<Allocator>::cur_buffer_remaining_capacity() const noexcept
{
  return buffer_size() - _cur_buffer_offset;
}

template <typename Allocator>
std::size_t BounceBufferRing<Allocator>::accumulate_and_submit_h2d(void* device_dst,
                                                                   void const* host_src,
                                                                   std::size_t size,
                                                                   CUstream stream)
{
  KVIKIO_NVTX_FUNC_RANGE();
  auto const* host_src_ptr = static_cast<std::byte const*>(host_src);
  auto* device_dst_ptr     = static_cast<std::byte*>(device_dst);

  // The data is split across multiple buffers if its size is greater than buffer_size()
  while (size > 0) {
    auto const remaining_bytes   = cur_buffer_remaining_capacity();
    auto const num_bytes_to_copy = std::min(size, remaining_bytes);

    // Copy from host to bounce buffer
    std::memcpy(cur_buffer(_cur_buffer_offset), host_src_ptr, num_bytes_to_copy);
    _cur_buffer_offset += num_bytes_to_copy;
    host_src_ptr += num_bytes_to_copy;
    size -= num_bytes_to_copy;

    if (_cur_buffer_offset >= buffer_size()) {
      submit_h2d(device_dst_ptr, buffer_size(), stream);
      device_dst_ptr += buffer_size();
    }
  }

  return static_cast<std::size_t>(device_dst_ptr - static_cast<std::byte*>(device_dst));
}

template <typename Allocator>
void BounceBufferRing<Allocator>::submit_h2d(void* device_dst, std::size_t size, CUstream stream)
{
  KVIKIO_NVTX_FUNC_RANGE();
  enqueue_h2d(device_dst, size, stream);
  CUDA_DRIVER_TRY(cudaAPI::instance().EventRecord(_events[_cur_buf_idx].get(), stream));
  advance();
}

template <typename Allocator>
std::size_t BounceBufferRing<Allocator>::flush_h2d(void* device_dst, CUstream stream)
{
  KVIKIO_NVTX_FUNC_RANGE();
  if (_cur_buffer_offset == 0) { return 0; }
  auto const flushed = _cur_buffer_offset;
  submit_h2d(device_dst, _cur_buffer_offset, stream);
  return flushed;
}

template <typename Allocator>
void BounceBufferRing<Allocator>::synchronize(CUstream stream)
{
  KVIKIO_NVTX_FUNC_RANGE();
  CUDA_DRIVER_TRY(cudaAPI::instance().StreamSynchronize(stream));
}

template <typename Allocator>
void BounceBufferRing<Allocator>::reset(CUstream stream)
{
  synchronize(stream);
  _cur_buf_idx       = 0;
  _cur_buffer_offset = 0;
}

// Explicit instantiations
template class BounceBufferRing<CudaPinnedAllocator>;
template class BounceBufferRing<CudaPageAlignedPinnedAllocator>;

template <typename Allocator>
BounceBufferRingCachePerThreadAndContext<Allocator>::Ring&
BounceBufferRingCachePerThreadAndContext<Allocator>::ring()
{
  KVIKIO_NVTX_FUNC_RANGE();

  CUcontext ctx{nullptr};
  CUDA_DRIVER_TRY(cudaAPI::instance().CtxGetCurrent(&ctx));
  KVIKIO_EXPECT(ctx != nullptr, "No CUDA context is current");
  auto key = std::make_pair(ctx, std::this_thread::get_id());

  std::lock_guard const lock(_mutex);

  auto it = _rings.find(key);
  if (it == _rings.end()) {
    auto ring = new Ring(defaults::bounce_buffer_count());
    it        = _rings.emplace(key, ring).first;
  }
  return *it->second;
}

template <typename Allocator>
BounceBufferRingCachePerThreadAndContext<Allocator>&
BounceBufferRingCachePerThreadAndContext<Allocator>::instance()
{
  static BounceBufferRingCachePerThreadAndContext<Allocator> instance;
  return instance;
}

// Explicit instantiations
template class BounceBufferRingCachePerThreadAndContext<CudaPinnedAllocator>;
template class BounceBufferRingCachePerThreadAndContext<CudaPageAlignedPinnedAllocator>;

}  // namespace kvikio
