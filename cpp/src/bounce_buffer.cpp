/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <memory>
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

namespace {
void separate_copy(std::span<void*> dsts,
                   std::span<void*> srcs,
                   std::span<std::size_t> sizes,
                   CUstream stream)
{
  // Fall back to the conventional H2D copy if the batch copy API is not available.
  for (std::size_t i = 0; i < srcs.size(); ++i) {
    CUDA_DRIVER_TRY(cudaAPI::instance().MemcpyHtoDAsync(
      convert_void2deviceptr(dsts[i]), reinterpret_cast<void*>(srcs[i]), sizes[i], stream));
  }
}

void batch_copy(std::span<void*> dsts,
                std::span<void*> srcs,
                std::span<std::size_t> sizes,
                CUstream stream)
{
  if (srcs.size() == 0) return;

#if CUDA_VERSION >= 12080
  if (cudaAPI::instance().MemcpyBatchAsync) {
    CUmemcpyAttributes attrs{};
    std::size_t attrs_idxs[] = {0};
    attrs.srcAccessOrder     = CUmemcpySrcAccessOrder_enum::CU_MEMCPY_SRC_ACCESS_ORDER_STREAM;
    CUDA_DRIVER_TRY(
      cudaAPI::instance().MemcpyBatchAsync(dsts.data(),
                                           srcs.data(),
                                           sizes.data(),
                                           srcs.size(),
                                           &attrs,
                                           attrs_idxs,
                                           static_cast<std::size_t>(1) /* num_attrs */,
#if CUDA_VERSION < 13000
                                           static_cast<std::size_t*>(nullptr),
#endif
                                           stream));
  } else {
    separate_copy(dsts, srcs, sizes, stream);
  }
#else
  separate_copy(dsts, srcs, sizes, stream);
#endif
}
}  // namespace

template <typename Allocator>
void BounceBufferRing<Allocator>::BatchTransferContext::add_entry(void* dst,
                                                                  void* src,
                                                                  std::size_t size)
{
  srcs.push_back(src);
  dsts.push_back(dst);
  sizes.push_back(size);
}

template <typename Allocator>
void BounceBufferRing<Allocator>::BatchTransferContext::clear()
{
  srcs.clear();
  dsts.clear();
  sizes.clear();
}

template <typename Allocator>
BounceBufferRing<Allocator>::BounceBufferRing(std::size_t num_buffers, bool batch_copy)
  : _num_buffers{num_buffers}, _batch_copy{batch_copy}
{
  KVIKIO_NVTX_FUNC_RANGE();
  KVIKIO_EXPECT(num_buffers >= 1, "BounceBufferRing requires at least 1 buffer");

  _buffers.reserve(_num_buffers);
  for (std::size_t i = 0; i < _num_buffers; ++i) {
    _buffers.emplace_back(BounceBufferPool<Allocator>::instance().get());
  }
}

template <typename Allocator>
void BounceBufferRing<Allocator>::enqueue_h2d(void* device_dst, std::size_t size, CUstream stream)
{
  KVIKIO_NVTX_FUNC_RANGE();
  if (_batch_copy) {
    _batch_transfer_ctx.add_entry(device_dst, cur_buffer(), size);
  } else {
    CUDA_DRIVER_TRY(cudaAPI::instance().MemcpyHtoDAsync(
      convert_void2deviceptr(device_dst), cur_buffer(), size, stream));
  }
}

template <typename Allocator>
void BounceBufferRing<Allocator>::enqueue_d2h(void* device_src, std::size_t size, CUstream stream)
{
  KVIKIO_NVTX_FUNC_RANGE();
  if (_batch_copy) {
    _batch_transfer_ctx.add_entry(cur_buffer(), device_src, size);
  } else {
    CUDA_DRIVER_TRY(cudaAPI::instance().MemcpyDtoHAsync(
      cur_buffer(), convert_void2deviceptr(device_src), size, stream));
  }
}

template <typename Allocator>
void BounceBufferRing<Allocator>::advance(CUstream stream)
{
  KVIKIO_NVTX_FUNC_RANGE();
  _cur_buffer_offset = 0;
  ++_cur_buf_idx;
  if (_cur_buf_idx >= _num_buffers) { _cur_buf_idx = 0; }
  if (_cur_buf_idx == _initial_buf_idx) {
    if (_batch_copy) {
      batch_copy(
        _batch_transfer_ctx.dsts, _batch_transfer_ctx.srcs, _batch_transfer_ctx.sizes, stream);
      _batch_transfer_ctx.clear();
    }
    CUDA_DRIVER_TRY(cudaAPI::instance().StreamSynchronize(stream));
    _initial_buf_idx = _cur_buf_idx;
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
  return _num_buffers;
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
void BounceBufferRing<Allocator>::accumulate_and_submit_h2d(void* device_dst,
                                                            void const* host_src,
                                                            std::size_t size,
                                                            CUstream stream)
{
  KVIKIO_NVTX_FUNC_RANGE();
  auto const* host_src_ptr = static_cast<char const*>(host_src);
  auto* device_dst_ptr     = static_cast<char*>(device_dst);

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
}

template <typename Allocator>
void BounceBufferRing<Allocator>::submit_h2d(void* device_dst, std::size_t size, CUstream stream)
{
  KVIKIO_NVTX_FUNC_RANGE();
  enqueue_h2d(device_dst, size, stream);
  advance(stream);
}

template <typename Allocator>
std::size_t BounceBufferRing<Allocator>::flush_h2d(void* device_dst, CUstream stream)
{
  KVIKIO_NVTX_FUNC_RANGE();
  if (_cur_buffer_offset == 0) { return 0; }
  auto const flushed = _cur_buffer_offset;
  enqueue_h2d(device_dst, _cur_buffer_offset, stream);
  advance(stream);
  return flushed;
}

template <typename Allocator>
void BounceBufferRing<Allocator>::synchronize(CUstream stream)
{
  KVIKIO_NVTX_FUNC_RANGE();
  if (_batch_copy) {
    batch_copy(
      _batch_transfer_ctx.dsts, _batch_transfer_ctx.srcs, _batch_transfer_ctx.sizes, stream);
    _batch_transfer_ctx.clear();
  }
  CUDA_DRIVER_TRY(cudaAPI::instance().StreamSynchronize(stream));
}

// Explicit instantiations
template class BounceBufferRing<CudaPinnedAllocator>;
template class BounceBufferRing<CudaPageAlignedPinnedAllocator>;

}  // namespace kvikio
