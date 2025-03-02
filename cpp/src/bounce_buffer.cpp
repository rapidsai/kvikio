/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <mutex>
#include <stack>
#include <stdexcept>

#include <kvikio/bounce_buffer.hpp>
#include <kvikio/defaults.hpp>
#include <kvikio/error.hpp>
#include <kvikio/shim/cuda.hpp>
#include <kvikio/threadpool_wrapper.hpp>
#include <kvikio/utils.hpp>

namespace kvikio {

AllocRetain::AllocRetain() : _size{defaults::bounce_buffer_size()} {}

AllocRetain::Alloc::Alloc(AllocRetain* manager, void* alloc, std::size_t size)
  : _manager(manager), _alloc{alloc}, _size{size}
{
}

AllocRetain::Alloc::~Alloc() noexcept { _manager->put(_alloc, _size); }

void* AllocRetain::Alloc::get() noexcept { return _alloc; }

void* AllocRetain::Alloc::get(std::ptrdiff_t offset) noexcept
{
  return static_cast<char*>(_alloc) + offset;
}

std::size_t AllocRetain::Alloc::size() noexcept { return _size; }

std::size_t AllocRetain::_clear()
{
  std::size_t ret = _free_allocs.size() * _size;
  while (!_free_allocs.empty()) {
    CUDA_DRIVER_TRY(cudaAPI::instance().MemFreeHost(_free_allocs.top()));
    _free_allocs.pop();
  }
  return ret;
}

void AllocRetain::_ensure_alloc_size()
{
  auto const bounce_buffer_size = defaults::bounce_buffer_size();
  if (_size != bounce_buffer_size) {
    _clear();
    _size = bounce_buffer_size;
  }
}

AllocRetain::Alloc AllocRetain::get()
{
  std::lock_guard const lock(_mutex);
  _ensure_alloc_size();

  // Check if we have an allocation available
  if (!_free_allocs.empty()) {
    void* ret = _free_allocs.top();
    _free_allocs.pop();
    return Alloc(this, ret, _size);
  }

  // If no available allocation, allocate and register a new one
  void* alloc{};
  // Allocate page-locked host memory
  // Under unified addressing, host memory allocated this way is automatically portable and mapped.
  CUDA_DRIVER_TRY(cudaAPI::instance().MemHostAlloc(&alloc, _size, CU_MEMHOSTALLOC_PORTABLE));
  return Alloc(this, alloc, _size);
}

void AllocRetain::put(void* alloc, std::size_t size)
{
  std::lock_guard const lock(_mutex);
  _ensure_alloc_size();

  // If the size of `alloc` matches the sizes of the retained allocations,
  // it is added to the set of free allocation otherwise it is freed.
  if (size == _size) {
    _free_allocs.push(alloc);
  } else {
    CUDA_DRIVER_TRY(cudaAPI::instance().MemFreeHost(alloc));
  }
}

std::size_t AllocRetain::clear()
{
  std::lock_guard const lock(_mutex);
  return _clear();
}

AllocRetain& AllocRetain::instance()
{
  static AllocRetain _instance;
  return _instance;
}

void Block::allocate(std::size_t bytes)
{
  _bytes = bytes;
  CUDA_DRIVER_TRY(cudaAPI::instance().MemHostAlloc(
    reinterpret_cast<void**>(&_buffer), _bytes, CU_MEMHOSTALLOC_PORTABLE));
}

void Block::deallocate()
{
  CUDA_DRIVER_TRY(cudaAPI::instance().MemFreeHost(_buffer));
  _bytes  = 0u;
  _buffer = nullptr;
}

BlockView Block::make_view(std::size_t start_byte_idx, std::size_t bytes)
{
  KVIKIO_EXPECT(start_byte_idx + bytes <= _bytes, "Block view out of bound.", std::runtime_error);
  return BlockView(_buffer + start_byte_idx, bytes);
}

std::size_t Block::size() const noexcept { return _bytes; }

std::byte* Block::data() const noexcept { return _buffer; }

BlockView::BlockView(std::byte* buffer, std::size_t bytes) : _buffer(buffer), _bytes(bytes) {}

std::size_t BlockView::size() const noexcept { return _bytes; }

std::byte* BlockView::data() const noexcept { return _buffer; }

BounceBuffer& BounceBuffer::instance()
{
  thread_local BounceBuffer _instance;
  return _instance;
}

BlockView BounceBuffer::get()
{
  thread_local std::size_t current_idx{0};
  if (this_thread::is_from_pool<BS_thread_pool>()) {
    if (current_idx >= _blockviews_pool.size()) { current_idx -= _blockviews_pool.size(); }
    return _blockviews_pool[current_idx++];
  } else {
    if (_blockviews.size() == 0) {
      initialize_per_thread(defaults::bounce_buffer_size(), defaults::num_subtasks_per_task());
    }
    if (current_idx >= _blockviews.size()) { current_idx -= _blockviews.size(); }
    return _blockviews[current_idx++];
  }
}

void BounceBuffer::preinitialize_for_pool(unsigned int num_threads,
                                          std::size_t requested_bytes_per_block,
                                          std::size_t num_blocks)
{
  // Round up to the multiples of page size
  std::size_t bytes_per_block = (requested_bytes_per_block + page_size - 1) & (~page_size + 1);
  auto total_bytes            = bytes_per_block * num_blocks * num_threads;

  block_pool.deallocate();
  block_pool.allocate(total_bytes);
}

void BounceBuffer::initialize_per_thread(std::size_t requested_bytes_per_block,
                                         std::size_t num_blocks)
{
  _requested_bytes_per_block = requested_bytes_per_block;
  _num_blocks                = num_blocks;

  // Round up to the multiples of page size
  std::size_t bytes_per_block = (_requested_bytes_per_block + page_size - 1) & (~page_size + 1);
  auto bytes_per_thread       = bytes_per_block * _num_blocks;

  if (this_thread::is_from_pool<BS_thread_pool>()) {
    _blockviews_pool.clear();
    std::size_t my_offset = this_thread::index<BS_thread_pool>().value() * bytes_per_thread;
    for (std::size_t i = 0; i < _num_blocks; ++i) {
      _blockviews_pool.emplace_back(
        block_pool.make_view(my_offset + bytes_per_block * i, bytes_per_block));
    }
  } else {
    _blockviews.clear();
    _block.deallocate();
    _block.allocate(bytes_per_thread);
    for (std::size_t i = 0; i < _num_blocks; ++i) {
      _blockviews.emplace_back(_block.make_view(bytes_per_block * i, bytes_per_block));
    }
  }
}

}  // namespace kvikio
