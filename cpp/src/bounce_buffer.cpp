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

#include <kvikio/bounce_buffer.hpp>
#include <kvikio/defaults.hpp>
#include <kvikio/detail/nvtx.hpp>
#include <kvikio/error.hpp>
#include <kvikio/shim/cuda.hpp>

namespace kvikio {

AllocRetain::Alloc::Alloc(AllocRetain* manager, void* alloc, std::size_t size)
  : _manager(manager), _alloc{alloc}, _size{size}
{
  KVIKIO_NVTX_FUNC_RANGE();
}

AllocRetain::Alloc::~Alloc() noexcept
{
  KVIKIO_NVTX_FUNC_RANGE();
  _manager->put(_alloc, _size);
}

void* AllocRetain::Alloc::get() noexcept
{
  KVIKIO_NVTX_FUNC_RANGE();
  return _alloc;
}

void* AllocRetain::Alloc::get(std::ptrdiff_t offset) noexcept
{
  KVIKIO_NVTX_FUNC_RANGE();
  return static_cast<char*>(_alloc) + offset;
}

std::size_t AllocRetain::Alloc::size() noexcept { return _size; }

std::size_t AllocRetain::_clear()
{
  KVIKIO_NVTX_FUNC_RANGE();
  std::size_t ret = _free_allocs.size() * _size;
  while (!_free_allocs.empty()) {
    CUDA_DRIVER_TRY(cudaAPI::instance().MemFreeHost(_free_allocs.top()));
    _free_allocs.pop();
  }
  return ret;
}

void AllocRetain::_ensure_alloc_size()
{
  KVIKIO_NVTX_FUNC_RANGE();
  auto const bounce_buffer_size = defaults::bounce_buffer_size();
  if (_size != bounce_buffer_size) {
    _clear();
    _size = bounce_buffer_size;
  }
}

AllocRetain::Alloc AllocRetain::get()
{
  KVIKIO_NVTX_FUNC_RANGE();
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
  KVIKIO_NVTX_FUNC_RANGE();
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
  KVIKIO_NVTX_FUNC_RANGE();
  std::lock_guard const lock(_mutex);
  return _clear();
}

AllocRetain& AllocRetain::instance()
{
  KVIKIO_NVTX_FUNC_RANGE();
  static AllocRetain _instance;
  return _instance;
}

}  // namespace kvikio
