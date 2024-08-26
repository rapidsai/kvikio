/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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
#pragma once

#include <mutex>
#include <stack>

namespace kvikio {

inline constexpr std::size_t _posix_bounce_buffer_size = 2 << 23;  // 16 MiB

/**
 * @brief Singleton class to retain host memory allocations
 *
 * Call `AllocRetain::get` to get an allocation that will be retained when it
 * goes out of scope (RAII). The size of all allocations are `posix_bounce_buffer_size`.
 */
class AllocRetain {
 private:
  std::stack<void*> _free_allocs;
  std::mutex _mutex;

 public:
  /**
   * @brief An host memory allocation
   */
  class Alloc {
   private:
    AllocRetain* _manager;
    void* _alloc;
    std::size_t _size;

   public:
    Alloc(AllocRetain* manager, void* alloc)
      : _manager(manager), _alloc{alloc}, _size{_posix_bounce_buffer_size}
    {
    }
    Alloc(const Alloc&)            = delete;
    Alloc& operator=(Alloc const&) = delete;
    Alloc(Alloc&& o)               = delete;
    Alloc& operator=(Alloc&& o)    = delete;
    ~Alloc() noexcept { _manager->put(_alloc); }
    void* get() noexcept { return _alloc; }
    std::size_t size() noexcept { return _size; }
  };

  AllocRetain() = default;
  [[nodiscard]] Alloc get()
  {
    const std::lock_guard lock(_mutex);
    // Check if we have an allocation available
    if (!_free_allocs.empty()) {
      void* ret = _free_allocs.top();
      _free_allocs.pop();
      return Alloc(this, ret);
    }

    // If no available allocation, allocate and register a new one
    void* alloc{};
    // Allocate page-locked host memory
    CUDA_DRIVER_TRY(cudaAPI::instance().MemHostAlloc(
      &alloc, _posix_bounce_buffer_size, CU_MEMHOSTREGISTER_PORTABLE));
    return Alloc(this, alloc);
  }

  void put(void* alloc)
  {
    const std::lock_guard lock(_mutex);
    _free_allocs.push(alloc);
  }

  void clear()
  {
    const std::lock_guard lock(_mutex);
    while (!_free_allocs.empty()) {
      CUDA_DRIVER_TRY(cudaAPI::instance().MemFreeHost(_free_allocs.top()));
      _free_allocs.pop();
    }
  }

  static AllocRetain& instance()
  {
    static AllocRetain _instance;
    return _instance;
  }

  AllocRetain(const AllocRetain&)            = delete;
  AllocRetain& operator=(AllocRetain const&) = delete;
  AllocRetain(AllocRetain&& o)               = delete;
  AllocRetain& operator=(AllocRetain&& o)    = delete;
  ~AllocRetain() noexcept                    = default;
};

}  // namespace kvikio
