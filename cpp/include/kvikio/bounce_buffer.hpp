/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

#include <stack>

#include <kvikio/defaults.hpp>

namespace kvikio {

/**
 * @brief Singleton class to retain host memory allocations
 *
 * Call `AllocRetain::get` to get an allocation that will be retained when it
 * goes out of scope (RAII). The size of all retained allocations are the same.
 */
class AllocRetain {
 private:
  std::mutex _mutex{};
  // Stack of free allocations
  std::stack<void*> _free_allocs{};
  // The size of each allocation in `_free_allocs`
  std::size_t _size{defaults::bounce_buffer_size()};

 public:
  /**
   * @brief An host memory allocation
   */
  class Alloc {
   private:
    AllocRetain* _manager;
    void* _alloc;
    std::size_t const _size;

   public:
    Alloc(AllocRetain* manager, void* alloc, std::size_t size);
    Alloc(Alloc const&)            = delete;
    Alloc& operator=(Alloc const&) = delete;
    Alloc(Alloc&& o)               = delete;
    Alloc& operator=(Alloc&& o)    = delete;
    ~Alloc() noexcept;
    void* get() noexcept;
    void* get(std::ptrdiff_t offset) noexcept;
    std::size_t size() noexcept;
  };

  AllocRetain() = default;

  // Notice, we do not clear the allocations at destruction thus the allocations leaks
  // at exit. We do this because `AllocRetain::instance()` stores the allocations in a
  // static stack that are destructed below main, which is not allowed in CUDA:
  // <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#initialization>
  ~AllocRetain() noexcept = default;

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
   * NB: `_mutex` must be taken prior to calling this function.
   */
  void _ensure_alloc_size();

 public:
  [[nodiscard]] Alloc get();

  void put(void* alloc, std::size_t size);

  /**
   * @brief Free all retained allocations
   *
   * @return The number of bytes cleared
   */
  std::size_t clear();

  KVIKIO_EXPORT static AllocRetain& instance();

  AllocRetain(AllocRetain const&)            = delete;
  AllocRetain& operator=(AllocRetain const&) = delete;
  AllocRetain(AllocRetain&& o)               = delete;
  AllocRetain& operator=(AllocRetain&& o)    = delete;
};

}  // namespace kvikio
