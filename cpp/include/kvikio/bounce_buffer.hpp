/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <stack>

#include <kvikio/defaults.hpp>

namespace kvikio {

class PageAlignedAllocator {
 public:
  void* allocate(std::size_t size);
  void deallocate(void* buffer, std::size_t size);
};

class CudaPinnedAllocator {
 public:
  void* allocate(std::size_t size);
  void deallocate(void* buffer, std::size_t size);
};

class CudaPageAlignedPinnedAllocator {
 public:
  void* allocate(std::size_t size);
  void deallocate(void* buffer, std::size_t size);
};

/**
 * @brief Singleton class to retain host memory allocations
 *
 * Call `BounceBufferPool::get` to get an allocation that will be retained when it
 * goes out of scope (RAII). The size of all retained allocations are the same.
 */
template <typename Allocator = CudaPinnedAllocator>
class BounceBufferPool {
 private:
  std::mutex _mutex{};
  // Stack of free allocations
  std::stack<void*> _free_buffers{};
  // The size of each allocation in `_free_buffers`
  std::size_t _size{defaults::bounce_buffer_size()};
  Allocator _allocator{};

 public:
  /**
   * @brief An host memory allocation
   */
  class Buffer {
   private:
    BounceBufferPool* _manager;
    void* _buffer;
    std::size_t const _size;

   public:
    Buffer(BounceBufferPool<Allocator>* manager, void* buffer, std::size_t size);
    Buffer(Buffer const&)            = delete;
    Buffer& operator=(Buffer const&) = delete;
    Buffer(Buffer&& o)               = delete;
    Buffer& operator=(Buffer&& o)    = delete;
    ~Buffer() noexcept;
    void* get() noexcept;
    void* get(std::ptrdiff_t offset) noexcept;
    std::size_t size() noexcept;
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
   * NB: `_mutex` must be taken prior to calling this function.
   */
  void _ensure_buffer_size();

 public:
  [[nodiscard]] Buffer get();

  void put(void* buffer, std::size_t size);

  /**
   * @brief Free all retained allocations
   *
   * @return The number of bytes cleared
   */
  std::size_t clear();

  KVIKIO_EXPORT static BounceBufferPool& instance();

  BounceBufferPool(BounceBufferPool const&)            = delete;
  BounceBufferPool& operator=(BounceBufferPool const&) = delete;
  BounceBufferPool(BounceBufferPool&& o)               = delete;
  BounceBufferPool& operator=(BounceBufferPool&& o)    = delete;
};

using PageAlignedBounceBufferPool           = BounceBufferPool<PageAlignedAllocator>;
using CudaPinnedBounceBufferPool            = BounceBufferPool<CudaPinnedAllocator>;
using CudaPageAlignedPinnedBounceBufferPool = BounceBufferPool<CudaPageAlignedPinnedAllocator>;
}  // namespace kvikio
