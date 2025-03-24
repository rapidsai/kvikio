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

#include <mutex>
#include <stack>
#include <vector>

#include <kvikio/shim/utils.hpp>

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
  std::size_t _size{};  // Decouple this class from the defaults singleton.

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

  AllocRetain();

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

class BlockView;

class Block {
 private:
  std::byte* _buffer{nullptr};
  std::size_t _bytes{0u};

 public:
  Block()                        = default;
  ~Block()                       = default;
  Block(Block const&)            = delete;
  Block& operator=(Block const&) = delete;
  Block(Block&&)                 = default;
  Block& operator=(Block&&)      = default;

  void allocate(std::size_t bytes);
  void deallocate();

  BlockView make_view(std::size_t start_byte_idx, std::size_t bytes);
  std::size_t size() const noexcept;
  std::byte* data() const noexcept;
};

class BlockView {
 private:
  std::byte* _buffer{nullptr};
  std::size_t _bytes{0u};

 public:
  BlockView(std::byte* buffer, std::size_t bytes);
  BlockView(BlockView const&)            = default;
  BlockView& operator=(BlockView const&) = default;
  BlockView(BlockView&&)                 = default;
  BlockView& operator=(BlockView&&)      = default;

  std::size_t size() const noexcept;
  std::byte* data() const noexcept;
};

class BounceBuffer {
 private:
  BounceBuffer() = default;

  std::size_t _requested_bytes_per_block{1024u * 1024u * 16u};
  std::size_t _num_blocks{4u};

  inline static Block block_pool;
  std::vector<BlockView> _blockviews_pool;

  Block _block;
  std::vector<BlockView> _blockviews;

 public:
  static BounceBuffer& instance();

  static void preinitialize_for_pool(unsigned int num_threads,
                                     std::size_t requested_bytes_per_block,
                                     std::size_t num_blocks);
  void initialize_per_thread(std::size_t requested_bytes_per_block, std::size_t num_blocks);

  BlockView get();

  BounceBuffer(BounceBuffer const&)            = delete;
  BounceBuffer& operator=(BounceBuffer const&) = delete;
  BounceBuffer(BounceBuffer&&)                 = delete;
  BounceBuffer& operator=(BounceBuffer&&)      = delete;
};

}  // namespace kvikio
