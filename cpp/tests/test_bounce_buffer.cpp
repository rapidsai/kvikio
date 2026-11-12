/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstddef>
#include <cstdint>
#include <cstring>

#include <gtest/gtest.h>

#include <kvikio/bounce_buffer.hpp>
#include <kvikio/defaults.hpp>
#include <kvikio/error.hpp>
#include <kvikio/utils.hpp>

#include "utils/utils.hpp"

class BounceBufferTest : public testing::Test {
 protected:
  void SetUp() override { KVIKIO_CHECK_CUDA(cudaSetDevice(0)); }
  void TearDown() override {}
};

TEST_F(BounceBufferTest, buffers_returned_to_pool)
{
  auto& pool = kvikio::CudaPinnedBounceBufferPool::instance();
  pool.clear();

  EXPECT_EQ(pool.num_free_buffers(), 0);

  // Buffers created
  {
    auto buf1 = pool.get();
    auto buf2 = pool.get();
    // All created buffers are in use
    EXPECT_EQ(pool.num_free_buffers(), 0);
  }  // buf1 and buf2 returned to the pool
  EXPECT_EQ(pool.num_free_buffers(), 2);

  // Buffers reused
  {
    auto buf1 = pool.get();
    auto buf2 = pool.get();
    EXPECT_EQ(pool.num_free_buffers(), 0);
  }  // buf1 and buf2 returned to the pool
  EXPECT_EQ(pool.num_free_buffers(), 2);
}

TEST_F(BounceBufferTest, move_construction_and_move_assignment)
{
  auto& pool = kvikio::CudaPinnedBounceBufferPool::instance();
  pool.clear();

  {
    auto buf1 = pool.get();

    // Move constructor that transfers the ownership
    auto buf2 = std::move(buf1);
  }

  // Only one return, not two
  EXPECT_EQ(pool.num_free_buffers(), 1);

  {
    // Buffer reused
    auto buf1 = pool.get();

    // Manually create a new buffer outside the pool
    kvikio::CudaPinnedAllocator allocator;
    auto* buffer = allocator.allocate(pool.buffer_size());
    kvikio::CudaPinnedBounceBufferPool::Buffer buf2(&pool, buffer, pool.buffer_size());
    // Move assignment that adds the previous buffer to the pool and then transfers the ownership
    buf2 = std::move(buf1);
  }

  EXPECT_EQ(pool.num_free_buffers(), 2);
}

TEST_F(BounceBufferTest, buffer_size_changes_clears_pool)
{
  auto& pool = kvikio::CudaPinnedBounceBufferPool::instance();
  pool.clear();

  auto original_size = kvikio::defaults::bounce_buffer_size();

  // Populate pool with buffers at current size
  {
    auto buf1 = pool.get();
    auto buf2 = pool.get();
  }
  EXPECT_EQ(pool.num_free_buffers(), 2);
  EXPECT_EQ(pool.buffer_size(), original_size);

  // Change buffer size
  auto new_size{original_size * 2};
  kvikio::defaults::set_bounce_buffer_size(new_size);

  // Next get() triggers _ensure_buffer_size(), clearing old buffers
  {
    auto buf = pool.get();
    EXPECT_EQ(buf.size(), new_size);
  }
  EXPECT_EQ(pool.num_free_buffers(), 1);  // Only the new buffer
  EXPECT_EQ(pool.buffer_size(), new_size);

  kvikio::defaults::set_bounce_buffer_size(original_size);
}

TEST_F(BounceBufferTest, old_size_buffer_deallocated_not_returned)
{
  auto& pool = kvikio::CudaPinnedBounceBufferPool::instance();
  pool.clear();

  auto original_size = kvikio::defaults::bounce_buffer_size();

  {
    auto buf = pool.get();  // Buffer at original size

    // Change size while buffer is outstanding
    kvikio::defaults::set_bounce_buffer_size(original_size * 2);
  }  // buf destructor will call put() with mismatched size

  // Old buffer should have been deallocated, not returned to pool
  EXPECT_EQ(pool.num_free_buffers(), 0);

  kvikio::defaults::set_bounce_buffer_size(original_size);
}

TEST_F(BounceBufferTest, pinned_allocation_with_any_prefault_thread_count)
{
  auto const original_nthreads = kvikio::defaults::bounce_buffer_prefault_nthreads();
  std::size_t const max_size   = 37 * 1024 * 1024;
  void* dev{};
  KVIKIO_CHECK_CUDA(cudaMalloc(&dev, max_size));

  for (unsigned int nthreads : {0U, 1U, 8U}) {
    kvikio::defaults::set_bounce_buffer_prefault_nthreads(nthreads);
    for (std::size_t size :
         {std::size_t{1}, std::size_t{4097}, std::size_t{3 * 1024 * 1024 + 1}, max_size}) {
      kvikio::CudaPageAlignedPinnedAllocator allocator;
      auto* buffer = static_cast<char*>(allocator.allocate(size));
      EXPECT_EQ(reinterpret_cast<std::uintptr_t>(buffer) % kvikio::get_page_size(), 0);

      // Pinned, and usable for a round trip through the device
      unsigned int flags{};
      KVIKIO_CHECK_CUDA(cudaHostGetFlags(&flags, buffer));
      std::memset(buffer, 0x5a, size);
      KVIKIO_CHECK_CUDA(cudaMemcpy(dev, buffer, size, cudaMemcpyHostToDevice));
      std::memset(buffer, 0, size);
      KVIKIO_CHECK_CUDA(cudaMemcpy(buffer, dev, size, cudaMemcpyDeviceToHost));
      EXPECT_EQ(buffer[0], 0x5a);
      EXPECT_EQ(buffer[size - 1], 0x5a);

      allocator.deallocate(buffer, size);
    }
  }

  kvikio::defaults::set_bounce_buffer_prefault_nthreads(original_nthreads);
  KVIKIO_CHECK_CUDA(cudaFree(dev));
}
