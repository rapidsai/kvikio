/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <atomic>
#include <chrono>
#include <thread>
#include <tuple>
#include <vector>

#include <cuda.h>
#include <gtest/gtest.h>

#include <kvikio/bounce_buffer.hpp>
#include <kvikio/defaults.hpp>
#include <kvikio/detail/bounce_buffer_cache.hpp>
#include <kvikio/error.hpp>
#include <kvikio/shim/cuda.hpp>

#include "utils/utils.hpp"

namespace {

CUcontext current_context()
{
  CUcontext ctx{};
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().CtxGetCurrent(&ctx));
  return ctx;
}

// Spin briefly waiting for `pred()` to become true, up to `timeout`. Used to give pending
// cuLaunchHostFunc recycle callbacks time to fire on the CUDA driver thread before assertions.
template <typename Pred>
bool wait_for(Pred pred, std::chrono::milliseconds timeout = std::chrono::seconds(2))
{
  auto const deadline = std::chrono::steady_clock::now() + timeout;
  while (!pred()) {
    if (std::chrono::steady_clock::now() >= deadline) { return false; }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return true;
}

}  // namespace

class BounceBufferCacheTest : public testing::Test {
 protected:
  void SetUp() override { KVIKIO_CHECK_CUDA(cudaSetDevice(0)); }
};

TEST_F(BounceBufferCacheTest, try_get_returns_buffer_under_cap)
{
  kvikio::detail::BounceBufferCachePerThreadAndContext<kvikio::CudaPinnedAllocator> cache(4);
  EXPECT_EQ(cache.cap(), 4u);

  auto ctx = current_context();
  auto b   = cache.try_get(ctx);
  ASSERT_TRUE(b.has_value());
  EXPECT_NE(b->get(), nullptr);
  EXPECT_EQ(b->size(), kvikio::defaults::bounce_buffer_size());
}

TEST_F(BounceBufferCacheTest, try_get_returns_nullopt_at_cap)
{
  kvikio::detail::BounceBufferCachePerThreadAndContext<kvikio::CudaPinnedAllocator> cache(2);

  auto ctx = current_context();
  auto b1  = cache.try_get(ctx);
  auto b2  = cache.try_get(ctx);
  ASSERT_TRUE(b1.has_value());
  ASSERT_TRUE(b2.has_value());

  // Cap of 2 is reached: third try_get must fail.
  auto b3 = cache.try_get(ctx);
  EXPECT_FALSE(b3.has_value());
}

TEST_F(BounceBufferCacheTest, cap_zero_means_unlimited)
{
  kvikio::detail::BounceBufferCachePerThreadAndContext<kvikio::CudaPinnedAllocator> cache(0);
  EXPECT_EQ(cache.cap(), 0u);

  auto ctx = current_context();
  std::vector<decltype(cache.try_get(ctx))> bufs;
  // Acquire more than any reasonable cap; cap=0 means no cap.
  for (int i = 0; i < 32; ++i) {
    auto b = cache.try_get(ctx);
    ASSERT_TRUE(b.has_value()) << "iteration " << i;
    bufs.push_back(std::move(b));
  }
}

TEST_F(BounceBufferCacheTest, recycle_now_returns_buffer_to_free_list)
{
  kvikio::detail::BounceBufferCachePerThreadAndContext<kvikio::CudaPinnedAllocator> cache(2);
  auto ctx = current_context();

  void* first_ptr = nullptr;
  {
    auto b = cache.try_get(ctx);
    ASSERT_TRUE(b.has_value());
    first_ptr = b->get();
    cache.recycle_now(ctx, std::move(*b));
  }

  // The recycled buffer should come back as the next try_get (LIFO via the free list).
  auto b2 = cache.try_get(ctx);
  ASSERT_TRUE(b2.has_value());
  EXPECT_EQ(b2->get(), first_ptr);
}

TEST_F(BounceBufferCacheTest, recycle_after_round_trip)
{
  kvikio::detail::BounceBufferCachePerThreadAndContext<kvikio::CudaPinnedAllocator> cache(2);
  auto ctx = current_context();

  CUstream stream{};
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamCreate(&stream, CU_STREAM_DEFAULT));

  void* first_ptr = nullptr;
  {
    auto b = cache.try_get(ctx);
    ASSERT_TRUE(b.has_value());
    first_ptr = b->get();
    // No actual H2D needed for the recycle callback to fire; cuLaunchHostFunc is itself
    // enough stream work to schedule on. Sync ensures the callback has been observed by us.
    cache.recycle_after(ctx, std::move(*b), stream);
  }

  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamSynchronize(stream));

  // After sync, the callback has fired. Give it a brief moment to land on the free list in
  // case the CUDA driver thread that ran the callback is just behind us, then verify reuse.
  ASSERT_TRUE(wait_for([&] {
    auto b = cache.try_get(ctx);
    if (b.has_value() && b->get() == first_ptr) {
      cache.recycle_now(ctx, std::move(*b));
      return true;
    }
    if (b.has_value()) { cache.recycle_now(ctx, std::move(*b)); }
    return false;
  }));

  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamDestroy(stream));
}

TEST_F(BounceBufferCacheTest, recycle_after_releases_in_flight_slot)
{
  kvikio::detail::BounceBufferCachePerThreadAndContext<kvikio::CudaPinnedAllocator> cache(2);
  auto ctx = current_context();

  CUstream stream{};
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamCreate(&stream, CU_STREAM_DEFAULT));

  // Fill the cap, schedule both to be recycled, then verify try_get eventually succeeds
  // after the callbacks fire.
  {
    auto b1 = cache.try_get(ctx);
    auto b2 = cache.try_get(ctx);
    ASSERT_TRUE(b1.has_value());
    ASSERT_TRUE(b2.has_value());
    EXPECT_FALSE(cache.try_get(ctx).has_value());  // at cap
    cache.recycle_after(ctx, std::move(*b1), stream);
    cache.recycle_after(ctx, std::move(*b2), stream);
  }

  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamSynchronize(stream));

  // After both callbacks have fired, the cache is back to fully free.
  ASSERT_TRUE(wait_for([&] {
    auto b = cache.try_get(ctx);
    if (b.has_value()) {
      cache.recycle_now(ctx, std::move(*b));
      return true;
    }
    return false;
  }));

  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamDestroy(stream));
}

TEST_F(BounceBufferCacheTest, multi_context_isolation)
{
  kvikio::detail::BounceBufferCachePerThreadAndContext<kvikio::CudaPinnedAllocator> cache(2);

  auto primary_ctx = current_context();
  ASSERT_NE(primary_ctx, nullptr);

  // Fill the primary-context cap.
  auto b1 = cache.try_get(primary_ctx);
  auto b2 = cache.try_get(primary_ctx);
  ASSERT_TRUE(b1.has_value());
  ASSERT_TRUE(b2.has_value());
  EXPECT_FALSE(cache.try_get(primary_ctx).has_value());

  // Create a second context on the same device. The per-key cap is per
  // (thread, ctx), so the second context has its own independent budget.
  CUdevice dev{};
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().CtxGetDevice(&dev));
  CUcontext second_ctx{};
#if CUDA_VERSION >= 13000
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().CtxCreate(&second_ctx, nullptr, 0, dev));
#else
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().CtxCreate(&second_ctx, 0, dev));
#endif
  ASSERT_EQ(current_context(), second_ctx);

  auto s1 = cache.try_get(second_ctx);
  auto s2 = cache.try_get(second_ctx);
  ASSERT_TRUE(s1.has_value());
  ASSERT_TRUE(s2.has_value());
  EXPECT_FALSE(cache.try_get(second_ctx).has_value());

  cache.recycle_now(second_ctx, std::move(*s1));
  cache.recycle_now(second_ctx, std::move(*s2));

  // Restore the primary context and clean up.
  CUcontext popped{};
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().CtxPopCurrent(&popped));
  ASSERT_EQ(popped, second_ctx);
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().CtxDestroy(second_ctx));
  ASSERT_EQ(current_context(), primary_ctx);

  cache.recycle_now(primary_ctx, std::move(*b1));
  cache.recycle_now(primary_ctx, std::move(*b2));
}

TEST_F(BounceBufferCacheTest, per_thread_isolation)
{
  kvikio::detail::BounceBufferCachePerThreadAndContext<kvikio::CudaPinnedAllocator> cache(1);
  auto ctx = current_context();

  // Main thread occupies its key's single slot.
  auto b1 = cache.try_get(ctx);
  ASSERT_TRUE(b1.has_value());
  EXPECT_FALSE(cache.try_get(ctx).has_value());

  // A worker thread has an independent (thread, ctx) key and should not be blocked.
  std::atomic<bool> worker_succeeded{false};
  std::thread worker([&] {
    KVIKIO_CHECK_CUDA(cudaSetDevice(0));
    auto wb = cache.try_get(ctx);
    if (wb.has_value()) {
      worker_succeeded = true;
      cache.recycle_now(ctx, std::move(*wb));
    }
  });
  worker.join();
  EXPECT_TRUE(worker_succeeded.load());

  cache.recycle_now(ctx, std::move(*b1));
}

TEST_F(BounceBufferCacheTest, concurrent_get_and_recycle_now)
{
  kvikio::detail::BounceBufferCachePerThreadAndContext<kvikio::CudaPinnedAllocator> cache(
    0);  // unlimited

  constexpr int num_threads           = 8;
  constexpr int iterations_per_thread = 64;
  std::atomic<int> errors{0};
  std::vector<std::thread> workers;
  workers.reserve(num_threads);

  for (int t = 0; t < num_threads; ++t) {
    workers.emplace_back([&] {
      try {
        KVIKIO_CHECK_CUDA(cudaSetDevice(0));
        auto ctx = current_context();
        for (int i = 0; i < iterations_per_thread; ++i) {
          auto b = cache.try_get(ctx);
          if (!b.has_value()) {
            ++errors;
            continue;
          }
          cache.recycle_now(ctx, std::move(*b));
        }
      } catch (...) {
        ++errors;
      }
    });
  }
  for (auto& w : workers) {
    w.join();
  }
  EXPECT_EQ(errors.load(), 0);
}

TEST_F(BounceBufferCacheTest, singleton_instance_has_default_cap)
{
  auto& s =
    kvikio::detail::BounceBufferCachePerThreadAndContext<kvikio::CudaPinnedAllocator>::instance();
  EXPECT_EQ(s.cap(), kvikio::defaults::num_bounce_buffers_per_cache());

  // try_get on the singleton works.
  auto b = s.try_get(current_context());
  ASSERT_TRUE(b.has_value());
  EXPECT_NE(b->get(), nullptr);
  s.recycle_now(current_context(), std::move(*b));
}
