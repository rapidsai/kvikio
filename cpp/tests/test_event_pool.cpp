/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <atomic>
#include <thread>
#include <tuple>
#include <vector>

#include <cuda.h>
#include <gtest/gtest.h>

#include <kvikio/detail/event.hpp>
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

}  // namespace

class EventPoolTest : public testing::Test {
 protected:
  void SetUp() override
  {
    // Ensures a primary context is created and current on the calling thread.
    KVIKIO_CHECK_CUDA(cudaSetDevice(0));
  }
};

TEST_F(EventPoolTest, get_returns_valid_event)
{
  auto& pool = kvikio::detail::EventPool::instance();

  auto e = pool.get();
  EXPECT_NE(e.get(), nullptr);
  EXPECT_EQ(e.cuda_context(), current_context());
}

TEST_F(EventPoolTest, raii_drop_returns_event_for_lifo_reuse)
{
  auto& pool = kvikio::detail::EventPool::instance();

  CUevent first_handle{};
  {
    auto e       = pool.get();
    first_handle = e.get();
  }  // drop returns the event to the pool

  // The next get from the same context should pull the same handle (LIFO).
  auto e2 = pool.get();
  EXPECT_EQ(e2.get(), first_handle);
}

TEST_F(EventPoolTest, move_construction_transfers_ownership)
{
  auto& pool = kvikio::detail::EventPool::instance();

  CUevent moved_handle{};
  {
    auto src     = pool.get();
    moved_handle = src.get();

    auto dst = std::move(src);
    EXPECT_EQ(dst.get(), moved_handle);
    EXPECT_EQ(src.get(), nullptr);
    EXPECT_EQ(src.cuda_context(), nullptr);
  }  // Only `dst`'s destructor returns the event; moved-from `src` is a no-op.

  // The event should be in the pool now and reusable.
  auto e = pool.get();
  EXPECT_EQ(e.get(), moved_handle);
}

TEST_F(EventPoolTest, move_assignment_returns_target_event)
{
  auto& pool = kvikio::detail::EventPool::instance();

  // Acquire two distinct events.
  auto src        = pool.get();
  auto dst        = pool.get();
  auto src_handle = src.get();
  auto dst_handle = dst.get();
  EXPECT_NE(src_handle, dst_handle);

  dst = std::move(src);

  // dst now owns src's event; src is empty.
  EXPECT_EQ(dst.get(), src_handle);
  EXPECT_EQ(src.get(), nullptr);

  // dst's prior event was returned to the pool by the move-assignment.
  // The next get retrieves it (LIFO).
  auto e = pool.get();
  EXPECT_EQ(e.get(), dst_handle);
}

TEST_F(EventPoolTest, self_move_assignment_is_noop)
{
  auto& pool = kvikio::detail::EventPool::instance();

  auto e      = pool.get();
  auto handle = e.get();

  // Suppress the obvious self-move warning via a reference indirection.
  auto& ref = e;
  e         = std::move(ref);

  EXPECT_EQ(e.get(), handle);
  EXPECT_EQ(e.cuda_context(), current_context());
}

TEST_F(EventPoolTest, record_synchronize_is_done_round_trip)
{
  auto& pool = kvikio::detail::EventPool::instance();

  CUstream stream{};
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamCreate(&stream, CU_STREAM_DEFAULT));

  auto e = pool.get();
  e.record(stream);
  e.synchronize();
  // After synchronize, the event must report done.
  EXPECT_TRUE(e.is_done());

  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamDestroy(stream));
}

TEST_F(EventPoolTest, is_done_on_fresh_event_returns_true)
{
  // An event that has never been recorded is considered already complete by CUDA.
  auto& pool = kvikio::detail::EventPool::instance();
  auto e     = pool.get();
  EXPECT_TRUE(e.is_done());
}

TEST_F(EventPoolTest, observability_counters_consistent)
{
  auto& pool = kvikio::detail::EventPool::instance();
  auto ctx   = current_context();

  // Snapshot.
  auto total_before       = pool.total_free_events();
  auto current_ctx_before = pool.num_free_events(ctx);

  // One full lifecycle: net effect on per-context count is either 0 (a pooled event was reused
  // and returned) or +1 (the pool was empty, a new event was created and then returned).
  CUevent handle{};
  {
    auto e = pool.get();
    handle = e.get();
  }

  auto total_after       = pool.total_free_events();
  auto current_ctx_after = pool.num_free_events(ctx);

  EXPECT_GE(total_after, total_before);
  EXPECT_LE(total_after - total_before, 1);
  EXPECT_GE(current_ctx_after, current_ctx_before);
  EXPECT_LE(current_ctx_after - current_ctx_before, 1);

  // After the lifecycle, this context's pool must hold at least one event (the one we just
  // returned), so subsequent get must succeed and may LIFO-reuse the same handle.
  EXPECT_GE(current_ctx_after, 1);
  auto e2 = pool.get();
  EXPECT_NE(e2.get(), nullptr);
}

TEST_F(EventPoolTest, no_current_context_throws)
{
  auto& pool = kvikio::detail::EventPool::instance();

  // Save current context, then unset.
  auto saved_ctx = current_context();
  EXPECT_NE(saved_ctx, nullptr);

  CUcontext popped{};
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().CtxPopCurrent(&popped));
  EXPECT_EQ(current_context(), nullptr);

  EXPECT_THROW(std::ignore = pool.get(), kvikio::CUfileException);

  // Restore
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().CtxPushCurrent(saved_ctx));
  EXPECT_EQ(current_context(), saved_ctx);

  EXPECT_NO_THROW(std::ignore = pool.get());
}

TEST_F(EventPoolTest, multi_context_isolation)
{
  auto& pool = kvikio::detail::EventPool::instance();

  // SetUp() already created and made current the primary context on device 0.
  auto primary_ctx = current_context();
  EXPECT_NE(primary_ctx, nullptr);

  // Acquire and release one event under the primary context, capturing its handle.
  CUevent primary_handle{};
  {
    auto e         = pool.get();
    primary_handle = e.get();
    EXPECT_EQ(e.cuda_context(), primary_ctx);
  }

  // Create a second, non-primary context on the same device. cuCtxCreate pushes the new
  // context onto the calling thread's stack and makes it current. The signature changed in
  // CUDA 13: the v2 entry point took (CUcontext*, unsigned int, CUdevice), and v4 added a
  // leading CUctxCreateParams* (passing nullptr selects regular-context behavior).
  CUdevice dev{};
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().CtxGetDevice(&dev));
  CUcontext second_ctx{};
#if CUDA_VERSION >= 13000
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().CtxCreate(&second_ctx, nullptr, 0, dev));
#else
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().CtxCreate(&second_ctx, 0, dev));
#endif
  EXPECT_EQ(current_context(), second_ctx);
  EXPECT_NE(second_ctx, primary_ctx);

  // Acquire an event under the second context. It must NOT be the primary-context handle,
  // since events are context-specific resources and the pool is keyed per-context.
  CUevent second_handle{};
  {
    auto e        = pool.get();
    second_handle = e.get();
    EXPECT_NE(e.get(), primary_handle);
    EXPECT_EQ(e.cuda_context(), second_ctx);
  }

  // Pop second_ctx. Primary becomes current again. Verify the primary-context event is still
  // cached and reused (LIFO).
  CUcontext popped{};
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().CtxPopCurrent(&popped));
  EXPECT_EQ(popped, second_ctx);
  EXPECT_EQ(current_context(), primary_ctx);
  {
    auto e = pool.get();
    EXPECT_EQ(e.get(), primary_handle);
    EXPECT_EQ(e.cuda_context(), primary_ctx);
  }

  // Push second_ctx again and verify the second-context event is still cached there.
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().CtxPushCurrent(second_ctx));
  EXPECT_EQ(current_context(), second_ctx);
  {
    auto e = pool.get();
    EXPECT_EQ(e.get(), second_handle);
    EXPECT_EQ(e.cuda_context(), second_ctx);
  }

  // Pop and destroy second_ctx, restoring the primary context for subsequent tests. The pool's
  // sub-pool keyed by second_ctx still holds the second_handle entry. That handle is no longer
  // valid after CtxDestroy, but the pool never touches it again.
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().CtxPopCurrent(&popped));
  EXPECT_EQ(popped, second_ctx);
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().CtxDestroy(second_ctx));
  EXPECT_EQ(current_context(), primary_ctx);
}

TEST_F(EventPoolTest, concurrent_get_and_release_is_thread_safe)
{
  auto& pool = kvikio::detail::EventPool::instance();

  constexpr int num_threads           = 8;
  constexpr int iterations_per_thread = 64;

  // total_free across all threads at start.
  auto ctx                = current_context();
  auto current_ctx_before = pool.num_free_events(ctx);
  auto total_before       = pool.total_free_events();

  std::atomic<int> errors{0};
  std::vector<std::thread> workers;
  workers.reserve(num_threads);

  for (int t = 0; t < num_threads; ++t) {
    workers.emplace_back([&] {
      try {
        // Each thread must set its own current context.
        KVIKIO_CHECK_CUDA(cudaSetDevice(0));
        for (int i = 0; i < iterations_per_thread; ++i) {
          auto e = pool.get();
          if (e.get() == nullptr) { ++errors; }
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
  // After everyone returned their events, the count must be at least where it started.
  EXPECT_GE(pool.num_free_events(ctx), current_ctx_before);
  EXPECT_GE(pool.total_free_events(), total_before);
}
