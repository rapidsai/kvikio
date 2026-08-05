/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <atomic>
#include <thread>
#include <vector>

#include <cuda.h>
#include <gtest/gtest.h>

#include <kvikio/detail/io_event_barrier.hpp>
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

class IoEventBarrierTest : public testing::Test {
 protected:
  void SetUp() override { KVIKIO_CHECK_CUDA(cudaSetDevice(0)); }
};

TEST_F(IoEventBarrierTest, cuda_context_stored)
{
  auto ctx = current_context();
  kvikio::detail::IoEventBarrier barrier(ctx);
  EXPECT_EQ(barrier.cuda_context(), ctx);
}

TEST_F(IoEventBarrierTest, sync_with_no_records_is_noop)
{
  kvikio::detail::IoEventBarrier barrier(current_context());
  // No slots, nothing to wait for.
  EXPECT_NO_THROW(barrier.sync_all_events());
}

TEST_F(IoEventBarrierTest, single_thread_record_and_sync)
{
  kvikio::detail::IoEventBarrier barrier(current_context());

  CUstream stream{};
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamCreate(&stream, CU_STREAM_DEFAULT));

  EXPECT_NO_THROW({
    barrier.record_event(stream);
    barrier.sync_all_events();
  });

  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamDestroy(stream));
}

TEST_F(IoEventBarrierTest, re_record_overwrites_same_slot)
{
  kvikio::detail::IoEventBarrier barrier(current_context());

  CUstream stream{};
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamCreate(&stream, CU_STREAM_DEFAULT));

  // Multiple records on the same thread reuse the same slot. sync_all_events should still succeed
  // after the final re-record.
  EXPECT_NO_THROW({
    barrier.record_event(stream);
    barrier.record_event(stream);
    barrier.record_event(stream);
    barrier.sync_all_events();
  });

  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamDestroy(stream));
}

TEST_F(IoEventBarrierTest, sync_is_context_agnostic)
{
  auto ctx = current_context();
  ASSERT_NE(ctx, nullptr);

  CUdevice dev{};
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().CtxGetDevice(&dev));

  kvikio::detail::IoEventBarrier barrier(ctx);

  CUstream stream{};
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamCreate(&stream, CU_STREAM_DEFAULT));
  barrier.record_event(stream);

  // Case 1: no context current on the calling thread.
  CUcontext popped{};
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().CtxPopCurrent(&popped));
  ASSERT_EQ(popped, ctx);
  ASSERT_EQ(current_context(), nullptr);
  EXPECT_NO_THROW(barrier.sync_all_events());

  // Case 2: a different context current on the calling thread.
  CUcontext other_ctx{};
#if CUDA_VERSION >= 13000
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().CtxCreate(&other_ctx, nullptr, 0, dev));
#else
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().CtxCreate(&other_ctx, 0, dev));
#endif
  ASSERT_NE(other_ctx, ctx);
  ASSERT_EQ(current_context(), other_ctx);
  EXPECT_NO_THROW(barrier.sync_all_events());

  // Restore the primary context.
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().CtxPopCurrent(&popped));
  ASSERT_EQ(popped, other_ctx);
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().CtxDestroy(other_ctx));
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().CtxPushCurrent(ctx));
  ASSERT_EQ(current_context(), ctx);

  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamDestroy(stream));
}

TEST_F(IoEventBarrierTest, multi_thread_record_then_sync_on_caller)
{
  kvikio::detail::IoEventBarrier barrier(current_context());

  CUstream stream{};
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamCreate(&stream, CU_STREAM_DEFAULT));

  constexpr int num_workers = 4;
  std::atomic<int> errors{0};
  std::vector<std::thread> workers;
  workers.reserve(num_workers);

  for (int i = 0; i < num_workers; ++i) {
    workers.emplace_back([&] {
      try {
        KVIKIO_CHECK_CUDA(cudaSetDevice(0));
        barrier.record_event(stream);
      } catch (...) {
        ++errors;
      }
    });
  }
  for (auto& w : workers) {
    w.join();
  }
  EXPECT_EQ(errors.load(), 0);

  EXPECT_NO_THROW(barrier.sync_all_events());

  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamDestroy(stream));
}
