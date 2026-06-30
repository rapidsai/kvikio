/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
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
  barrier.sync_all_events();  // No slots, nothing to wait for.
}

TEST_F(IoEventBarrierTest, single_thread_record_and_sync)
{
  kvikio::detail::IoEventBarrier barrier(current_context());

  CUstream stream{};
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamCreate(&stream, CU_STREAM_DEFAULT));

  barrier.record_event(stream);
  barrier.sync_all_events();

  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamDestroy(stream));
}

TEST_F(IoEventBarrierTest, re_record_overwrites_same_slot)
{
  kvikio::detail::IoEventBarrier barrier(current_context());

  CUstream stream{};
  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamCreate(&stream, CU_STREAM_DEFAULT));

  // Multiple records on the same thread reuse the same slot. sync_all_events should still
  // succeed after the final re-record.
  barrier.record_event(stream);
  barrier.record_event(stream);
  barrier.record_event(stream);
  barrier.sync_all_events();

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

  // Caller (main thread) waits for every worker thread's recorded event.
  barrier.sync_all_events();

  KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamDestroy(stream));
}
