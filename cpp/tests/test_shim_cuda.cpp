/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstddef>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <cuda.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <kvikio/error.hpp>
#include <kvikio/shim/cuda.hpp>
#include <kvikio/utils.hpp>

#include "utils/utils.hpp"

using ::testing::HasSubstr;
using ::testing::ThrowsMessage;

namespace {
constexpr std::size_t buffer_size = 4096;
}  // namespace

class CudaMemcpyBatchAsyncTest : public testing::Test {
 protected:
  void SetUp() override
  {
    KVIKIO_CHECK_CUDA(cudaSetDevice(0));
    KVIKIO_CUDA_DRIVER_TRY(
      kvikio::cudaAPI::instance().StreamCreate(&_stream, CU_STREAM_NON_BLOCKING));
    KVIKIO_CHECK_CUDA(cudaMallocHost(&_host, buffer_size));
    _device = kvikio::test::DevBuffer<std::byte>{buffer_size};

    auto* host_bytes = static_cast<std::byte*>(_host);
    for (std::size_t i = 0; i < buffer_size; ++i) {
      host_bytes[i] = static_cast<std::byte>(i % 251);
    }
  }

  void TearDown() override
  {
    KVIKIO_CHECK_CUDA(cudaFreeHost(_host));
    KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamDestroy(_stream));
  }

  [[nodiscard]] CUdeviceptr host_at(std::size_t offset) const
  {
    return kvikio::convert_void2deviceptr(static_cast<std::byte*>(_host) + offset);
  }

  [[nodiscard]] CUdeviceptr device_at(std::size_t offset) const
  {
    return kvikio::convert_void2deviceptr(static_cast<std::byte*>(_device.ptr) + offset);
  }

  [[nodiscard]] std::vector<std::byte> device_slice(std::size_t offset, std::size_t size) const
  {
    auto const all = _device.to_vector();
    return {all.begin() + offset, all.begin() + offset + size};
  }

  [[nodiscard]] std::vector<std::byte> host_slice(std::size_t offset, std::size_t size) const
  {
    auto const* host_bytes = static_cast<std::byte const*>(_host);
    return {host_bytes + offset, host_bytes + offset + size};
  }

  void sync() const
  {
    KVIKIO_CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamSynchronize(_stream));
  }

  CUstream _stream{};
  void* _host{nullptr};
  kvikio::test::DevBuffer<std::byte> _device{};
};

TEST_F(CudaMemcpyBatchAsyncTest, single_entry)
{
  std::vector<CUdeviceptr> const dsts{device_at(0)};
  std::vector<CUdeviceptr> const srcs{host_at(0)};
  std::vector<std::size_t> const sizes{buffer_size};

  ASSERT_EQ(kvikio::cudaAPI::cuda_memcpy_batch_async(dsts, srcs, sizes, _stream), CUDA_SUCCESS);
  sync();

  EXPECT_EQ(device_slice(0, buffer_size), host_slice(0, buffer_size));
}

TEST_F(CudaMemcpyBatchAsyncTest, copy_disjoint_ranges)
{
  struct Piece {
    std::size_t src_offset;
    std::size_t dst_offset;
    std::size_t size;
  };
  std::vector<Piece> const pieces{{0, 0, 100}, {600, 512, 250}, {2000, 1024, 64}};

  std::vector<CUdeviceptr> dsts;
  std::vector<CUdeviceptr> srcs;
  std::vector<std::size_t> sizes;
  for (auto const& piece : pieces) {
    dsts.push_back(device_at(piece.dst_offset));
    srcs.push_back(host_at(piece.src_offset));
    sizes.push_back(piece.size);
  }

  ASSERT_EQ(kvikio::cudaAPI::cuda_memcpy_batch_async(dsts, srcs, sizes, _stream), CUDA_SUCCESS);
  sync();

  for (auto const& piece : pieces) {
    EXPECT_EQ(device_slice(piece.dst_offset, piece.size), host_slice(piece.src_offset, piece.size));
  }
}

TEST_F(CudaMemcpyBatchAsyncTest, default_stream_fallback)
{
  std::vector<CUdeviceptr> const dsts{device_at(0), device_at(1024)};
  std::vector<CUdeviceptr> const srcs{host_at(64), host_at(2048)};
  std::vector<std::size_t> const sizes{256, 128};

  ASSERT_EQ(kvikio::cudaAPI::cuda_memcpy_batch_async(dsts, srcs, sizes, nullptr), CUDA_SUCCESS);
  KVIKIO_CHECK_CUDA(cudaDeviceSynchronize());

  EXPECT_EQ(device_slice(0, 256), host_slice(64, 256));
  EXPECT_EQ(device_slice(1024, 128), host_slice(2048, 128));
}

TEST_F(CudaMemcpyBatchAsyncTest, empty_batch)
{
  EXPECT_EQ(kvikio::cudaAPI::cuda_memcpy_batch_async({}, {}, {}, _stream), CUDA_SUCCESS);
}

TEST_F(CudaMemcpyBatchAsyncTest, mismatched_lengths_throw)
{
  std::vector<CUdeviceptr> const dsts{device_at(0), device_at(1024)};
  std::vector<CUdeviceptr> const srcs{host_at(0)};
  std::vector<std::size_t> const sizes{128, 128};

  EXPECT_THAT(
    [&] { std::ignore = kvikio::cudaAPI::cuda_memcpy_batch_async(dsts, srcs, sizes, _stream); },
    ThrowsMessage<std::invalid_argument>(HasSubstr("same length")));
}

TEST_F(CudaMemcpyBatchAsyncTest, single_copy_wrapper)
{
  ASSERT_EQ(kvikio::cudaAPI::cuda_memcpy_async(device_at(512), host_at(128), 320, _stream),
            CUDA_SUCCESS);
  sync();

  EXPECT_EQ(device_slice(512, 320), host_slice(128, 320));
}
