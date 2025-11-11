/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <fcntl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <kvikio/defaults.hpp>
#include <kvikio/detail/utils.hpp>
#include <kvikio/error.hpp>
#include <kvikio/file_handle.hpp>
#include <kvikio/file_utils.hpp>
#include <kvikio/utils.hpp>

#include "utils/env.hpp"
#include "utils/utils.hpp"

using namespace kvikio::test;

class BasicIOTest : public testing::Test {
 protected:
  void SetUp() override
  {
    TempDir tmp_dir{false};
    _filepath = tmp_dir.path() / "test";

    _dev_a = std::move(DevBuffer<value_type>::arange(100));
    _dev_b = std::move(DevBuffer<value_type>::zero_like(_dev_a));
  }

  void TearDown() override {}

  std::filesystem::path _filepath;
  using value_type = std::int64_t;
  DevBuffer<value_type> _dev_a;
  DevBuffer<value_type> _dev_b;
};

TEST_F(BasicIOTest, write_read)
{
  {
    kvikio::FileHandle f(_filepath, "w");
    auto nbytes = f.write(_dev_a.ptr, _dev_a.nbytes, 0, 0);
    EXPECT_EQ(nbytes, _dev_a.nbytes);
  }

  {
    kvikio::FileHandle f(_filepath, "r");
    auto nbytes = f.read(_dev_b.ptr, _dev_b.nbytes, 0, 0);
    EXPECT_EQ(nbytes, _dev_b.nbytes);
    expect_equal(_dev_a, _dev_b);
  }
}

TEST_F(BasicIOTest, write_read_async)
{
  CUstream stream{};
  CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamCreate(&stream, CU_STREAM_NON_BLOCKING));

  // Default compatibility mode (AUTO)
  {
    kvikio::FileHandle f(_filepath, "w");
    auto stream_future = f.write_async(_dev_a.ptr, _dev_a.nbytes, 0, 0, stream);
    auto nbytes        = stream_future.check_bytes_done();
    EXPECT_EQ(nbytes, _dev_a.nbytes);
  }

  {
    kvikio::FileHandle f(_filepath, "r");
    auto stream_future = f.read_async(_dev_b.ptr, _dev_b.nbytes, 0, 0, stream);
    auto nbytes        = stream_future.check_bytes_done();
    EXPECT_EQ(nbytes, _dev_b.nbytes);
    expect_equal(_dev_a, _dev_b);
  }

  // Explicitly set compatibility mode
  std::array<kvikio::CompatMode, 2> compat_modes{kvikio::CompatMode::AUTO, kvikio::CompatMode::ON};
  for (auto const& compat_mode : compat_modes) {
    {
      kvikio::FileHandle f(_filepath, "w", kvikio::FileHandle::m644, compat_mode);
      auto stream_future = f.write_async(_dev_a.ptr, _dev_a.nbytes, 0, 0, stream);
      auto nbytes        = stream_future.check_bytes_done();
      EXPECT_EQ(nbytes, _dev_a.nbytes);
    }

    {
      kvikio::FileHandle f(_filepath, "r", kvikio::FileHandle::m644, compat_mode);
      auto stream_future = f.read_async(_dev_b.ptr, _dev_b.nbytes, 0, 0, stream);
      auto nbytes        = stream_future.check_bytes_done();
      EXPECT_EQ(nbytes, _dev_b.nbytes);
      expect_equal(_dev_a, _dev_b);
    }
  }

  CUDA_DRIVER_TRY(kvikio::cudaAPI::instance().StreamDestroy(stream));
}

class DirectIOTest : public testing::Test {
 public:
  using value_type = std::int64_t;

 protected:
  void SetUp() override
  {
    TempDir tmp_dir{false};
    _filepath = tmp_dir.path() / "test";

    // Skip the fixture is Direct I/O is not supported
    try {
      [[maybe_unused]] auto fd =
        kvikio::open_fd(_filepath.c_str(), "w", true /* o_direct */, kvikio::FileHandle::m644);
    } catch (...) {
      GTEST_SKIP() << "Direct I/O is not supported for the test file: " << _filepath;
    }

    // Create a sequence of numbers as a ground truth
    _num_elements = 8ULL * 1024ULL * 1024ULL + 1234ULL;
    _total_bytes  = _num_elements * sizeof(value_type);
    _ground_truth.resize(_num_elements);
    std::iota(_ground_truth.begin(), _ground_truth.end(), 0);
  }

  void TearDown() override {}

  std::filesystem::path _filepath;
  std::size_t _num_elements{};
  std::vector<value_type> _ground_truth;
  std::size_t _total_bytes{};

 public:
  static std::size_t constexpr page_size{4096};
  using AlignedAllocator   = kvikio::test::CustomHostAllocator<value_type, 4096>;
  using UnalignedAllocator = kvikio::test::CustomHostAllocator<value_type, 4096, 123>;
};

TEST_F(DirectIOTest, pwrite)
{
  // Create host buffers (page-aligned and unaligned) and device buffer for testing
  std::vector<value_type, AlignedAllocator> aligned_host_buf(_num_elements);
  std::vector<value_type, UnalignedAllocator> unaligned_host_buf(_num_elements);
  DevBuffer<value_type> dev_buf(_num_elements);

  std::array<void*, 3> buffers{aligned_host_buf.data(), unaligned_host_buf.data(), dev_buf.ptr};
  std::array auto_direct_io_statuses{true, false};

  for (const auto& flag : auto_direct_io_statuses) {
    std::string flag_str = flag ? "true" : "false";
    kvikio::test::EnvVarContext env_var_ctx{{"KVIKIO_AUTO_DIRECT_IO_WRITE", flag_str}};
    for (const auto buf : buffers) {
      // Fill up the buffer and write data to file (using KvikIO's pwrite)
      {
        if (kvikio::is_host_memory(buf)) {
          std::memcpy(buf, _ground_truth.data(), _total_bytes);
        } else {
          KVIKIO_CHECK_CUDA(
            cudaMemcpy(buf, _ground_truth.data(), _total_bytes, cudaMemcpyKind::cudaMemcpyDefault));
        }

        kvikio::FileHandle f(_filepath, "w");
        auto fut               = f.pwrite(buf, _total_bytes);
        auto num_bytes_written = fut.get();
        EXPECT_EQ(num_bytes_written, _total_bytes);
      }

      // Read data from file (using Linux syscall) and check correctness
      {
        auto fd = open(_filepath.c_str(), O_RDONLY);
        SYSCALL_CHECK(fd, "File cannot be opened");

        std::vector<value_type> result(_ground_truth.size(), 0);
        SYSCALL_CHECK(read(fd, result.data(), _total_bytes));
        EXPECT_EQ(result, _ground_truth);

        SYSCALL_CHECK(close(fd));
      }
    }
  }
}

TEST_F(DirectIOTest, pread)
{
  // Write ground truth data to file (using Linux syscall)
  {
    auto fd = open(_filepath.c_str(), O_WRONLY | O_CREAT | O_TRUNC, kvikio::FileHandle::m644);
    SYSCALL_CHECK(fd, "File cannot be opened");
    SYSCALL_CHECK(write(fd, _ground_truth.data(), _total_bytes));
    SYSCALL_CHECK(close(fd));
  }

  // Create host buffers (page-aligned and unaligned) and device buffer for testing
  std::vector<value_type, AlignedAllocator> aligned_host_buf(_num_elements);
  std::vector<value_type, UnalignedAllocator> unaligned_host_buf(_num_elements);
  DevBuffer<value_type> dev_buf(_num_elements);

  std::array<void*, 3> buffers{aligned_host_buf.data(), unaligned_host_buf.data(), dev_buf.ptr};
  std::array auto_direct_io_statuses{true, false};

  for (const auto& flag : auto_direct_io_statuses) {
    std::string flag_str = flag ? "true" : "false";
    kvikio::test::EnvVarContext env_var_ctx{{"KVIKIO_AUTO_DIRECT_IO_READ", flag_str}};
    for (const auto buf : buffers) {
      // Read data from file (using KvikIO's pread) and check correctness
      {
        kvikio::FileHandle f(_filepath, "r");
        auto fut            = f.pread(buf, _total_bytes);
        auto num_bytes_read = fut.get();
        EXPECT_EQ(num_bytes_read, _total_bytes);

        if (kvikio::is_host_memory(buf)) {
          auto* buf_helper = reinterpret_cast<value_type*>(buf);
          for (std::size_t i = 0; i < _num_elements; ++i) {
            EXPECT_EQ(buf_helper[i], _ground_truth[i]);
          }
        } else {
          std::vector<value_type> result(_num_elements);
          KVIKIO_CHECK_CUDA(
            cudaMemcpy(result.data(), buf, _total_bytes, cudaMemcpyKind::cudaMemcpyDefault));
          EXPECT_EQ(result, _ground_truth);
        }
      }
    }
  }
}
