/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <kvikio/defaults.hpp>
#include <kvikio/file_handle.hpp>
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
