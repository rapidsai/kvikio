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

#include <kvikio/defaults.hpp>
#include <kvikio/file_handle.hpp>
#include "utils.hpp"

using namespace kvikio::test;

class BasicIOTest : public testing::Test {
 protected:
  void SetUp() override
  {
    TempDir tmp_dir{false};
    _filepath = tmp_dir.path() / "test";

    _dev_a = std::move(DevBuffer::arange(100));
    _dev_b = std::move(DevBuffer::zero_like(_dev_a));
  }

  void TearDown() override {}

  std::filesystem::path _filepath;
  DevBuffer _dev_a;
  DevBuffer _dev_b;
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
