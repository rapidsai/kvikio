/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <kvikio/error.hpp>
#include <kvikio/mmap.hpp>
#include "kvikio/file_handle.hpp"
#include "kvikio/utils.hpp"
#include "utils/utils.hpp"

using ::testing::HasSubstr;
using ::testing::ThrowsMessage;

class MmapTest : public testing::Test {
 protected:
  void SetUp() override
  {
    kvikio::test::TempDir tmp_dir{false};
    _filepath                = tmp_dir.path() / "test.bin";
    std::size_t num_elements = 1024ull * 1024ull;
    _host_buf                = CreateTempFile<double>(_filepath, num_elements);
    _page_size               = kvikio::get_page_size();
  }

  void TearDown() override {}

  template <typename T>
  std::vector<T> CreateTempFile(std::string const& filepath, std::size_t num_elements)
  {
    std::vector<T> v(num_elements);
    for (std::size_t i = 0; i < v.size(); ++i) {
      v[i] = static_cast<T>(i);
    }
    kvikio::FileHandle f(filepath, "w");
    auto fut = f.pwrite(v.data(), v.size() * sizeof(T));
    fut.get();
    _file_size = f.nbytes();
    return v;
  }

  std::filesystem::path _filepath;
  std::size_t _file_size;
  std::size_t _page_size;
  std::vector<double> _host_buf;
};

TEST_F(MmapTest, external_buffer_unspecified)
{
  // Emtpy file open flag
  EXPECT_THAT(
    [=] {
      {
        [[maybe_unused]] auto mmap_handle = kvikio::MmapHandle(_filepath, "");
      }
    },
    ThrowsMessage<std::invalid_argument>(HasSubstr("Unknown file open flag")));

  // Invalid file open flag
  EXPECT_THAT(
    [=] {
      {
        [[maybe_unused]] auto mmap_handle = kvikio::MmapHandle(_filepath, "z");
      }
    },
    ThrowsMessage<std::invalid_argument>(HasSubstr("Unknown file open flag")));

  // Read beyond EOF
  EXPECT_THAT(
    [=] {
      [[maybe_unused]] auto mmap_handle = kvikio::MmapHandle(_filepath, "r", _file_size + 1000);
    },
    ThrowsMessage<std::overflow_error>(HasSubstr("Offset is past the end of file")));

  {
    [[maybe_unused]] auto mmap_handle = kvikio::MmapHandle(_filepath, "r");
  }
}

TEST_F(MmapTest, external_host_buffer_specified) {}
