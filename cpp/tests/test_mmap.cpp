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
#include <kvikio/file_handle.hpp>
#include <kvikio/mmap.hpp>
#include <kvikio/utils.hpp>

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

  using value_type = decltype(_host_buf)::value_type;
};

TEST_F(MmapTest, file_open_flag_in_constructor)
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
}

TEST_F(MmapTest, eof_in_constructor)
{
  // size is too large (by 1 char)
  EXPECT_THAT(
    [=] { kvikio::MmapHandle(_filepath, "r", _file_size + 1); },
    ThrowsMessage<std::overflow_error>(HasSubstr("Mapped region is past the end of file")));

  // size is exactly equal to file size
  EXPECT_NO_THROW({ kvikio::MmapHandle(_filepath, "r", _file_size); });

  // file_offset is too large (by 1 char)
  EXPECT_THAT([=] { kvikio::MmapHandle(_filepath, "r", 0, _file_size); },
              ThrowsMessage<std::overflow_error>(HasSubstr("Offset is past the end of file")));

  // file_offset is exactly on the last char
  EXPECT_NO_THROW({
    kvikio::MmapHandle mmap_handle(_filepath, "r", 0, _file_size - 1);
    EXPECT_EQ(mmap_handle.requested_size(), 1);
  });
}

TEST_F(MmapTest, read)
{
  auto do_test = [&](std::size_t num_elements_to_skip, bool prefault) {
    auto offset = num_elements_to_skip * sizeof(value_type);
    kvikio::MmapHandle mmap_handle(_filepath, "r");
    auto const [buf, read_size] =
      mmap_handle.read(mmap_handle.requested_size() - offset, offset, prefault);
    auto result_buf = static_cast<value_type*>(buf);
    for (std::size_t i = num_elements_to_skip; i < _host_buf.size(); ++i) {
      EXPECT_EQ(_host_buf[i], result_buf[i - num_elements_to_skip]);
    }
    EXPECT_EQ(read_size, (_host_buf.size() - num_elements_to_skip) * sizeof(value_type));
  };

  for (const auto& num_elements_to_skip : {0, 1, 100, 1000, 99999}) {
    do_test(num_elements_to_skip, true);
    do_test(num_elements_to_skip, false);
  }
}
