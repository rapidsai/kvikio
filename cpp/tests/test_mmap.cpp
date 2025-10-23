/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <sys/mman.h>
#include <optional>
#include <stdexcept>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <kvikio/defaults.hpp>
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
    _host_buf                = CreateTempFile<value_type>(_filepath, num_elements);
    _dev_buf                 = kvikio::test::DevBuffer<value_type>{_host_buf};
    _page_size               = kvikio::get_page_size();
  }

  void TearDown() override {}

  template <typename T>
  std::vector<T> CreateTempFile(std::string const& filepath, std::size_t num_elements)
  {
    std::vector<T> v(num_elements);
    std::iota(v.begin(), v.end(), 0);
    kvikio::FileHandle f(filepath, "w");
    auto fut = f.pwrite(v.data(), v.size() * sizeof(T));
    fut.get();
    _file_size = f.nbytes();
    return v;
  }

  std::filesystem::path _filepath;
  std::size_t _file_size;
  std::size_t _page_size;
  std::vector<std::int64_t> _host_buf;
  using value_type = decltype(_host_buf)::value_type;
  kvikio::test::DevBuffer<value_type> _dev_buf;
};

TEST_F(MmapTest, invalid_file_open_flag)
{
  // Empty file open flag
  EXPECT_THAT(
    [&] {
      {
        kvikio::MmapHandle(_filepath, "");
      }
    },
    ThrowsMessage<std::invalid_argument>(HasSubstr("Unknown file open flag")));

  // Invalid file open flag
  EXPECT_THAT(
    [&] {
      {
        kvikio::MmapHandle(_filepath, "z");
      }
    },
    ThrowsMessage<std::invalid_argument>(HasSubstr("Unknown file open flag")));
}

TEST_F(MmapTest, invalid_mmap_flag)
{
  EXPECT_THAT(
    [&] {
      {
        int invalid_flag{-1};
        kvikio::MmapHandle(_filepath, "r", std::nullopt, 0, kvikio::FileHandle::m644, invalid_flag);
      }
    },
    ThrowsMessage<kvikio::GenericSystemError>(HasSubstr("Invalid argument")));
}

TEST_F(MmapTest, constructor_invalid_range)
{
  // init_size is too large (by 1 char)
  EXPECT_THAT([&] { kvikio::MmapHandle(_filepath, "r", _file_size + 1); },
              ThrowsMessage<std::out_of_range>(HasSubstr("Mapped region is past the end of file")));

  // init_file_offset is too large (by 1 char)
  EXPECT_THAT(
    [&] { kvikio::MmapHandle(_filepath, "r", std::nullopt, _file_size); },
    ThrowsMessage<std::out_of_range>(HasSubstr("Offset must be less than the file size")));

  // init_size is 0
  EXPECT_THAT(
    [&] { kvikio::MmapHandle(_filepath, "r", 0); },
    ThrowsMessage<std::invalid_argument>(HasSubstr("Mapped region should not be zero byte")));
}

TEST_F(MmapTest, constructor_valid_range)
{
  // init_size is exactly equal to file size
  EXPECT_NO_THROW({ kvikio::MmapHandle(_filepath, "r", _file_size); });

  // init_file_offset is exactly on the last char
  EXPECT_NO_THROW({
    kvikio::MmapHandle mmap_handle(_filepath, "r", std::nullopt, _file_size - 1);
    EXPECT_EQ(mmap_handle.initial_map_size(), 1);
  });
}

TEST_F(MmapTest, read_invalid_range)
{
  std::size_t const initial_map_size{1024};
  std::size_t const initial_file_offset{512};
  std::vector<value_type> out_host_buf(_file_size / sizeof(value_type), {});

  // Right bound is too large
  EXPECT_THAT(
    [&] {
      kvikio::MmapHandle mmap_handle(_filepath, "r", initial_map_size, initial_file_offset);
      mmap_handle.read(out_host_buf.data(), initial_map_size, _file_size);
    },
    ThrowsMessage<std::out_of_range>(HasSubstr("Read is out of bound")));

  // Left bound is too large
  EXPECT_THAT(
    [&] {
      kvikio::MmapHandle mmap_handle(_filepath, "r", initial_map_size, initial_file_offset);
      mmap_handle.read(out_host_buf.data(), 0, initial_file_offset + initial_map_size + 1);
    },
    ThrowsMessage<std::out_of_range>(HasSubstr("Read is out of bound")));

  EXPECT_THAT(
    [&] {
      kvikio::MmapHandle mmap_handle(_filepath, "r");
      mmap_handle.read(out_host_buf.data(), 0, _file_size + 1);
    },
    ThrowsMessage<std::out_of_range>(HasSubstr("Offset is past the end of file")));

  // Left bound is too small
  EXPECT_THAT(
    [&] {
      kvikio::MmapHandle mmap_handle(_filepath, "r", initial_map_size, initial_file_offset);
      mmap_handle.read(out_host_buf.data(), initial_map_size, initial_file_offset - 128);
    },
    ThrowsMessage<std::out_of_range>(HasSubstr("Read is out of bound")));

  // size is too large
  EXPECT_THAT(
    [&] {
      kvikio::MmapHandle mmap_handle(_filepath, "r", initial_map_size, initial_file_offset);
      mmap_handle.read(out_host_buf.data(), initial_map_size + 128, initial_file_offset);
    },
    ThrowsMessage<std::out_of_range>(HasSubstr("Read is out of bound")));
}

TEST_F(MmapTest, read_valid_range)
{
  std::size_t const initial_map_size{1024};
  std::size_t const initial_file_offset{512};
  std::vector<value_type> out_host_buf(_file_size / sizeof(value_type), {});

  // size is 0
  EXPECT_NO_THROW({
    kvikio::MmapHandle mmap_handle(_filepath, "r", initial_map_size, initial_file_offset);
    mmap_handle.read(out_host_buf.data(), 0, initial_file_offset + initial_map_size);
  });

  EXPECT_NO_THROW({
    kvikio::MmapHandle mmap_handle(_filepath, "r");
    mmap_handle.read(out_host_buf.data(), 0, _file_size);
  });
}

TEST_F(MmapTest, read_seq)
{
  auto do_test = [&](std::size_t num_elements_to_skip, std::size_t num_elements_to_read) {
    kvikio::MmapHandle mmap_handle(_filepath, "r");
    auto const offset             = num_elements_to_skip * sizeof(value_type);
    auto const expected_read_size = num_elements_to_read * sizeof(value_type);

    // host
    {
      std::vector<value_type> out_host_buf(num_elements_to_read, {});
      auto const read_size = mmap_handle.read(out_host_buf.data(), expected_read_size, offset);
      for (std::size_t i = num_elements_to_skip; i < num_elements_to_read; ++i) {
        EXPECT_EQ(_host_buf[i], out_host_buf[i - num_elements_to_skip]);
      }
      EXPECT_EQ(read_size, expected_read_size);
    }

    // device
    {
      kvikio::test::DevBuffer<value_type> out_device_buf(num_elements_to_read);
      auto const read_size = mmap_handle.read(out_device_buf.ptr, expected_read_size, offset);
      auto out_host_buf    = out_device_buf.to_vector();
      for (std::size_t i = num_elements_to_skip; i < num_elements_to_read; ++i) {
        EXPECT_EQ(_host_buf[i], out_host_buf[i - num_elements_to_skip]);
      }
      EXPECT_EQ(read_size, expected_read_size);
    }
  };

  for (const auto& num_elements_to_read : {10, 9999}) {
    for (const auto& num_elements_to_skip : {0, 10, 100, 1000, 9999}) {
      do_test(num_elements_to_skip, num_elements_to_read);
    }
  }
}

TEST_F(MmapTest, read_parallel)
{
  auto do_test =
    [&](std::size_t num_elements_to_skip, std::size_t num_elements_to_read, std::size_t task_size) {
      kvikio::MmapHandle mmap_handle(_filepath, "r");
      auto const offset             = num_elements_to_skip * sizeof(value_type);
      auto const expected_read_size = num_elements_to_read * sizeof(value_type);

      // host
      {
        std::vector<value_type> out_host_buf(num_elements_to_read, {});
        auto fut = mmap_handle.pread(out_host_buf.data(), expected_read_size, offset, task_size);
        auto const read_size = fut.get();
        for (std::size_t i = num_elements_to_skip; i < num_elements_to_read; ++i) {
          EXPECT_EQ(_host_buf[i], out_host_buf[i - num_elements_to_skip]);
        }
        EXPECT_EQ(read_size, expected_read_size);
      }

      // device
      {
        kvikio::test::DevBuffer<value_type> out_device_buf(num_elements_to_read);
        auto fut             = mmap_handle.pread(out_device_buf.ptr, expected_read_size, offset);
        auto const read_size = fut.get();
        auto out_host_buf    = out_device_buf.to_vector();
        for (std::size_t i = num_elements_to_skip; i < num_elements_to_read; ++i) {
          EXPECT_EQ(_host_buf[i], out_host_buf[i - num_elements_to_skip]);
        }
        EXPECT_EQ(read_size, expected_read_size);
      }
    };

  std::vector<std::size_t> task_sizes{256, 1024, kvikio::defaults::task_size()};
  for (const auto& task_size : task_sizes) {
    for (const auto& num_elements_to_read : {10, 9999}) {
      for (const auto& num_elements_to_skip : {0, 10, 100, 1000, 9999}) {
        do_test(num_elements_to_skip, num_elements_to_read, task_size);
      }
    }
  }
}

TEST_F(MmapTest, read_with_default_arguments)
{
  std::size_t num_elements = _file_size / sizeof(value_type);
  kvikio::MmapHandle mmap_handle(_filepath, "r");

  // host
  {
    std::vector<value_type> out_host_buf(num_elements, {});

    {
      auto const read_size = mmap_handle.read(out_host_buf.data());
      for (std::size_t i = 0; i < num_elements; ++i) {
        EXPECT_EQ(_host_buf[i], out_host_buf[i]);
      }
      EXPECT_EQ(read_size, _file_size);
    }

    {
      auto fut             = mmap_handle.pread(out_host_buf.data());
      auto const read_size = fut.get();
      for (std::size_t i = 0; i < num_elements; ++i) {
        EXPECT_EQ(_host_buf[i], out_host_buf[i]);
      }
      EXPECT_EQ(read_size, _file_size);
    }
  }

  // device
  {
    kvikio::test::DevBuffer<value_type> out_device_buf(num_elements);

    {
      auto const read_size = mmap_handle.read(out_device_buf.ptr);
      auto out_host_buf    = out_device_buf.to_vector();
      for (std::size_t i = 0; i < num_elements; ++i) {
        EXPECT_EQ(_host_buf[i], out_host_buf[i]);
      }
      EXPECT_EQ(read_size, _file_size);
    }

    {
      auto fut             = mmap_handle.pread(out_device_buf.ptr);
      auto const read_size = fut.get();
      auto out_host_buf    = out_device_buf.to_vector();
      for (std::size_t i = 0; i < num_elements; ++i) {
        EXPECT_EQ(_host_buf[i], out_host_buf[i]);
      }
      EXPECT_EQ(read_size, _file_size);
    }
  }
}

TEST_F(MmapTest, closed_handle)
{
  kvikio::MmapHandle mmap_handle(_filepath, "r");
  mmap_handle.close();

  EXPECT_TRUE(mmap_handle.closed());
  EXPECT_EQ(mmap_handle.file_size(), 0);

  std::size_t num_elements = _file_size / sizeof(value_type);
  std::vector<value_type> out_host_buf(num_elements, {});

  EXPECT_THAT([&] { mmap_handle.read(out_host_buf.data()); },
              ThrowsMessage<std::runtime_error>(HasSubstr("Cannot read from a closed MmapHandle")));

  EXPECT_THAT([&] { mmap_handle.pread(out_host_buf.data()); },
              ThrowsMessage<std::runtime_error>(HasSubstr("Cannot read from a closed MmapHandle")));
}

TEST_F(MmapTest, cpp_move)
{
  auto do_test = [&](kvikio::MmapHandle& mmap_handle) {
    std::size_t num_elements = _file_size / sizeof(value_type);
    std::vector<value_type> out_host_buf(num_elements, {});

    EXPECT_NO_THROW({ mmap_handle.read(out_host_buf.data()); });
    auto fut             = mmap_handle.pread(out_host_buf.data());
    auto const read_size = fut.get();
    for (std::size_t i = 0; i < num_elements; ++i) {
      EXPECT_EQ(_host_buf[i], out_host_buf[i]);
    }
    EXPECT_EQ(read_size, _file_size);
  };

  {
    kvikio::MmapHandle mmap_handle{};
    EXPECT_TRUE(mmap_handle.closed());
    mmap_handle = kvikio::MmapHandle(_filepath, "r");
    EXPECT_FALSE(mmap_handle.closed());
    do_test(mmap_handle);
  }

  {
    kvikio::MmapHandle mmap_handle_1(_filepath, "r");
    kvikio::MmapHandle mmap_handle_2{std::move(mmap_handle_1)};
    EXPECT_TRUE(mmap_handle_1.closed());
    EXPECT_FALSE(mmap_handle_2.closed());
    do_test(mmap_handle_2);
  }
}
