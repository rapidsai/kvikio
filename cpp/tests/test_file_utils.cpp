/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <filesystem>
#include <kvikio/error.hpp>
#include <kvikio/file_handle.hpp>
#include <kvikio/file_utils.hpp>
#include <kvikio/utils.hpp>

#include "utils/utils.hpp"

using ::testing::HasSubstr;
using ::testing::ThrowsMessage;

class PageCacheTest : public testing::Test {
 protected:
  void SetUp() override
  {
    _filepath = _tmp_dir.path() / "test";
    // The file is 10 pages long
    _filesize = 10 * kvikio::get_page_size();

    kvikio::FileHandle file(_filepath.string(), "w");
    std::vector<std::byte> v(_filesize, {});
    auto fut = file.pwrite(v.data(), _filesize);
    fut.get();
  }

  void TearDown() override {}

  // Read a range of the file to populate the page cache
  void WarmPageCache(std::size_t file_offset, std::size_t size)
  {
    kvikio::FileHandle file(_filepath.string(), "r");
    std::vector<std::byte> v(file.nbytes());
    auto fut = file.pread(v.data(), size, file_offset);
    fut.get();
  }

  kvikio::test::TempDir _tmp_dir;
  std::filesystem::path _filepath;
  std::size_t _filesize;
};

TEST_F(PageCacheTest, drop_file_page_cache_full_range)
{
  // Read the full file
  WarmPageCache(0, _filesize);

  {
    // Verify pages are cached
    auto [cached_pages, total_pages] = kvikio::get_page_cache_info(_filepath.string());
    EXPECT_EQ(cached_pages, total_pages);  // All pages should be resident
  }

  kvikio::drop_file_page_cache(_filepath.string());

  {
    // Verify pages are evicted
    auto [cached_pages, _] = kvikio::get_page_cache_info(_filepath.string());
    EXPECT_EQ(cached_pages, 0);  // No pages should be resident
  }
}

TEST_F(PageCacheTest, drop_file_page_cache_partial_range)
{
  // Read the full file
  WarmPageCache(0, _filesize);

  // Drop pages 3, 4, 5, 6
  // Skip pages 0, 1, 2, and 7, 8, 9
  std::size_t file_offset = 3 * kvikio::get_page_size();
  std::size_t length      = 4 * kvikio::get_page_size();
  kvikio::drop_file_page_cache(_filepath.string(), file_offset, length);

  // 6 pages remain in the page cache
  auto [cached_pages, _] = kvikio::get_page_cache_info(_filepath.string());
  EXPECT_EQ(cached_pages, 6);
}

TEST_F(PageCacheTest, drop_file_page_cache_unaligned_range)
{
  // Read the full file
  WarmPageCache(0, _filesize);

  // Attempt to drop pages 3 (half), 4, 5, 6, 7 (half)
  // Actually drop pages 4, 5, 6
  std::size_t file_offset = 3 * kvikio::get_page_size() + kvikio::get_page_size() / 2;
  std::size_t length      = 4 * kvikio::get_page_size();
  kvikio::drop_file_page_cache(_filepath.string(), file_offset, length);

  // 7 pages remain in the page cache
  auto [cached_pages, _] = kvikio::get_page_cache_info(_filepath.string());
  EXPECT_EQ(cached_pages, 7);
}

TEST(FileUtilsTest, get_block_device_info)
{
  EXPECT_THAT(
    [] {
      std::string const nonexistent_file_path{"nonexistent_file_path"};
      kvikio::get_block_device_info(nonexistent_file_path);
    },
    ThrowsMessage<kvikio::GenericSystemError>(HasSubstr("No such file or directory")));
}
