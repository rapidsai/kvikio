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
    _pagesize = kvikio::get_page_size();
    _filepath = _tmp_dir.path() / "test";
    // The file is 10 pages long
    _filesize = 10 * _pagesize;

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
  std::size_t _pagesize;
};

TEST_F(PageCacheTest, get_page_cache_info_cold_file)
{
  // Drop all pages
  kvikio::drop_file_page_cache(_filepath.string());

  auto [cached_pages, total_pages] = kvikio::get_page_cache_info(_filepath.string());
  EXPECT_EQ(cached_pages, 0);
  EXPECT_EQ(total_pages, 10);
}

TEST_F(PageCacheTest, get_page_cache_info_fully_warmed)
{
  // Warm the entire file
  WarmPageCache(0, _filesize);

  auto [cached_pages, total_pages] = kvikio::get_page_cache_info(_filepath.string());
  EXPECT_EQ(cached_pages, 10);
  EXPECT_EQ(total_pages, 10);
}

TEST_F(PageCacheTest, get_page_cache_info_partially_warmed)
{
  // Warm the entire file, then drop pages 5-9, and keep 0-4 warm
  WarmPageCache(0, _filesize);
  kvikio::drop_file_page_cache(_filepath.string(), 5 * _pagesize, 5 * _pagesize);

  // Full file query should show 5 cached out of 10
  auto [cached_pages, total_pages] = kvikio::get_page_cache_info(_filepath.string());
  EXPECT_EQ(cached_pages, 5);
  EXPECT_EQ(total_pages, 10);
}

TEST_F(PageCacheTest, get_page_cache_info_partial_query_warm_region)
{
  // Warm the entire file, then drop pages 0-2, 7-9 and keep 3-6 warm
  WarmPageCache(0, _filesize);
  kvikio::drop_file_page_cache(_filepath.string(), 0, 2 * _pagesize);
  kvikio::drop_file_page_cache(_filepath.string(), 7 * _pagesize, 3 * _pagesize);

  // Query only the warm region
  auto [cached_pages, total_pages] =
    kvikio::get_page_cache_info(_filepath.string(), 3 * _pagesize, 4 * _pagesize);
  EXPECT_EQ(cached_pages, 4);
  EXPECT_EQ(total_pages, 4);
}

TEST_F(PageCacheTest, get_page_cache_info_partial_query_cold_region)
{
  // Warm the entire file, then drop pages 5-9
  WarmPageCache(0, _filesize);
  kvikio::drop_file_page_cache(_filepath.string(), 5 * _pagesize, 5 * _pagesize);

  // Query only the cold region (pages 5-9)
  auto [cached_pages, total_pages] =
    kvikio::get_page_cache_info(_filepath.string(), 5 * _pagesize, 5 * _pagesize);
  EXPECT_EQ(cached_pages, 0);
  EXPECT_EQ(total_pages, 5);
}

TEST_F(PageCacheTest, get_page_cache_info_unaligned_offset)
{
  // Warm the entire file
  WarmPageCache(0, _filesize);

  // The page where unaligned starting offset occurs is warm
  // Query with an offset at the midpoint of page 2. The implementation should align down to page 2,
  // so pages 2-9 (8 pages) should be queried, all warm
  {
    auto [cached_pages, total_pages] =
      kvikio::get_page_cache_info(_filepath.string(), 2 * _pagesize + _pagesize / 2, 0);
    EXPECT_EQ(cached_pages, 8);
    EXPECT_EQ(total_pages, 8);
  }

  // The page where unaligned starting offset occurs is cold
  // Drop page 2
  kvikio::drop_file_page_cache(_filepath.string(), 2 * _pagesize, 1 * _pagesize);
  {
    auto [cached_pages, total_pages] =
      kvikio::get_page_cache_info(_filepath.string(), 2 * _pagesize + _pagesize / 2, 0);
    EXPECT_EQ(cached_pages, 7);
    EXPECT_EQ(total_pages, 8);
  }
}

TEST_F(PageCacheTest, get_page_cache_info_unaligned_length)
{
  // Warm the entire file
  WarmPageCache(0, _filesize);

  // The page where unaligned ending offset occurs is warm
  // Query 2.5 pages starting at page 3. Should cover pages 3, 4, 5 (3 total pages)
  {
    auto [cached_pages, total_pages] =
      kvikio::get_page_cache_info(_filepath.string(), 3 * _pagesize, 2 * _pagesize + _pagesize / 2);
    EXPECT_EQ(cached_pages, 3);
    EXPECT_EQ(total_pages, 3);
  }

  // The page where unaligned ending offset occurs is cold
  // Drop page 5
  kvikio::drop_file_page_cache(_filepath.string(), 5 * _pagesize, 1 * _pagesize);
  {
    auto [cached_pages, total_pages] =
      kvikio::get_page_cache_info(_filepath.string(), 3 * _pagesize, 2 * _pagesize + _pagesize / 2);
    EXPECT_EQ(cached_pages, 2);
    EXPECT_EQ(total_pages, 3);
  }
}

TEST_F(PageCacheTest, get_page_cache_info_length_zero_from_offset)
{
  WarmPageCache(0, _filesize);

  // length=0 means entire file from offset; offset at page 4 should give pages 4-9
  auto [cached_pages, total_pages] =
    kvikio::get_page_cache_info(_filepath.string(), 4 * _pagesize, 0);
  EXPECT_EQ(total_pages, 6);
  EXPECT_EQ(cached_pages, 6);
}

TEST_F(PageCacheTest, get_page_cache_info_offset_at_file_size)
{
  auto [cached_pages, total_pages] = kvikio::get_page_cache_info(_filepath.string(), _filesize, 0);
  EXPECT_EQ(cached_pages, 0);
  EXPECT_EQ(total_pages, 0);
}

TEST_F(PageCacheTest, get_page_cache_info_offset_beyond_file_size)
{
  auto [cached_pages, total_pages] =
    kvikio::get_page_cache_info(_filepath.string(), _filesize + _pagesize, 0);
  EXPECT_EQ(cached_pages, 0);
  EXPECT_EQ(total_pages, 0);
}

TEST_F(PageCacheTest, get_page_cache_info_length_clamped_to_file_size)
{
  WarmPageCache(0, _filesize);

  // Query from page 8 with length far exceeding EOF. Should clamp to pages 8-9
  auto [cached_pages, total_pages] =
    kvikio::get_page_cache_info(_filepath.string(), 8 * _pagesize, 100 * _pagesize);
  EXPECT_EQ(total_pages, 2);
  EXPECT_EQ(cached_pages, 2);
}

TEST_F(PageCacheTest, get_page_cache_info_empty_file)
{
  // Create a separate empty file
  auto empty_path = _tmp_dir.path() / "empty";
  kvikio::FileHandle empty_file(empty_path.string(), "w");

  auto [cached_pages, total_pages] = kvikio::get_page_cache_info(empty_path.string());
  EXPECT_EQ(cached_pages, 0);
  EXPECT_EQ(total_pages, 0);
}

TEST_F(PageCacheTest, get_page_cache_info_fd_matches_path)
{
  WarmPageCache(0, _filesize);

  auto result_path = kvikio::get_page_cache_info(_filepath.string());

  kvikio::FileHandle file(_filepath.string(), "r");
  auto result_fd = kvikio::get_page_cache_info(file.fd());

  EXPECT_EQ(result_path.first, result_fd.first);
  EXPECT_EQ(result_path.second, result_fd.second);
}

TEST_F(PageCacheTest, get_page_cache_info_fd_with_offset_and_length)
{
  // Warm the entire file, then drop pages 0-2 and 7-9, keeping 3-6 warm
  WarmPageCache(0, _filesize);
  kvikio::drop_file_page_cache(_filepath.string(), 0, 3 * _pagesize);
  kvikio::drop_file_page_cache(_filepath.string(), 7 * _pagesize, 3 * _pagesize);

  kvikio::FileHandle file(_filepath.string(), "r");
  auto [cached_pages, total_pages] =
    kvikio::get_page_cache_info(file.fd(), 3 * _pagesize, 4 * _pagesize);
  EXPECT_EQ(cached_pages, 4);
  EXPECT_EQ(total_pages, 4);
}

TEST_F(PageCacheTest, get_page_cache_info_invalid_fd)
{
  EXPECT_THROW(kvikio::get_page_cache_info(-1), kvikio::GenericSystemError);
}

TEST_F(PageCacheTest, get_page_cache_info_nonexistent_file)
{
  EXPECT_THROW(kvikio::get_page_cache_info("/nonexistent/path/file.bin"),
               kvikio::GenericSystemError);
}

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
  std::size_t file_offset = 3 * _pagesize;
  std::size_t length      = 4 * _pagesize;
  kvikio::drop_file_page_cache(_filepath.string(), file_offset, length);

  // 6 pages remain in the page cache
  auto [cached_pages, _] = kvikio::get_page_cache_info(_filepath.string());
  EXPECT_EQ(cached_pages, 6);
}

// Test the fd-based overload
TEST_F(PageCacheTest, drop_file_page_cache_with_fd)
{
  WarmPageCache(0, _filesize);

  {
    auto [cached_pages, total_pages] = kvikio::get_page_cache_info(_filepath.string());
    EXPECT_EQ(cached_pages, total_pages);
  }

  kvikio::FileHandle file(_filepath.string(), "r");
  kvikio::drop_file_page_cache(file.fd());

  {
    auto [cached_pages, _] = kvikio::get_page_cache_info(_filepath.string());
    EXPECT_EQ(cached_pages, 0);
  }
}

TEST_F(PageCacheTest, drop_file_page_cache_invalid_fd)
{
  EXPECT_THROW(kvikio::drop_file_page_cache(-1), kvikio::GenericSystemError);
}

TEST_F(PageCacheTest, drop_file_page_cache_nonexistent_file)
{
  EXPECT_THROW(kvikio::drop_file_page_cache("/nonexistent/path/file.bin"),
               kvikio::GenericSystemError);
}

TEST_F(PageCacheTest, drop_file_page_cache_unaligned_range)
{
  // Read the full file
  WarmPageCache(0, _filesize);

  // Attempt to drop pages 3 (half), 4, 5, 6, 7 (half)
  // Actually drop pages 4, 5, 6
  std::size_t file_offset = 3 * _pagesize + _pagesize / 2;
  std::size_t length      = 4 * _pagesize;
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
