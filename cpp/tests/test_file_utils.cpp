/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <kvikio/error.hpp>
#include <kvikio/file_utils.hpp>

using ::testing::HasSubstr;
using ::testing::ThrowsMessage;

TEST(FileUtilsTest, get_block_device_info)
{
  EXPECT_THAT(
    [] {
      std::string const nonexistent_file_path{"nonexistent_file_path"};
      kvikio::get_block_device_info(nonexistent_file_path);
    },
    ThrowsMessage<kvikio::GenericSystemError>(HasSubstr("No such file or directory")));
}
