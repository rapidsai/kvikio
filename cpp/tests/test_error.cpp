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

#include <fcntl.h>
#include <sys/mman.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <kvikio/error.hpp>
#include <kvikio/file_handle.hpp>

using ::testing::HasSubstr;
using ::testing::ThrowsMessage;

TEST(ErrorTest, syscall_check_for_int_return_value)
{
  auto open_nonexistent_file = []() -> auto {
    int flag    = O_RDONLY;
    mode_t mode = kvikio::FileHandle::m644;
    return open("kvikio_nonexistent_file.bin", flag, mode);
  };

  // If the file does not exist, open() returns (int)-1, and the error number is ENOENT (No such
  // file or directory).
  EXPECT_THAT([=] { SYSCALL_CHECK(open_nonexistent_file()); },
              ThrowsMessage<kvikio::GenericSystemError>(HasSubstr("No such file or directory")));
  EXPECT_THAT([=] { SYSCALL_CHECK(open_nonexistent_file(), "open failed.", -1); },
              ThrowsMessage<kvikio::GenericSystemError>(HasSubstr("No such file or directory")));
}

TEST(ErrorTest, syscall_check_for_voidp_return_value)
{
  auto map_anonymous_with_0_length = []() -> auto {
    std::size_t length = 0;
    int fd             = -1;
    off_t offset       = 0;
    return mmap(nullptr, length, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, fd, offset);
  };

  // If the mapping fails, mmap() returns MAP_FAILED i.e. (void*)-1, and the error number is EINVAL
  // (invalid argument).
  EXPECT_THAT([=] { SYSCALL_CHECK(map_anonymous_with_0_length(), "mmap failed.", MAP_FAILED); },
              ThrowsMessage<kvikio::GenericSystemError>(HasSubstr("Invalid argument")));
}
