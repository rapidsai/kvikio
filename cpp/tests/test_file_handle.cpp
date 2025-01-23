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

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "gmock/gmock.h"
#include "kvikio/defaults.hpp"
#include "kvikio/file_handle.hpp"
#include "mock/mock_file_handle_dep.hpp"
#include "utils.hpp"

namespace kvikio::test {

class FileHandleTest : public testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}

  std::string _file_path{"mocked_file_path"};
  std::string _config_path{"mocked_config_path"};
  int _fd_direct_off{1000};
  int _fd_direct_on{2000};

  using DependencyMock = ::testing::NiceMock<mock::FileHandleDependencyMock>;
};

// Mock the simplest case where FileHandle constructor receives argument CompatMode::ON.
TEST_F(FileHandleTest, compat_mode_on)
{
  CompatMode input_compat_mode{CompatMode::ON};
  auto dep_mock = std::make_unique<DependencyMock>();

  ON_CALL(*dep_mock, is_compat_mode_preferred()).WillByDefault(::testing::Return(true));
  ON_CALL(*dep_mock, is_compat_mode_preferred(::testing::_)).WillByDefault(::testing::Return(true));
  ON_CALL(*dep_mock, is_stream_api_available).WillByDefault(::testing::Return(true));
  ON_CALL(*dep_mock, config_path).WillByDefault(::testing::ReturnRef(_config_path));

  EXPECT_CALL(*dep_mock, open_fd).Times(1).WillOnce(::testing::Return(_fd_direct_off));
  EXPECT_CALL(*dep_mock, close_fd).Times(1);
  EXPECT_CALL(*dep_mock, cuFile_handle_register).Times(0);

  {
    FileHandle file_handle(
      _file_path, "w", FileHandle::m644, input_compat_mode, std::move(dep_mock));

    EXPECT_TRUE(file_handle.is_compat_mode_preferred_for_async(input_compat_mode));
    EXPECT_EQ(file_handle.fd(), _fd_direct_off);
  }
}

// Mock the case where FileHandle constructor receives argument CompatMode::OFF. The file cannot be
// opened with O_DIRECT flag, and an exception is thrown.
TEST_F(FileHandleTest, compat_mode_off_1)
{
  CompatMode input_compat_mode{CompatMode::OFF};
  auto dep_mock = std::make_unique<DependencyMock>();

  ON_CALL(*dep_mock, is_compat_mode_preferred()).WillByDefault(::testing::Return(false));
  EXPECT_CALL(*dep_mock, open_fd)
    .Times(2)
    .WillOnce(::testing::Return(_fd_direct_off))
    .WillOnce(
      ::testing::Throw(std::system_error(errno, std::generic_category(), "Unable to open file")));
  EXPECT_CALL(*dep_mock, close_fd).Times(0);
  EXPECT_CALL(*dep_mock, cuFile_handle_register).Times(0);
  EXPECT_CALL(*dep_mock, cuFile_handle_deregister).Times(0);

  EXPECT_THROW(
    {
      FileHandle file_handle(
        _file_path, "w", FileHandle::m644, input_compat_mode, std::move(dep_mock));
    },
    std::system_error);
}

// Mock the case where FileHandle constructor receives argument CompatMode::AUTO. The file cannot be
// opened with O_DIRECT flag, and the compatibility mode is set to ON.
TEST_F(FileHandleTest, compat_mode_auto_1)
{
  CompatMode input_compat_mode{CompatMode::AUTO};
  auto dep_mock = std::make_unique<DependencyMock>();

  EXPECT_CALL(*dep_mock, is_compat_mode_preferred())
    .Times(::testing::AtLeast(2))
    .WillOnce(::testing::Return(false))
    .WillRepeatedly(::testing::Return(true));
  EXPECT_CALL(*dep_mock, open_fd)
    .Times(2)
    .WillOnce(::testing::Return(_fd_direct_off))
    .WillOnce(
      ::testing::Throw(std::system_error(errno, std::generic_category(), "Unable to open file")));
  EXPECT_CALL(*dep_mock, cuFile_handle_register).Times(0);
  EXPECT_CALL(*dep_mock, close_fd).Times(1);

  EXPECT_NO_THROW({
    FileHandle file_handle(
      _file_path, "w", FileHandle::m644, input_compat_mode, std::move(dep_mock));
  });
}

}  // namespace kvikio::test
