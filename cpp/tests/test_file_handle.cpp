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

 public:
  std::string _mocked_file_path{"mocked_file_path"};
  std::string _mocked_config_path{"mocked_config_path"};
  int _mocked_fd_direct_off{1000};
  int _mocked_fd_direct_on{2000};

  using DependencyMock = ::testing::NiceMock<mock::FileHandleDependencyMock>;
};

// Mock the simplest case where the desired compatibility mode is ON.
TEST_F(FileHandleTest, compat_mode_on)
{
  CompatMode desired_compat_mode{CompatMode::ON};
  auto dep_mock = std::make_unique<DependencyMock>();

  ON_CALL(*dep_mock, is_compat_mode_preferred()).WillByDefault(::testing::Return(true));
  ON_CALL(*dep_mock, is_compat_mode_preferred(::testing::_)).WillByDefault(::testing::Return(true));
  ON_CALL(*dep_mock, is_stream_api_available).WillByDefault(::testing::Return(true));
  ON_CALL(*dep_mock, config_path).WillByDefault(::testing::ReturnRef(_mocked_config_path));

  EXPECT_CALL(*dep_mock, open_fd).Times(1).WillOnce(::testing::Return(_mocked_fd_direct_off));
  EXPECT_CALL(*dep_mock, close_fd).Times(1);
  // Expect the cuFile I/O path to be not taken.
  EXPECT_CALL(*dep_mock, cuFile_handle_register).Times(0);
  EXPECT_CALL(*dep_mock, cuFile_handle_deregister).Times(0);
  EXPECT_CALL(*dep_mock, is_stream_api_available).Times(0);
  EXPECT_CALL(*dep_mock, config_path).Times(0);

  {
    FileHandle file_handle(
      _mocked_file_path, "w", FileHandle::m644, desired_compat_mode, std::move(dep_mock));

    EXPECT_TRUE(file_handle.is_compat_mode_preferred_for_async(desired_compat_mode));
    EXPECT_EQ(file_handle.fd(), _mocked_fd_direct_off);
  }
}

// Mock the case where the desired compatibility mode is OFF. The file cannot be opened with
// O_DIRECT flag, and an exception is thrown.
TEST_F(FileHandleTest, compat_mode_off_1)
{
  CompatMode desired_compat_mode{CompatMode::OFF};
  auto dep_mock = std::make_unique<DependencyMock>();

  ON_CALL(*dep_mock, is_compat_mode_preferred()).WillByDefault(::testing::Return(false));

  // Two calls to open_fd, without and with O_DIRECT flag. Let the latter case throw an exception.
  EXPECT_CALL(*dep_mock, open_fd)
    .Times(2)
    .WillOnce(::testing::Return(_mocked_fd_direct_off))
    .WillOnce(
      ::testing::Throw(std::system_error(errno, std::generic_category(), "Unable to open file")));
  EXPECT_CALL(*dep_mock, cuFile_handle_register).Times(0);
  EXPECT_CALL(*dep_mock, cuFile_handle_deregister).Times(0);

  // TODO: Currently there will be a file resource leak if this case happens.
  // See https://github.com/rapidsai/kvikio/issues/607
  // EXPECT_CALL(*dep_mock, close_fd).Times(1);

  EXPECT_THROW(
    {
      FileHandle file_handle(
        _mocked_file_path, "w", FileHandle::m644, desired_compat_mode, std::move(dep_mock));
    },
    std::system_error);
}

// Mock the case where the desired compatibility mode is OFF. The file cannot be
// opened with O_DIRECT flag, and an exception is thrown.
TEST_F(FileHandleTest, compat_mode_off_2) {}

// Mock the case where the desired compatibility mode is AUTO. The file cannot be opened with
// O_DIRECT flag, and the compatibility mode is adjusted to ON.
TEST_F(FileHandleTest, compat_mode_auto_1)
{
  CompatMode desired_compat_mode{CompatMode::AUTO};
  auto dep_mock = std::make_unique<DependencyMock>();

  EXPECT_CALL(*dep_mock, is_compat_mode_preferred())
    .WillOnce(::testing::Return(false))
    .WillRepeatedly(::testing::Return(true));
  // Two calls to open_fd, without and with O_DIRECT flag. Let the latter case throw an exception.
  EXPECT_CALL(*dep_mock, open_fd)
    .Times(2)
    .WillOnce(::testing::Return(_mocked_fd_direct_off))
    .WillOnce(
      ::testing::Throw(std::system_error(errno, std::generic_category(), "Unable to open file")));
  EXPECT_CALL(*dep_mock, close_fd).Times(1);

  // Expect the cuFile I/O path to be not taken.
  EXPECT_CALL(*dep_mock, cuFile_handle_register).Times(0);
  EXPECT_CALL(*dep_mock, cuFile_handle_deregister).Times(0);

  EXPECT_NO_THROW({
    FileHandle file_handle(
      _mocked_file_path, "w", FileHandle::m644, desired_compat_mode, std::move(dep_mock));
  });
}

}  // namespace kvikio::test
