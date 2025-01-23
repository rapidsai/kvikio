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
#pragma once

#include <sys/types.h>
#include <string>

#include <gmock/gmock.h>
#include <kvikio/defaults.hpp>
#include <kvikio/detail/file_handle_dep.hpp>
#include <kvikio/shim/cufile.hpp>

namespace kvikio {
namespace mock {

class FileHandleDependencyMock : public detail::FileHandleDependencyBase {
 public:
  MOCK_METHOD(int,
              open_fd,
              (const std::string& file_path, const std::string& flags, bool o_direct, mode_t mode),
              (override));
  MOCK_METHOD(void, close_fd, (int fd), (override));
  MOCK_METHOD(CUfileError_t,
              cuFile_handle_register,
              (CUfileHandle_t * fh, CUfileDescr_t* descr),
              (override));
  MOCK_METHOD(void, cuFile_handle_deregister, (CUfileHandle_t fh), (override));
  MOCK_METHOD(bool, is_compat_mode_preferred, (), (const, noexcept, override));
  MOCK_METHOD(bool,
              is_compat_mode_preferred,
              (CompatMode compat_mode),
              (const, noexcept, override));
  MOCK_METHOD(bool, is_stream_api_available, (), (noexcept, override));
  MOCK_METHOD(const std::string&, config_path, (), (override));
};

}  // namespace mock
}  // namespace kvikio
