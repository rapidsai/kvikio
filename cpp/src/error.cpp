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

#include <iostream>
#include <sstream>

#include <kvikio/error.hpp>

namespace kvikio {

GenericSystemError::GenericSystemError(const std::string& msg) : GenericSystemError(msg.c_str()) {}

GenericSystemError::GenericSystemError(const char* msg)
  : std::system_error(errno, std::generic_category(), msg)
{
}

namespace detail {

void log_error(std::string_view err_msg, int line_number, char const* filename)
{
  std::cerr << "KvikIO error at: " << filename << ":" << line_number << ": " << err_msg << "\n";
}

void handle_linux_call_error(int line_number, char const* filename)
{
  auto err_name = strerrorname_np(errno);
  KVIKIO_EXPECT(err_name != nullptr, "Error name should not be null.");
  std::stringstream ss;
  ss << "Linux system/library function call error at: " << filename << ":" << line_number << ". "
     << err_name;

  // std::system_error::what() automatically contains the detailed error description
  // equivalent to calling strerrordesc_np(errno)
  throw kvikio::GenericSystemError(ss.str());
}

}  // namespace detail

}  // namespace kvikio
