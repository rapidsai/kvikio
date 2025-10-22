/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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

void handle_linux_call_error(int line_number, char const* filename, std::string_view extra_msg)
{
  std::stringstream ss;
  if (!extra_msg.empty()) { ss << extra_msg << " "; }
  ss << "Linux system/library function call error at: " << filename << ":" << line_number;

  // std::system_error::what() automatically contains the detailed error description
  // equivalent to calling strerrordesc_np(errno)
  throw kvikio::GenericSystemError(ss.str());
}

}  // namespace detail

}  // namespace kvikio
