/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>
#include <sstream>

#include <kvikio/error.hpp>

namespace kvikio {

GenericSystemError::GenericSystemError(int err_code, std::string const& msg)
  : std::system_error(err_code, std::generic_category(), msg)
{
}

GenericSystemError::GenericSystemError(int err_code, char const* msg)
  : std::system_error(err_code, std::generic_category(), msg)
{
}

GenericSystemError::GenericSystemError(std::string const& msg) : GenericSystemError(errno, msg) {}

GenericSystemError::GenericSystemError(char const* msg) : GenericSystemError(errno, msg) {}

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
