/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

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

}  // namespace kvikio
