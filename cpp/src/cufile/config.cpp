/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdlib>
#include <filesystem>
#include <string>

#include <kvikio/cufile/config.hpp>

namespace kvikio {
namespace {

[[nodiscard]] char const* lookup_config_path()
{
  char const* env = std::getenv("CUFILE_ENV_PATH_JSON");
  if (env != nullptr && std::filesystem::exists(env)) { return env; }
  if (std::filesystem::exists("/etc/cufile.json")) { return "/etc/cufile.json"; }
  return "";
}

}  // namespace

std::string const& config_path()
{
  static std::string const ret = lookup_config_path();
  return ret;
}

}  // namespace kvikio
