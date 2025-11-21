/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <dlfcn.h>
#include <sys/utsname.h>
#include <filesystem>
#include <sstream>
#include <vector>

#include <kvikio/error.hpp>
#include <kvikio/shim/utils.hpp>

namespace kvikio {

void* load_library(std::string const& name, int mode)
{
  ::dlerror();  // Clear old errors
  void* ret = ::dlopen(name.c_str(), mode);
  KVIKIO_EXPECT(ret != nullptr, ::dlerror(), std::runtime_error);
  return ret;
}

bool is_running_in_wsl() noexcept
{
  try {
    struct utsname buf{};
    int err = ::uname(&buf);
    if (err == 0) {
      std::string const name(static_cast<char*>(buf.release));
      // 'Microsoft' for WSL1 and 'microsoft' for WSL2
      return name.find("icrosoft") != std::string::npos;
    }
    return false;
  } catch (...) {
    return false;
  }
}

bool run_udev_readable() noexcept
{
  try {
    return std::filesystem::is_directory("/run/udev");
  } catch (...) {
    return false;
  }
}

}  // namespace kvikio
