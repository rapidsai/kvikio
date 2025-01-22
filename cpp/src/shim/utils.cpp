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

#include <dlfcn.h>
#include <sys/utsname.h>
#include <filesystem>
#include <sstream>
#include <vector>

#include <kvikio/shim/utils.hpp>

namespace kvikio {

void* load_library(const char* name, int mode)
{
  ::dlerror();  // Clear old errors
  void* ret = ::dlopen(name, mode);
  if (ret == nullptr) { throw std::runtime_error(::dlerror()); }
  return ret;
}

void* load_library(const std::vector<const char*>& names, int mode)
{
  std::stringstream ss;
  for (const char* name : names) {
    ss << name << " ";
    try {
      return load_library(name, mode);
    } catch (const std::runtime_error&) {
    }
  }
  throw std::runtime_error("cannot open shared object file, tried: " + ss.str());
}

bool is_running_in_wsl() noexcept
{
  try {
    struct utsname buf {};
    int err = ::uname(&buf);
    if (err == 0) {
      const std::string name(static_cast<char*>(buf.release));
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
