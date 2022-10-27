/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <dlfcn.h>
#include <sys/utsname.h>
#include <filesystem>
#include <sstream>
#include <vector>

namespace kvikio {

#define KVIKIO_STRINGIFY_DETAIL(x) #x
#define KVIKIO_STRINGIFY(x)        KVIKIO_STRINGIFY_DETAIL(x)

/**
 * @brief Load shared library
 *
 * @param name Name of the library to load.
 * @return The library handle.
 */
inline void* load_library(const char* name, int mode = RTLD_LAZY | RTLD_LOCAL | RTLD_NODELETE)
{
  ::dlerror();  // Clear old errors
  void* ret = ::dlopen(name, mode);
  if (ret == nullptr) { throw std::runtime_error(::dlerror()); }
  return ret;
}

/**
 * @brief Load shared library
 *
 * @param names Vector of names to try when loading shared library.
 * @return The library handle.
 */
inline void* load_library(const std::vector<const char*>& names,
                          int mode = RTLD_LAZY | RTLD_LOCAL | RTLD_NODELETE)
{
  std::stringstream ss;
  for (const char* name : names) {
    ss << name << " ";
    try {
      return load_library(name);
    } catch (const std::runtime_error&) {
    }
  }
  throw std::runtime_error("cannot open shared object file, tried: " + ss.str());
}

/**
 * @brief Get symbol using `dlsym`
 *
 * @tparam T The type of the function pointer.
 * @param handle The function pointer (output).
 * @param lib The library handle returned by `dlopen`.
 * @param name Name of the symbol/function to load.
 */
template <typename T>
void get_symbol(T& handle, void* lib, const char* name)
{
  ::dlerror();  // Clear old errors
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
  handle          = reinterpret_cast<T>(::dlsym(lib, name));
  const char* err = ::dlerror();
  if (err != nullptr) { throw std::runtime_error(err); }
}

/**
 * @brief Try to detect if running in Windows Subsystem for Linux (WSL)
 *
 * When unable to determine environment, `false` is returned.
 *
 * @return The boolean answer
 */
[[nodiscard]] inline bool is_running_in_wsl()
{
  struct utsname buf {
  };
  int err = ::uname(&buf);
  if (err == 0) {
    const std::string name(static_cast<char*>(buf.release));
    // 'Microsoft' for WSL1 and 'microsoft' for WSL2
    return name.find("icrosoft") != std::string::npos;
  }
  return false;
}

/**
 * @brief Check if `/run/udev` is readable
 *
 * cuFile files with `internal error` when `/run/udev` isn't readable.
 * This typically happens when running inside a docker image not launched
 * with `--volume /run/udev:/run/udev:ro`.
 *
 * @return The boolean answer
 */
[[nodiscard]] inline bool run_udev_readable()
{
  try {
    return std::filesystem::is_directory("/run/udev");
  } catch (const std::filesystem::filesystem_error&) {
    return false;
  }
}

}  // namespace kvikio
