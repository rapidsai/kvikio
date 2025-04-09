/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
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
#include <stdexcept>
#include <string>
#include <vector>

namespace kvikio {

// Macros used for defining symbol visibility.
// Since KvikIO declares global default values in headers, we rely on the linker to disambiguate
// inline and static methods that have (or return) static references. To do this, the relevant
// function/method must have `__attribute__((visibility("default")))`. If not, then if KvikIO is
// used in two different DSOs, the function will appear twice, and there will be two static objects.
// See <https://gcc.gnu.org/wiki/Visibility> and <https://github.com/rapidsai/kvikio/issues/442>.
#if (defined(__GNUC__) || defined(__clang__)) && !defined(__MINGW32__) && !defined(__MINGW64__)
#define KVIKIO_EXPORT __attribute__((visibility("default")))
#define KVIKIO_HIDDEN __attribute__((visibility("hidden")))
#else
#define KVIKIO_EXPORT
#define KVIKIO_HIDDEN
#endif

#define KVIKIO_STRINGIFY_DETAIL(x) #x
#define KVIKIO_STRINGIFY(x)        KVIKIO_STRINGIFY_DETAIL(x)

/**
 * @brief Load shared library
 *
 * @param name Name of the library to load.
 * @return The library handle.
 */
void* load_library(std::string const& name, int mode = RTLD_LAZY | RTLD_LOCAL | RTLD_NODELETE);

/**
 * @brief Load shared library
 *
 * @param names Vector of names to try when loading shared library.
 * @return The library handle.
 */
void* load_library(std::vector<std::string> const& names,
                   int mode = RTLD_LAZY | RTLD_LOCAL | RTLD_NODELETE);

/**
 * @brief Get symbol using `dlsym`
 *
 * @tparam T The type of the function pointer.
 * @param handle The function pointer (output).
 * @param lib The library handle returned by `dlopen`.
 * @param name Name of the symbol/function to load.
 */
template <typename T>
void get_symbol(T& handle, void* lib, std::string const& name)
{
  ::dlerror();  // Clear old errors
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
  handle          = reinterpret_cast<T>(::dlsym(lib, name.c_str()));
  char const* err = ::dlerror();
  if (err != nullptr) { throw std::runtime_error(err); }
}

/**
 * @brief Try to detect if running in Windows Subsystem for Linux (WSL)
 *
 * When unable to determine environment, `false` is returned.
 *
 * @return The boolean answer
 */
[[nodiscard]] bool is_running_in_wsl() noexcept;

/**
 * @brief Check if `/run/udev` is readable
 *
 * cuFile files with `internal error` when `/run/udev` isn't readable.
 * This typically happens when running inside a docker image not launched
 * with `--volume /run/udev:/run/udev:ro`.
 *
 * @return The boolean answer
 */
[[nodiscard]] bool run_udev_readable() noexcept;

}  // namespace kvikio
