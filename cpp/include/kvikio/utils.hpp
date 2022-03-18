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
#include <chrono>
#include <cstring>
#include <filesystem>
#include <future>
#include <iostream>
#include <tuple>

#include <cuda.h>

#include <cufile.h>
#include <kvikio/error.hpp>

namespace kvikio {

[[nodiscard]] inline off_t convert_size2off(std::size_t x)
{
  if (x >= std::numeric_limits<off_t>::max()) {
    throw CUfileException("size_t argument too large to fit off_t");
  }
  return static_cast<off_t>(x);
}

[[nodiscard]] inline CUdeviceptr convert_void2deviceptr(const void* devPtr)
{
  /*NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)*/
  return reinterpret_cast<CUdeviceptr>(devPtr);
}

[[nodiscard]] inline CUcontext get_context_from_device_pointer(const void* devPtr)
{
  CUcontext ctx{};
  auto dev = convert_void2deviceptr(devPtr);
  CUDA_DRIVER_TRY(cuPointerGetAttribute(&ctx, CU_POINTER_ATTRIBUTE_CONTEXT, dev));
  return ctx;
}

/**
 * @brief Push CUDA context on creation and pop it on destruction
 */
class PushAndPopContext {
 private:
  CUcontext _ctx;

 public:
  PushAndPopContext(CUcontext ctx) : _ctx{ctx} { CUDA_DRIVER_TRY(cuCtxPushCurrent(_ctx)); }
  PushAndPopContext(const void* devPtr) : _ctx{get_context_from_device_pointer(devPtr)}
  {
    CUDA_DRIVER_TRY(cuCtxPushCurrent(_ctx));
  }
  PushAndPopContext(const PushAndPopContext&) = delete;
  PushAndPopContext& operator=(PushAndPopContext const&) = delete;
  PushAndPopContext(PushAndPopContext&&)                 = delete;
  PushAndPopContext&& operator=(PushAndPopContext&&) = delete;
  ~PushAndPopContext()
  {
    try {
      CUDA_DRIVER_TRY(cuCtxPopCurrent(&_ctx), CUfileException);
    } catch (const CUfileException& e) {
      std::cerr << e.what() << std::endl;
    }
  }
};

// Find the base and offset of the memory allocation `devPtr` is in
inline std::tuple<void*, std::size_t, std::size_t> get_alloc_info(const void* devPtr,
                                                                  CUcontext* ctx = nullptr)
{
  auto dev = convert_void2deviceptr(devPtr);
  CUdeviceptr base_ptr{};
  std::size_t base_size{};
  CUcontext _ctx{};
  if (ctx != nullptr) {
    _ctx = *ctx;
  } else {
    _ctx = get_context_from_device_pointer(devPtr);
  }
  PushAndPopContext context(_ctx);
  CUDA_DRIVER_TRY(cuMemGetAddressRange(&base_ptr, &base_size, dev));
  std::size_t offset = dev - base_ptr;
  /*NOLINTNEXTLINE(performance-no-int-to-ptr, cppcoreguidelines-pro-type-reinterpret-cast)*/
  return std::make_tuple(reinterpret_cast<void*>(base_ptr), base_size, offset);
}

template <typename T>
inline bool is_future_done(const T& future)
{
  return future.wait_for(std::chrono::seconds(0)) != std::future_status::timeout;
}

/**
 * @brief Load shared library
 *
 * @param name Name of the library to load.
 * @return The library handle.
 */
void* load_library(const char* name, int mode = RTLD_LAZY | RTLD_LOCAL | RTLD_NODELETE)
{
  ::dlerror();  // Clear old errors
  void* ret = ::dlopen(name, mode);
  if (ret == nullptr) {
    throw CUfileException{std::string{__FILE__} + ":" + KVIKIO_STRINGIFY(__LINE__) + ": " +
                          ::dlerror()};
  }
  return ret;
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
  /*NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)*/
  handle          = reinterpret_cast<T>(::dlsym(lib, name));
  const char* err = ::dlerror();
  if (err != nullptr) {
    throw CUfileException{std::string{__FILE__} + ":" + KVIKIO_STRINGIFY(__LINE__) + ": " + err};
  }
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
  } catch (const std::filesystem::filesystem_error& e) {
    return false;
  }
}

}  // namespace kvikio
