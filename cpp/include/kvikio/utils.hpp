/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

#include <chrono>
#include <cstring>
#include <future>
#include <iostream>
#include <map>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <type_traits>

#ifdef KVIKIO_CUDA_FOUND
#include <nvtx3/nvtx3.hpp>
#endif

#include <kvikio/error.hpp>
#include <kvikio/shim/cuda.hpp>

// Macros used for defining symbol visibility, only GLIBC is supported.
// Since KvikIO is header-only, we rely on the linker to disambiguate inline functions
// that have (or return) static references. To do this, the relevant function must have
// `__attribute__((visibility("default")))`. If not, then if KvikIO is used in two
// different DSOs, the function will appear twice, and there will be two static objects.
// See <https://github.com/rapidsai/kvikio/issues/442>.
#if (defined(__GNUC__) || defined(__clang__)) && !defined(__MINGW32__) && !defined(__MINGW64__)
#define KVIKIO_EXPORT __attribute__((visibility("default")))
#define KVIKIO_HIDDEN __attribute__((visibility("hidden")))
#else
#define KVIKIO_EXPORT
#define KVIKIO_HIDDEN
#endif

namespace kvikio {

// cuFile defines a page size to 4 KiB
inline constexpr std::size_t page_size = 4096;

[[nodiscard]] inline off_t convert_size2off(std::size_t x)
{
  if (x >= static_cast<std::size_t>(std::numeric_limits<off_t>::max())) {
    throw CUfileException("size_t argument too large to fit off_t");
  }
  return static_cast<off_t>(x);
}

[[nodiscard]] inline ssize_t convert_size2ssize(std::size_t x)
{
  if (x >= static_cast<std::size_t>(std::numeric_limits<ssize_t>::max())) {
    throw CUfileException("size_t argument too large to fit ssize_t");
  }
  return static_cast<ssize_t>(x);
}

[[nodiscard]] inline CUdeviceptr convert_void2deviceptr(const void* devPtr)
{
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
  return reinterpret_cast<CUdeviceptr>(devPtr);
}

/**
 * @brief Help function to convert value to 64 bit signed integer
 */
template <typename T, std::enable_if_t<std::is_integral_v<T>>* = nullptr>
[[nodiscard]] std::int64_t convert_to_64bit(T value)
{
  if constexpr (std::numeric_limits<T>::max() > std::numeric_limits<std::int64_t>::max()) {
    if (value > std::numeric_limits<std::int64_t>::max()) {
      throw std::overflow_error("convert_to_64bit(x): x too large to fit std::int64_t");
    }
  }
  return std::int64_t(value);
}

/**
 * @brief Help function to convert value to 64 bit float
 */
template <typename T, std::enable_if_t<std::is_floating_point_v<T>>* = nullptr>
[[nodiscard]] double convert_to_64bit(T value)
{
  return double(value);
}

/**
 * @brief Check if `ptr` points to host memory (as opposed to device memory)
 *
 * In this context, managed memory counts as device memory
 *
 * @param ptr Memory pointer to query
 * @return The boolean answer
 */
#ifdef KVIKIO_CUDA_FOUND
inline bool is_host_memory(const void* ptr)
{
  CUpointer_attribute attrs[1] = {
    CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
  };
  CUmemorytype memtype{};
  void* data[1] = {&memtype};
  CUresult result =
    cudaAPI::instance().PointerGetAttributes(1, attrs, data, convert_void2deviceptr(ptr));

  // We assume that `ptr` is host memory when CUDA_ERROR_NOT_INITIALIZED
  if (result == CUDA_ERROR_NOT_INITIALIZED) { return true; }
  CUDA_DRIVER_TRY(result);

  // Notice, queying `CU_POINTER_ATTRIBUTE_MEMORY_TYPE` returns zero when the memory
  // is unregistered host memory. This is undocumented but how the Runtime CUDA API
  // does it to support `cudaMemoryTypeUnregistered`.
  return memtype == 0 || memtype == CU_MEMORYTYPE_HOST;
}
#else
constexpr bool is_host_memory(const void* ptr) { return true; }
#endif

/**
 * @brief Return the device owning the pointer
 *
 * @param ptr Device pointer to query
 * @return The device ordinal
 */
[[nodiscard]] inline int get_device_ordinal_from_pointer(CUdeviceptr dev_ptr)
{
  int ret = 0;
  CUDA_DRIVER_TRY(
    cudaAPI::instance().PointerGetAttribute(&ret, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, dev_ptr));
  return ret;
}

/**
 * @brief Given a device ordinal, return the primary context of the device.
 *
 * This function caches the primary contexts retrieved until program exit
 *
 * @param ordinal Device ordinal - an integer between 0 and the number of CUDA devices
 * @return Primary CUDA context
 */
[[nodiscard]] KVIKIO_EXPORT inline CUcontext get_primary_cuda_context(int ordinal)
{
  static std::map<int, CUcontext> _cache;
  static std::mutex _mutex;
  std::lock_guard const lock(_mutex);

  if (_cache.find(ordinal) == _cache.end()) {
    CUdevice dev{};
    CUcontext ctx{};
    CUDA_DRIVER_TRY(cudaAPI::instance().DeviceGet(&dev, ordinal));

    // Notice, we let the primary context leak at program exit. We do this because `_cache`
    // is static and we are not allowed to call `cuDevicePrimaryCtxRelease()` after main:
    // <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#initialization>
    CUDA_DRIVER_TRY(cudaAPI::instance().DevicePrimaryCtxRetain(&ctx, dev));
    _cache.emplace(ordinal, ctx);
  }
  return _cache.at(ordinal);
}

/**
 * @brief Return the CUDA context associated the given device pointer, if any.
 *
 * @param dev_ptr Device pointer to query
 * @return Usable CUDA context, if one were found.
 */
[[nodiscard]] inline std::optional<CUcontext> get_context_associated_pointer(CUdeviceptr dev_ptr)
{
  CUcontext ctx = nullptr;
  const CUresult err =
    cudaAPI::instance().PointerGetAttribute(&ctx, CU_POINTER_ATTRIBUTE_CONTEXT, dev_ptr);
  if (err == CUDA_SUCCESS && ctx != nullptr) { return ctx; }
  if (err != CUDA_ERROR_INVALID_VALUE) { CUDA_DRIVER_TRY(err); }
  return {};
}

/**
 * @brief Check if the current CUDA context can access the given device pointer
 *
 * @param dev_ptr Device pointer to query
 * @return The boolean answer
 */
[[nodiscard]] inline bool current_context_can_access_pointer(CUdeviceptr dev_ptr)
{
  CUdeviceptr current_ctx_dev_ptr{};
  const CUresult err = cudaAPI::instance().PointerGetAttribute(
    &current_ctx_dev_ptr, CU_POINTER_ATTRIBUTE_DEVICE_POINTER, dev_ptr);
  if (err == CUDA_SUCCESS && current_ctx_dev_ptr == dev_ptr) { return true; }
  if (err != CUDA_ERROR_INVALID_VALUE) { CUDA_DRIVER_TRY(err); }
  return false;
}

/**
 * @brief Return a CUDA context that can be used with the given device pointer
 *
 * For robustness, we look for an usabale context in the following order:
 *   1) If a context has been associated with `devPtr`, it is returned.
 *   2) If the current context exists and can access `devPtr`, it is returned.
 *   3) Return the primary context of the device that owns `devPtr`. We assume the
 *      primary context can access `devPtr`, which might not be true in the exceptional
 *      disjoint addressing cases mention in the CUDA docs[1]. In these cases, the user
 *      has to set an usable current context before reading/writing using KvikIO.
 *
 * [1] <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__UNIFIED.html>
 *
 * @param devPtr Device pointer to query
 * @return Usable CUDA context
 */
[[nodiscard]] inline CUcontext get_context_from_pointer(const void* devPtr)
{
  CUdeviceptr dev_ptr = convert_void2deviceptr(devPtr);

  // First we check if a context has been associated with `devPtr`.
  {
    auto ctx = get_context_associated_pointer(dev_ptr);
    if (ctx.has_value()) { return ctx.value(); }
  }

  // If this isn't the case, we check the current context. If it exist and can access `devPtr`, we
  // return the current context.
  {
    CUcontext ctx = nullptr;
    CUDA_DRIVER_TRY(cudaAPI::instance().CtxGetCurrent(&ctx));
    if (ctx != nullptr && current_context_can_access_pointer(dev_ptr)) { return ctx; }
  }

  // Finally, if we didn't find any usable context, we return the primary context of the
  // device that owns `devPtr`. If the primary context cannot access `devPtr`, we accept failure.
  return get_primary_cuda_context(get_device_ordinal_from_pointer(dev_ptr));
}

/**
 * @brief Push CUDA context on creation and pop it on destruction
 */
class PushAndPopContext {
 private:
  CUcontext _ctx;

 public:
  PushAndPopContext(CUcontext ctx) : _ctx{ctx}
  {
    CUDA_DRIVER_TRY(cudaAPI::instance().CtxPushCurrent(_ctx));
  }
  PushAndPopContext(const PushAndPopContext&)            = delete;
  PushAndPopContext& operator=(PushAndPopContext const&) = delete;
  PushAndPopContext(PushAndPopContext&&)                 = delete;
  PushAndPopContext&& operator=(PushAndPopContext&&)     = delete;
  ~PushAndPopContext()
  {
    try {
      CUDA_DRIVER_TRY(cudaAPI::instance().CtxPopCurrent(&_ctx), CUfileException);
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
    _ctx = get_context_from_pointer(devPtr);
  }
  PushAndPopContext context(_ctx);
  CUDA_DRIVER_TRY(cudaAPI::instance().MemGetAddressRange(&base_ptr, &base_size, dev));
  std::size_t offset = dev - base_ptr;
  // NOLINTNEXTLINE(performance-no-int-to-ptr, cppcoreguidelines-pro-type-reinterpret-cast)
  return std::make_tuple(reinterpret_cast<void*>(base_ptr), base_size, offset);
}

template <typename T>
inline bool is_future_done(const T& future)
{
  return future.wait_for(std::chrono::seconds(0)) != std::future_status::timeout;
}

#ifdef KVIKIO_CUDA_FOUND
/**
 * @brief Tag type for libkvikio's NVTX domain.
 */
struct libkvikio_domain {
  static constexpr char const* name{"libkvikio"};
};

// Macro overloads of KVIKIO_NVTX_FUNC_RANGE
#define KVIKIO_NVTX_FUNC_RANGE_1() NVTX3_FUNC_RANGE_IN(libkvikio_domain)
#define KVIKIO_NVTX_FUNC_RANGE_2(msg, val)                    \
  nvtx3::scoped_range_in<libkvikio_domain> _kvikio_nvtx_range \
  {                                                           \
    nvtx3::event_attributes                                   \
    {                                                         \
      msg, nvtx3::payload { convert_to_64bit(val) }           \
    }                                                         \
  }
#define GET_KVIKIO_NVTX_FUNC_RANGE_MACRO(_1, _2, NAME, ...) NAME
#endif

/**
 * @brief Convenience macro for generating an NVTX range in the `libkvikio` domain
 * from the lifetime of a function.
 *
 * Takes two arguments (message, payload) or no arguments, in which case the name
 * of the immediately enclosing function returned by `__func__` is used.
 *
 * Example:
 * ```
 * void some_function1(){
 *    KVIKIO_NVTX_FUNC_RANGE("my function", 42);
 *    ...
 * }
 * void some_function2(){
 *    KVIKIO_NVTX_FUNC_RANGE();  // The name `some_function2` is used
 *    ...
 * }
 * ```
 */
#ifdef KVIKIO_CUDA_FOUND
#define KVIKIO_NVTX_FUNC_RANGE(...)                                  \
  GET_KVIKIO_NVTX_FUNC_RANGE_MACRO(                                  \
    __VA_ARGS__, KVIKIO_NVTX_FUNC_RANGE_2, KVIKIO_NVTX_FUNC_RANGE_1) \
  (__VA_ARGS__)
#else
#define KVIKIO_NVTX_FUNC_RANGE(...) \
  do {                              \
  } while (0)
#endif

}  // namespace kvikio
