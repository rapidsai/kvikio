/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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
#include <tuple>

#include <kvikio/error.hpp>
#include <kvikio/shim/cuda.hpp>

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
 * @brief Check if `ptr` points to host memory (as opposed to device memory)
 *
 * In this context, managed memory counts as device memory
 *
 * @param ptr Memory pointer to query
 * @return The boolean answer
 */
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
 * @brief RAII wrapper for a CUDA primary context
 */
class CudaPrimaryContext {
 public:
  CUdevice dev{};
  CUcontext ctx{};

  CudaPrimaryContext(int device_ordinal)
  {
    CUDA_DRIVER_TRY(cudaAPI::instance().DeviceGet(&dev, device_ordinal));
    CUDA_DRIVER_TRY(cudaAPI::instance().DevicePrimaryCtxRetain(&ctx, dev));
  }
  CudaPrimaryContext(const CudaPrimaryContext&)            = delete;
  CudaPrimaryContext& operator=(CudaPrimaryContext const&) = delete;
  CudaPrimaryContext(CudaPrimaryContext&&)                 = delete;
  CudaPrimaryContext&& operator=(CudaPrimaryContext&&)     = delete;
  ~CudaPrimaryContext()
  {
    try {
      CUDA_DRIVER_TRY(cudaAPI::instance().DevicePrimaryCtxRelease(dev), CUfileException);
    } catch (const CUfileException& e) {
      std::cerr << e.what() << std::endl;
    }
  }
};

/**
 * @brief Given a device ordinal, return the primary context of the device.
 *
 * This function caches the primary contexts retrieved until program exit
 *
 * @param ordinal Device ordinal - an integer between 0 and the number of CUDA devices
 * @return Primary CUDA context
 */
[[nodiscard]] inline CUcontext get_primary_cuda_context(int ordinal)
{
  static std::map<int, CudaPrimaryContext> _primary_contexts;
  _primary_contexts.try_emplace(ordinal, ordinal);
  return _primary_contexts.at(ordinal).ctx;
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

}  // namespace kvikio
