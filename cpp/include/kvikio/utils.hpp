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

[[nodiscard]] inline off_t convert_size2ssize(std::size_t x)
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
 * @brief Get the current cuda context
 *
 * Previously, we got the cuda context from the provided device pointer by calling
 * `cuPointerGetAttribute(..., CU_POINTER_ATTRIBUTE_CONTEXT)`. However, this doesn't
 * work for stream ordered device memory allocations[1] so we now get the current
 * cuda context instead.
 * [1] <https://docs.nvidia.com/cuda/cuda-c-programming-guide/#pointer-attributes>
 *
 * @param check_owning_devPtr If not NULL, a device memory pointer that must have
 * been allocated by, mapped by, or registered with the current context. If this
 * isn't the case, a CUfileException is thrown.
 *
 * @return The current cuda context
 */
[[nodiscard]] inline CUcontext get_current_context(const void* check_owning_devPtr = nullptr)
{
  if (check_owning_devPtr != nullptr) {
    CUdeviceptr current_ctx_devPtr{};
    CUdeviceptr dev_ptr = convert_void2deviceptr(check_owning_devPtr);

    CUresult const err = cudaAPI::instance().PointerGetAttribute(
      &current_ctx_devPtr, CU_POINTER_ATTRIBUTE_DEVICE_POINTER, dev_ptr);
    if (err != CUDA_SUCCESS || current_ctx_devPtr != dev_ptr) {
      throw CUfileException("The current CUDA context must own the given device memory");
    }
  }

  CUcontext ret{};
  CUDA_DRIVER_TRY(cudaAPI::instance().CtxGetCurrent(&ret));
  return ret;
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
  PushAndPopContext(const PushAndPopContext&) = delete;
  PushAndPopContext& operator=(PushAndPopContext const&) = delete;
  PushAndPopContext(PushAndPopContext&&)                 = delete;
  PushAndPopContext&& operator=(PushAndPopContext&&) = delete;
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
    _ctx = get_current_context(devPtr);
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
