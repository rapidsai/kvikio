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

#include <unistd.h>
#include <cstring>
#include <iostream>
#include <map>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <type_traits>

#include <kvikio/error.hpp>
#include <kvikio/shim/cuda.hpp>
#include <kvikio/utils.hpp>
#include "shim/nvml.hpp"

namespace kvikio {

std::size_t get_page_size()
{
  static auto const page_size = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
  return page_size;
}

off_t convert_size2off(std::size_t x)
{
  KVIKIO_EXPECT(x < static_cast<std::size_t>(std::numeric_limits<off_t>::max()),
                "size_t argument too large to fit off_t");
  return static_cast<off_t>(x);
}

ssize_t convert_size2ssize(std::size_t x)
{
  KVIKIO_EXPECT(x < static_cast<std::size_t>(std::numeric_limits<ssize_t>::max()),
                "size_t argument too large to fit ssize_t");
  return static_cast<ssize_t>(x);
}

CUdeviceptr convert_void2deviceptr(void const* devPtr)
{
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
  return reinterpret_cast<CUdeviceptr>(devPtr);
}

#ifdef KVIKIO_CUDA_FOUND
bool is_host_memory(void const* ptr)
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
#endif

int get_device_ordinal_from_pointer(CUdeviceptr dev_ptr)
{
  int ret = 0;
  CUDA_DRIVER_TRY(
    cudaAPI::instance().PointerGetAttribute(&ret, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, dev_ptr));
  return ret;
}

CUcontext get_primary_cuda_context(int ordinal)
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

std::optional<CUcontext> get_context_associated_pointer(CUdeviceptr dev_ptr)
{
  CUcontext ctx = nullptr;
  CUresult const err =
    cudaAPI::instance().PointerGetAttribute(&ctx, CU_POINTER_ATTRIBUTE_CONTEXT, dev_ptr);
  if (err == CUDA_SUCCESS && ctx != nullptr) { return ctx; }
  if (err != CUDA_ERROR_INVALID_VALUE) { CUDA_DRIVER_TRY(err); }
  return {};
}

bool current_context_can_access_pointer(CUdeviceptr dev_ptr)
{
  CUdeviceptr current_ctx_dev_ptr{};
  CUresult const err = cudaAPI::instance().PointerGetAttribute(
    &current_ctx_dev_ptr, CU_POINTER_ATTRIBUTE_DEVICE_POINTER, dev_ptr);
  if (err == CUDA_SUCCESS && current_ctx_dev_ptr == dev_ptr) { return true; }
  if (err != CUDA_ERROR_INVALID_VALUE) { CUDA_DRIVER_TRY(err); }
  return false;
}

CUcontext get_context_from_pointer(void const* devPtr)
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

PushAndPopContext::PushAndPopContext(CUcontext ctx) : _ctx{ctx}
{
  CUDA_DRIVER_TRY(cudaAPI::instance().CtxPushCurrent(_ctx));
}

PushAndPopContext::~PushAndPopContext()
{
  try {
    CUDA_DRIVER_TRY(cudaAPI::instance().CtxPopCurrent(&_ctx), CUfileException);
  } catch (CUfileException const& e) {
    std::cerr << e.what() << std::endl;
  }
}

std::tuple<void*, std::size_t, std::size_t> get_alloc_info(void const* devPtr, CUcontext* ctx)
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

#ifdef KVIKIO_CUDA_FOUND
bool is_c2c_available(int device_idx, DeviceIdType device_id_type)
{
  // todo: Remove the version checking once CUDA 11 support is dropped
  // Version format: 1000 * major + 10 * minor
  int cuda_driver_version{};
  cudaAPI::instance().DriverGetVersion(&cuda_driver_version);
  if (cuda_driver_version <= 12000) { return false; }

  nvmlDevice_t device_handle_nvml{};
  if (device_id_type == DeviceIdType::CUDA) {
    CUdevice device_handle_cuda{};
    cudaAPI::instance().CtxGetDevice(&device_handle_cuda);
    device_handle_nvml = convert_device_handle_from_cuda_to_nvml(device_handle_cuda);
  } else {
    CHECK_NVML(NvmlAPI::instance().DeviceGetHandleByIndex(device_idx, &device_handle_nvml));
  }

  nvmlFieldValue_t field{};
  field.fieldId = NVML_FI_DEV_C2C_LINK_COUNT;
  CHECK_NVML(NvmlAPI::instance().DeviceGetFieldValues(device_handle_nvml, 1, &field));

  return (field.nvmlReturn == nvmlReturn_t::NVML_SUCCESS) && (field.value.uiVal > 0);
}
#endif

}  // namespace kvikio
