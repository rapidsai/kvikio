/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <unistd.h>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <optional>
#include <tuple>

#include <kvikio/detail/utils.hpp>
#include <kvikio/error.hpp>
#include <kvikio/shim/cuda.hpp>
#include <kvikio/utils.hpp>

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

namespace detail {

std::size_t align_up(std::size_t value, std::size_t alignment) noexcept
{
  return (value + alignment - 1) & ~(alignment - 1);
}

void* align_up(void* addr, std::size_t alignment) noexcept
{
  auto res = (reinterpret_cast<uintptr_t>(addr) + alignment - 1) & ~(alignment - 1);
  return reinterpret_cast<void*>(res);
}

std::size_t align_down(std::size_t value, std::size_t alignment) noexcept
{
  return value & ~(alignment - 1);
}

void* align_down(void* addr, std::size_t alignment) noexcept
{
  auto res = reinterpret_cast<uintptr_t>(addr) & ~(alignment - 1);
  return reinterpret_cast<void*>(res);
}

}  // namespace detail
}  // namespace kvikio
