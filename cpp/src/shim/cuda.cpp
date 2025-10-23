/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdexcept>

#include <kvikio/error.hpp>
#include <kvikio/shim/cuda.hpp>

namespace kvikio {

cudaAPI::cudaAPI()
{
  void* lib = load_library("libcuda.so.1");
  // Notice, the API version loaded must match the version used downstream. That is,
  // if a project uses the `_v2` CUDA Driver API or the newest Runtime API, the symbols
  // loaded should also be the `_v2` symbols. Thus, we use KVIKIO_STRINGIFY() to get
  // the name of the symbol through cude.h.
  get_symbol(MemHostAlloc, lib, KVIKIO_STRINGIFY(cuMemHostAlloc));
  get_symbol(MemFreeHost, lib, KVIKIO_STRINGIFY(cuMemFreeHost));
  get_symbol(MemcpyHtoDAsync, lib, KVIKIO_STRINGIFY(cuMemcpyHtoDAsync));
  get_symbol(MemcpyDtoHAsync, lib, KVIKIO_STRINGIFY(cuMemcpyDtoHAsync));
  get_symbol(PointerGetAttribute, lib, KVIKIO_STRINGIFY(cuPointerGetAttribute));
  get_symbol(PointerGetAttributes, lib, KVIKIO_STRINGIFY(cuPointerGetAttributes));
  get_symbol(CtxPushCurrent, lib, KVIKIO_STRINGIFY(cuCtxPushCurrent));
  get_symbol(CtxPopCurrent, lib, KVIKIO_STRINGIFY(cuCtxPopCurrent));
  get_symbol(CtxGetCurrent, lib, KVIKIO_STRINGIFY(cuCtxGetCurrent));
  get_symbol(CtxGetDevice, lib, KVIKIO_STRINGIFY(cuCtxGetDevice));
  get_symbol(MemGetAddressRange, lib, KVIKIO_STRINGIFY(cuMemGetAddressRange));
  get_symbol(GetErrorName, lib, KVIKIO_STRINGIFY(cuGetErrorName));
  get_symbol(GetErrorString, lib, KVIKIO_STRINGIFY(cuGetErrorString));
  get_symbol(DeviceGet, lib, KVIKIO_STRINGIFY(cuDeviceGet));
  get_symbol(DeviceGetCount, lib, KVIKIO_STRINGIFY(cuDeviceGetCount));
  get_symbol(DeviceGetAttribute, lib, KVIKIO_STRINGIFY(cuDeviceGetAttribute));
  get_symbol(DevicePrimaryCtxRetain, lib, KVIKIO_STRINGIFY(cuDevicePrimaryCtxRetain));
  get_symbol(DevicePrimaryCtxRelease, lib, KVIKIO_STRINGIFY(cuDevicePrimaryCtxRelease));
  get_symbol(StreamSynchronize, lib, KVIKIO_STRINGIFY(cuStreamSynchronize));
  get_symbol(StreamCreate, lib, KVIKIO_STRINGIFY(cuStreamCreate));
  get_symbol(StreamDestroy, lib, KVIKIO_STRINGIFY(cuStreamDestroy));
  get_symbol(DriverGetVersion, lib, KVIKIO_STRINGIFY(cuDriverGetVersion));

  CUDA_DRIVER_TRY(DriverGetVersion(&driver_version));

#if CUDA_VERSION >= 12080
  // cuMemcpyBatchAsync was introduced in CUDA 12.8, and its parameters were changed in CUDA 13.0.
  try {
    decltype(cuMemcpyBatchAsync)* fp;
    get_symbol(fp, lib, KVIKIO_STRINGIFY(cuMemcpyBatchAsync));
    MemcpyBatchAsync.set(fp);
  } catch (std::runtime_error const&) {
    // Rethrow the exception if the CUDA driver version at runtime is satisfied but
    // cuMemcpyBatchAsync is not found.
    if (driver_version >= 12080) { throw; }
    // If the CUDA driver version at runtime is not satisfied, reset the wrapper. At the call site,
    // use the conventional cuMemcpyXtoXAsync API as the fallback.
    MemcpyBatchAsync.reset();
  }
#endif
}

cudaAPI& cudaAPI::instance()
{
  static cudaAPI _instance;
  return _instance;
}

bool is_cuda_available()
{
  try {
    cudaAPI::instance();
  } catch (std::runtime_error const&) {
    return false;
  }
  return true;
}

}  // namespace kvikio
