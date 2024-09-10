/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include <kvikio/shim/cuda_h_wrapper.hpp>
#include <kvikio/shim/utils.hpp>

namespace kvikio {

/**
 * @brief Shim layer of the cuda C-API
 *
 * This is a singleton class that use `dlopen` on construction to load the C-API of cuda.
 *
 * For example, `cudaAPI::instance().MemHostAlloc()` corresponds to calling `cuMemHostAlloc()`
 */
class cudaAPI {
 public:
  decltype(cuInit)* Init{nullptr};
  decltype(cuMemHostAlloc)* MemHostAlloc{nullptr};
  decltype(cuMemFreeHost)* MemFreeHost{nullptr};
  decltype(cuMemcpyHtoDAsync)* MemcpyHtoDAsync{nullptr};
  decltype(cuMemcpyDtoHAsync)* MemcpyDtoHAsync{nullptr};
  decltype(cuPointerGetAttribute)* PointerGetAttribute{nullptr};
  decltype(cuPointerGetAttributes)* PointerGetAttributes{nullptr};
  decltype(cuCtxPushCurrent)* CtxPushCurrent{nullptr};
  decltype(cuCtxPopCurrent)* CtxPopCurrent{nullptr};
  decltype(cuCtxGetCurrent)* CtxGetCurrent{nullptr};
  decltype(cuMemGetAddressRange)* MemGetAddressRange{nullptr};
  decltype(cuGetErrorName)* GetErrorName{nullptr};
  decltype(cuGetErrorString)* GetErrorString{nullptr};
  decltype(cuDeviceGet)* DeviceGet{nullptr};
  decltype(cuDevicePrimaryCtxRetain)* DevicePrimaryCtxRetain{nullptr};
  decltype(cuDevicePrimaryCtxRelease)* DevicePrimaryCtxRelease{nullptr};
  decltype(cuStreamSynchronize)* StreamSynchronize{nullptr};
  decltype(cuStreamCreate)* StreamCreate{nullptr};
  decltype(cuStreamDestroy)* StreamDestroy{nullptr};
  decltype(cuCtxGetDevice)* CtxGetDevice{nullptr};
  decltype(cuDevicePrimaryCtxGetState)* DevicePrimaryCtxGetState{nullptr};

 private:
#ifdef KVIKIO_CUDA_FOUND
  cudaAPI()
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
    get_symbol(MemGetAddressRange, lib, KVIKIO_STRINGIFY(cuMemGetAddressRange));
    get_symbol(GetErrorName, lib, KVIKIO_STRINGIFY(cuGetErrorName));
    get_symbol(GetErrorString, lib, KVIKIO_STRINGIFY(cuGetErrorString));
    get_symbol(DeviceGet, lib, KVIKIO_STRINGIFY(cuDeviceGet));
    get_symbol(DevicePrimaryCtxRetain, lib, KVIKIO_STRINGIFY(cuDevicePrimaryCtxRetain));
    get_symbol(DevicePrimaryCtxRelease, lib, KVIKIO_STRINGIFY(cuDevicePrimaryCtxRelease));
    get_symbol(StreamSynchronize, lib, KVIKIO_STRINGIFY(cuStreamSynchronize));
    get_symbol(StreamCreate, lib, KVIKIO_STRINGIFY(cuStreamCreate));
    get_symbol(StreamDestroy, lib, KVIKIO_STRINGIFY(cuStreamDestroy));
    get_symbol(CtxGetDevice, lib, KVIKIO_STRINGIFY(cuCtxGetDevice));
    get_symbol(DevicePrimaryCtxGetState, lib, KVIKIO_STRINGIFY(cuDevicePrimaryCtxGetState));
  }
#else
  cudaAPI() { throw std::runtime_error("KvikIO not compiled with CUDA support"); }
#endif

 public:
  cudaAPI(cudaAPI const&)        = delete;
  void operator=(cudaAPI const&) = delete;

  static cudaAPI& instance()
  {
    static cudaAPI _instance;
    return _instance;
  }
};

/**
 * @brief Check if the CUDA library is available
 *
 * Notice, this doesn't check if the runtime environment supports CUDA.
 *
 * @return The boolean answer
 */
#ifdef KVIKIO_CUDA_FOUND
inline bool is_cuda_available()
{
  try {
    cudaAPI::instance();
  } catch (const std::runtime_error&) {
    return false;
  }
  return true;
}
#else
constexpr bool is_cuda_available() { return false; }
#endif

}  // namespace kvikio
