/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <cuda.h>

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
  decltype(cuMemcpyHtoD)* MemcpyHtoD{nullptr};
  decltype(cuMemcpyDtoH)* MemcpyDtoH{nullptr};
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

 private:
  cudaAPI()
  {
    void* lib = load_library("libcuda.so.1");
    // Notice, the API version loaded must match the version used downstream. That is,
    // if a project uses the `_v2` CUDA Driver API or the newest Runtime API, the symbols
    // loaded should also be the `_v2` symbols. Thus, we use KVIKIO_STRINGIFY() to get
    // the name of the symbol through cude.h.
    get_symbol(MemHostAlloc, lib, KVIKIO_STRINGIFY(cuMemHostAlloc));
    get_symbol(MemFreeHost, lib, KVIKIO_STRINGIFY(cuMemFreeHost));
    get_symbol(MemcpyHtoD, lib, KVIKIO_STRINGIFY(cuMemcpyHtoD));
    get_symbol(MemcpyDtoH, lib, KVIKIO_STRINGIFY(cuMemcpyDtoH));
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
  }

 public:
  cudaAPI(cudaAPI const&)        = delete;
  void operator=(cudaAPI const&) = delete;

  static cudaAPI& instance()
  {
    static cudaAPI _instance;
    return _instance;
  }
};

}  // namespace kvikio
