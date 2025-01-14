/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

 private:
  cudaAPI();

 public:
  cudaAPI(cudaAPI const&)        = delete;
  void operator=(cudaAPI const&) = delete;

  KVIKIO_EXPORT static cudaAPI& instance();
};

/**
 * @brief Check if the CUDA library is available
 *
 * Notice, this doesn't check if the runtime environment supports CUDA.
 *
 * @return The boolean answer
 */
#ifdef KVIKIO_CUDA_FOUND
bool is_cuda_available();
#else
constexpr bool is_cuda_available() { return false; }
#endif

}  // namespace kvikio
