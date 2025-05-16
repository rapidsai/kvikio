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

#include "nvml.hpp"
#include <kvikio/error.hpp>
#include "kvikio/shim/cuda.hpp"

namespace kvikio {

#ifdef KVIKIO_CUDA_FOUND
NvmlAPI::NvmlAPI()
{
  auto lib_handle = load_library("libnvidia-ml.so.1");
  get_symbol(Init, lib_handle, "nvmlInit_v2");
  get_symbol(Shutdown, lib_handle, "nvmlShutdown");
  get_symbol(ErrorString, lib_handle, "nvmlErrorString");
  get_symbol(DeviceGetHandleByIndex, lib_handle, "nvmlDeviceGetHandleByIndex_v2");
  get_symbol(DeviceGetFieldValues, lib_handle, "nvmlDeviceGetFieldValues");
  CHECK_NVML(Init());
}
#else
NvmlAPI::NvmlAPI() { KVIKIO_FAIL("KvikIO not compiled with CUDA support.", std::runtime_error); }
#endif

NvmlAPI& NvmlAPI::instance()
{
  static NvmlAPI instance;
  return instance;
}

#ifdef KVIKIO_CUDA_FOUND
bool is_nvml_available()
{
  try {
    NvmlAPI::instance();
  } catch (std::runtime_error const&) {
    return false;
  }
  return true;
}
#endif

#ifdef KVIKIO_CUDA_FOUND
bool is_c2c_available()
{
  // todo: remove this once CUDA 11 support is dropped
#if CUDA_VERSION < 12000
  return false;
#else
  nvmlDevice_t device_handle_nvml{};
  CUdevice device_handle_cuda{};
  cudaAPI::instance().CtxGetDevice(&device_handle_cuda);
  CHECK_NVML(NvmlAPI::instance().DeviceGetHandleByIndex(device_handle_cuda, &device_handle_nvml));

  nvmlFieldValue_t field{};
  field.fieldId = NVML_FI_DEV_C2C_LINK_COUNT;
  CHECK_NVML(NvmlAPI::instance().DeviceGetFieldValues(device_handle_nvml, 1, &field));

  return (field.nvmlReturn == nvmlReturn_t::NVML_SUCCESS) && (field.value.uiVal > 0);
#endif
}
#endif

}  // namespace kvikio