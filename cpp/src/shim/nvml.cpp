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

#include <iomanip>
#include <iostream>
#include <sstream>

#include <kvikio/error.hpp>
#include <kvikio/shim/cuda.hpp>

namespace kvikio {

#ifdef KVIKIO_NVML_FOUND
NvmlAPI::NvmlAPI()
{
  auto lib_handle = load_library("libnvidia-ml.so.1");
  get_symbol(Init, lib_handle, "nvmlInit_v2");
  get_symbol(Shutdown, lib_handle, "nvmlShutdown");
  get_symbol(ErrorString, lib_handle, "nvmlErrorString");
  get_symbol(DeviceGetHandleByIndex, lib_handle, "nvmlDeviceGetHandleByIndex_v2");
  get_symbol(DeviceGetFieldValues, lib_handle, "nvmlDeviceGetFieldValues");
  get_symbol(DeviceGetHandleByUUID, lib_handle, "nvmlDeviceGetHandleByUUID");
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

#ifdef KVIKIO_NVML_FOUND
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

#ifdef KVIKIO_NVML_FOUND
nvmlDevice_t convert_device_handle_from_cuda_to_nvml(CUdevice cuda_device_handle)
{
  // CUDA UUID is a 16-byte array, e.g.: 0x0011 0x2233 0x4455 0x6677 0x8899 0xaabb 0xccdd 0xeeff.
  // There is no prefix to distinguish GPU from MIG.

  // NVML UUID is a string with prefix and hyphens.
  // e.g.: GPU-00112233-4455-6677-8899-aabbccddeeff.
  // The prefix can be GPU or MIG.

  CUuuid uuid{};
  CUDA_DRIVER_TRY(cudaAPI::instance().DeviceGetUuid(&uuid, cuda_device_handle));
  std::stringstream ss;
  for (std::size_t i = 0; i < sizeof(uuid.bytes); ++i) {
    if (i == 4 || i == 6 || i == 8 || i == 10) { ss << '-'; }
    ss << std::hex << std::setfill('0') << std::setw(2) << (static_cast<int>(uuid.bytes[i]) & 0xff);
  }

  nvmlDevice_t nvml_device_handle{};
  try {
    std::string const gpu_uuid = "GPU-" + ss.str();
    CHECK_NVML(NvmlAPI::instance().DeviceGetHandleByUUID(gpu_uuid.c_str(), &nvml_device_handle));
  } catch (...) {
    std::string const mig_uuid = "MIG-" + ss.str();
    CHECK_NVML(NvmlAPI::instance().DeviceGetHandleByUUID(mig_uuid.c_str(), &nvml_device_handle));
  }

  return nvml_device_handle;
}
#else
nvmlDevice_t convert_device_handle_from_cuda_to_nvml(CUdevice cuda_device_handle)
{
  KVIKIO_FAIL("KvikIO not compiled with CUDA support.", std::runtime_error);
  return nvmlDevice_t{};
}
#endif

}  // namespace kvikio
