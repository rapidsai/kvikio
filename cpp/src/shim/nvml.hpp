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
#pragma once

#include "nvml_h_wrapper.hpp"

#include <sstream>

#include <kvikio/shim/cuda.hpp>

/**
 * @brief Macro for checking the error code of the NVML API call.
 *
 * @param err_code The error code of the NVML API call.
 *
 * @throws std::runtime_error if the NVML API call fails.
 */
#define CHECK_NVML(err_code)                          \
  do {                                                \
    kvikio::check_nvml(err_code, __FILE__, __LINE__); \
  } while (0)

namespace kvikio {

/**
 * @brief Singleton class to manage dynamic loading of the NVML library.
 *
 * @note NVML initialization is costly, and is performed in the constructor of this singleton class.
 * NVML shutdown is not performed in the destructor, but a wrapper is provided anyway to be
 * explicitly called on users' discretion.
 */
class NvmlAPI {
 public:
  NvmlAPI(NvmlAPI const&)            = delete;
  NvmlAPI(NvmlAPI&&)                 = delete;
  NvmlAPI& operator=(NvmlAPI const&) = delete;
  NvmlAPI& operator=(NvmlAPI&&)      = delete;

  /**
   * @brief Get the NvmlAPI singleton instance
   *
   * @return The NvmlAPI singleton instance
   *
   * @throws std::runtime_error if the NVML shared library exists but the symbols fail to load.
   */
  static NvmlAPI& instance();

  /**
   * @brief Call nvmlShutdown
   *
   * @throws std::runtime_error if the shutdown is not successful.
   */
  void shutdown();

  decltype(&nvmlInit_v2) Init{nullptr};
  decltype(&nvmlShutdown) Shutdown{nullptr};
  decltype(&nvmlErrorString) ErrorString{nullptr};
  decltype(&nvmlDeviceGetHandleByIndex_v2) DeviceGetHandleByIndex{nullptr};
  decltype(&nvmlDeviceGetFieldValues) DeviceGetFieldValues{nullptr};
  decltype(&nvmlDeviceGetHandleByUUID) DeviceGetHandleByUUID{nullptr};

 private:
  NvmlAPI();
};

/**
 * @brief Helper function to check the error code of the NVML API call.
 *
 * @param err_code The error code of the NVML API call.
 * @param file The source file name where the NVML API call fails.
 * @param line The line number where the NVML API call fails.
 *
 * @throws std::runtime_error if the NVML API call fails.
 */
inline void check_nvml(nvmlReturn_t err_code, const char* file, int line)
{
  if (err_code == nvmlReturn_t::NVML_SUCCESS) { return; }
  std::stringstream ss;
  ss << "NVML error: " << err_code << " " << NvmlAPI::instance().ErrorString(err_code) << " in "
     << file << " at line " << line << std::endl;
  throw std::runtime_error(ss.str());
}

/**
 * @brief Given a CUDA device handle, get its corresponding NVML device handle.
 *
 * On a multi-GPU system, CUDA and NVML enumerate devices in different ways. This utility function
 * performs the CUDA-to-NVML conversion on the device handle. The conversion is implemented using
 * the invariant UUID, which is the most recommended. The other, less optimal way would be using
 * PCIe bus.
 *
 * @param cuda_device_handle CUDA device handle
 * @return NVML device handle
 */
nvmlDevice_t convert_device_handle_from_cuda_to_nvml(CUdevice cuda_device_handle);

}  // namespace kvikio
