/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <cstddef>
#include <iostream>
#include <utility>

#include <cuda.h>
#include <nvml.h>

#include <kvikio/error.hpp>

namespace kvikio {

class NVML {
 private:
  bool _initialized{false};

  // Because Cython does not handle exceptions in the default
  // constructor, we initialize lazily.
  void lazy_init()
  {
    if (!_initialized) {
      _initialized = true;
      CUDA_TRY(cuInit(0));
      NVML_TRY(nvmlInit());
    }
  }

 public:
  [[nodiscard]] nvmlDevice_t get_current_nvml_device()
  {
    lazy_init();
    CUdevice cu_dev{};
    CUDA_TRY(cuCtxGetDevice(&cu_dev));
    char pciBusId[15];
    CUDA_TRY(cuDeviceGetPCIBusId(pciBusId, 15, cu_dev));
    nvmlDevice_t ret{};
    NVML_TRY(nvmlDeviceGetHandleByPciBusId(pciBusId, &ret));
    return ret;
  }

  NVML() = default;

  NVML(NVML const&) = delete;
  NVML& operator=(NVML const&) = delete;
  NVML(NVML&&) noexcept        = delete;
  NVML& operator=(NVML&&) noexcept = delete;

  ~NVML()
  {
    if (_initialized) {
      _initialized = false;
      try {
        NVML_TRY(nvmlShutdown());
      } catch (const CUfileException& e) {
        std::cerr << "Unable to close NVML: ";
        std::cerr << e.what();
        std::cerr << std::endl;
      }
    }
  }

  std::string get_name(nvmlDevice_t device)
  {
    lazy_init();
    char name[NVML_DEVICE_NAME_V2_BUFFER_SIZE];
    NVML_TRY(nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_V2_BUFFER_SIZE));
    return std::string{name};
  }
  std::string get_name() { return get_name(get_current_nvml_device()); }

  std::pair<std::size_t, std::size_t> get_memory(nvmlDevice_t device)
  {
    lazy_init();
    nvmlMemory_t info{};
    NVML_TRY(nvmlDeviceGetMemoryInfo(device, &info));
    return std::make_pair(static_cast<std::size_t>(info.total),
                          static_cast<std::size_t>(info.used));
  };
  std::pair<std::size_t, std::size_t> get_memory()
  {
    return get_memory(get_current_nvml_device());
  };

  std::pair<std::size_t, std::size_t> get_bar1_memory(nvmlDevice_t device)
  {
    lazy_init();
    nvmlBAR1Memory_t info{};
    NVML_TRY(nvmlDeviceGetBAR1MemoryInfo(device, &info));
    return std::make_pair(static_cast<std::size_t>(info.bar1Total),
                          static_cast<std::size_t>(info.bar1Used));
  };
  std::pair<std::size_t, std::size_t> get_bar1_memory()
  {
    return get_bar1_memory(get_current_nvml_device());
  };
};

}  // namespace kvikio
