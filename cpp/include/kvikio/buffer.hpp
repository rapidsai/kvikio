/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

#include <algorithm>
#include <iostream>
#include <map>
#include <vector>

#include <kvikio/defaults.hpp>
#include <kvikio/error.hpp>
#include <kvikio/shim/cufile.hpp>
#include <kvikio/shim/cufile_h_wrapper.hpp>
#include <kvikio/utils.hpp>

namespace kvikio {

/**
 * @brief register an existing cudaMalloced memory with cuFile to pin for
 * GPUDirect Storage access.
 *
 * @param devPtr_base  device pointer to allocated
 * @param length  size of memory region from the above specified devPtr
 * @param flags   should be zero or `CU_FILE_RDMA_REGISTER` (experimental)
 * @param errors_to_ignore  CuFile errors to ignore such as `CU_FILE_MEMORY_ALREADY_REGISTERED`
 * or `CU_FILE_INVALID_MAPPING_SIZE`
 *
 * @note This memory will be use to perform GPU direct DMA from the supported
 * storage.
 * @warning This API is intended for usecases where the memory is used as
 * streaming buffer that is reused across multiple cuFile IO operations.
 */
/*NOLINTNEXTLINE(readability-function-cognitive-complexity)*/
inline void buffer_register(const void* devPtr_base,
                            std::size_t size,
                            int flags                                = 0,
                            const std::vector<int>& errors_to_ignore = std::vector<int>())
{
  if (defaults::compat_mode()) { return; }
#ifdef KVIKIO_CUFILE_FOUND
  CUfileError_t status = cuFileAPI::instance().BufRegister(devPtr_base, size, flags);
  if (status.err != CU_FILE_SUCCESS) {
    // Check if `status.err` is in `errors_to_ignore`
    if (std::find(errors_to_ignore.begin(), errors_to_ignore.end(), status.err) ==
        errors_to_ignore.end()) {
      CUFILE_TRY(status);
    }
  }
#endif
}

/**
 * @brief deregister an already registered device memory from cuFile
 *
 * @param devPtr_base  device pointer to deregister
 */
inline void buffer_deregister(const void* devPtr_base)
{
  if (defaults::compat_mode()) { return; }
#ifdef KVIKIO_CUFILE_FOUND
  CUFILE_TRY(cuFileAPI::instance().BufDeregister(devPtr_base));
#endif
}

/**
 * @brief Register device memory allocation which is part of devPtr. Use this
 * together with FileHandle::pread() and FileHandle::pwrite().
 *
 * @param devPtr Device pointer
 * @param flags Should be zero or `CU_FILE_RDMA_REGISTER` (experimental)
 * @param errors_to_ignore CuFile errors to ignore such as `CU_FILE_MEMORY_ALREADY_REGISTERED`
 * or `CU_FILE_INVALID_MAPPING_SIZE`
 *
 * @note This memory will be use to perform GPU direct DMA from the supported
 * storage.
 * @warning This API is intended for usecases where the memory is used as
 * streaming buffer that is reused across multiple cuFile IO operations.
 */
inline void memory_register(const void* devPtr,
                            int flags                                = 0,
                            const std::vector<int>& errors_to_ignore = {})
{
  auto [base, nbytes, offset] = get_alloc_info(devPtr);
  buffer_register(base, nbytes, flags, errors_to_ignore);
}

/**
 * @brief  deregister an already registered device memory from cuFile.
 *
 * @param devPtr device pointer to deregister
 */
inline void memory_deregister(const void* devPtr)
{
  auto [base, nbytes, offset] = get_alloc_info(devPtr);
  buffer_deregister(base);
}

}  // namespace kvikio
