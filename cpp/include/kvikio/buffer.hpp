/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

#include <vector>

namespace kvikio {

/**
 * @brief register an existing cudaMalloced memory with cuFile to pin for
 * GPUDirect Storage access.
 *
 * @param devPtr_base  device pointer to allocated
 * @param size  size of memory region from the above specified devPtr
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
void buffer_register(void const* devPtr_base,
                     std::size_t size,
                     int flags                                = 0,
                     std::vector<int> const& errors_to_ignore = std::vector<int>());

/**
 * @brief deregister an already registered device memory from cuFile
 *
 * @param devPtr_base  device pointer to deregister
 */
void buffer_deregister(void const* devPtr_base);

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
void memory_register(void const* devPtr,
                     int flags                                = 0,
                     std::vector<int> const& errors_to_ignore = {});

/**
 * @brief  deregister an already registered device memory from cuFile.
 *
 * @param devPtr device pointer to deregister
 */
void memory_deregister(void const* devPtr);

}  // namespace kvikio
