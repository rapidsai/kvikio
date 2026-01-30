/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <vector>

namespace kvikio {

/**
 * @brief Register a device memory region with cuFile for GPUDirect Storage access.
 *
 * This is the low-level registration function that requires the caller to specify the exact base
 * address and size of the memory region to register. For a convenience wrapper that automatically
 * discovers the allocation boundaries, see memory_register().
 *
 * Registration pins the memory for GPU Direct DMA transfers, which can improve performance when the
 * same buffer is reused across multiple cuFile I/O operations.
 *
 * In compatibility mode (when GDS is unavailable), this function is a no-op.
 *
 * @warning This API is intended for streaming buffers reused across multiple cuFile I/O operations.
 * For one-time transfers, the overhead of registration may outweigh the benefits.
 *
 * @param devPtr_base Base address of the device memory region to register.
 * @param size Size in bytes of the memory region to register.
 * @param flags Registration flags. Should be 0 or `CU_FILE_RDMA_REGISTER` (experimental).
 * @param errors_to_ignore cuFile error codes to silently ignore, such as
 * `CU_FILE_MEMORY_ALREADY_REGISTERED` or `CU_FILE_INVALID_MAPPING_SIZE`.
 *
 * @exception CUfileException If cuFile registration fails with an error not in @p errors_to_ignore.
 *
 * @see memory_register for automatic discovery of allocation base address and size.
 * @see buffer_deregister to deregister the memory.
 */
void buffer_register(void const* devPtr_base,
                     std::size_t size,
                     int flags                                = 0,
                     std::vector<int> const& errors_to_ignore = std::vector<int>());

/**
 * @brief Deregister a device memory region from cuFile.
 *
 * This is the low-level deregistration function that requires the caller to specify the exact base
 * address that was previously registered. For a convenience wrapper that automatically discovers
 * the allocation boundaries, see memory_deregister().
 *
 * In compatibility mode (when GDS is unavailable), this function is a no-op.
 *
 * @param devPtr_base Base address of the device memory region to deregister. Must match the address
 * used in the corresponding buffer_register() call.
 *
 * @exception CUfileException If cuFile deregistration fails.
 *
 * @see memory_deregister for automatic discovery of allocation base address.
 * @see buffer_register to register the memory.
 */
void buffer_deregister(void const* devPtr_base);

/**
 * @brief Register a device memory allocation with cuFile for GPUDirect Storage access. Use this
 * function together with FileHandle::pread() and FileHandle::pwrite().
 *
 * This is a convenience wrapper around buffer_register() that automatically discovers the base
 * address and size of the CUDA memory allocation containing @p devPtr. The entire underlying
 * allocation is registered, regardless of which portion @p devPtr points to.
 *
 * Registration pins the memory for GPU Direct DMA transfers, which can improve performance when the
 * same buffer is reused across multiple cuFile I/O operations.
 *
 * In compatibility mode (when GDS is unavailable), this function is a no-op.
 *
 * @warning This API is intended for streaming buffers reused across multiple cuFile I/O operations.
 * For one-time transfers, the overhead of registration may outweigh the benefits.
 *
 * @param devPtr Pointer anywhere within a CUDA device memory allocation.
 * @param flags Registration flags. Should be 0 or `CU_FILE_RDMA_REGISTER` (experimental).
 * @param errors_to_ignore cuFile error codes to silently ignore, such as
 * `CU_FILE_MEMORY_ALREADY_REGISTERED` or `CU_FILE_INVALID_MAPPING_SIZE`.
 *
 * @exception CUfileException If cuFile registration fails with an error not in @p errors_to_ignore.
 *
 * @see buffer_register for registering with explicit base address and size.
 * @see memory_deregister to deregister the memory.
 */
void memory_register(void const* devPtr,
                     int flags                                = 0,
                     std::vector<int> const& errors_to_ignore = {});

/**
 * @brief Deregister a device memory allocation from cuFile.
 *
 * This is a convenience wrapper around buffer_deregister() that automatically discovers the base
 * address of the CUDA memory allocation containing @p devPtr. The entire underlying allocation is
 * deregistered, regardless of which portion @p devPtr points to.
 *
 * In compatibility mode (when GDS is unavailable), this function is a no-op.
 *
 * @param devPtr Pointer anywhere within a previously registered CUDA device
 *        memory allocation.
 *
 * @exception CUfileException If cuFile deregistration fails.
 *
 * @see buffer_deregister for deregistering with explicit base address.
 * @see memory_register to register the memory.
 */
void memory_deregister(void const* devPtr);

}  // namespace kvikio
