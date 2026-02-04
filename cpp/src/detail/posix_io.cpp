/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <unistd.h>
#include <cstddef>
#include <cstdlib>

#include <kvikio/bounce_buffer.hpp>
#include <kvikio/detail/nvtx.hpp>
#include <kvikio/detail/posix_io.hpp>
#include <kvikio/error.hpp>
#include <kvikio/shim/cuda.hpp>
#include <kvikio/utils.hpp>

namespace kvikio::detail {
std::size_t posix_device_read(int fd_direct_off,
                              void const* devPtr_base,
                              std::size_t size,
                              std::size_t file_offset,
                              std::size_t devPtr_offset,
                              int fd_direct_on)
{
  KVIKIO_NVTX_FUNC_RANGE(size);
  // If Direct I/O is supported and requested
  if (fd_direct_on != -1 && defaults::auto_direct_io_read()) {
    return detail::posix_device_io<IOOperationType::READ, CudaPageAlignedPinnedBounceBufferPool>(
      fd_direct_off, devPtr_base, size, file_offset, devPtr_offset, fd_direct_on);
  } else {
    return detail::posix_device_io<IOOperationType::READ>(
      fd_direct_off, devPtr_base, size, file_offset, devPtr_offset);
  }
}

std::size_t posix_device_write(int fd_direct_off,
                               void const* devPtr_base,
                               std::size_t size,
                               std::size_t file_offset,
                               std::size_t devPtr_offset,
                               int fd_direct_on)
{
  KVIKIO_NVTX_FUNC_RANGE(size);
  // If Direct I/O is supported and requested
  if (fd_direct_on != -1 && defaults::auto_direct_io_write()) {
    return detail::posix_device_io<IOOperationType::WRITE, CudaPageAlignedPinnedBounceBufferPool>(
      fd_direct_off, devPtr_base, size, file_offset, devPtr_offset, fd_direct_on);
  } else {
    return detail::posix_device_io<IOOperationType::WRITE>(
      fd_direct_off, devPtr_base, size, file_offset, devPtr_offset);
  }
}

}  // namespace kvikio::detail
