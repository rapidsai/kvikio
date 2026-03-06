/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <unistd.h>
#include <cstddef>
#include <cstdlib>

#include <kvikio/bounce_buffer.hpp>
#include <kvikio/defaults.hpp>
#include <kvikio/detail/nvtx.hpp>
#include <kvikio/detail/posix_io.hpp>
#include <kvikio/error.hpp>
#include <kvikio/shim/cuda.hpp>
#include <kvikio/utils.hpp>

namespace kvikio::detail {
std::size_t posix_device_read_aligned(int fd_direct_off,
                                      void const* devPtr_base,
                                      std::size_t size,
                                      std::size_t file_offset,
                                      std::size_t devPtr_offset,
                                      int fd_direct_on)
{
  auto bounce_buffer                   = CudaPageAlignedPinnedBounceBufferPool::instance().get();
  CUdeviceptr devPtr                   = convert_void2deviceptr(devPtr_base) + devPtr_offset;
  std::size_t const bounce_buffer_size = bounce_buffer.size();
  std::size_t const page_size          = get_page_size();
  std::size_t cur_file_offset          = file_offset;
  std::size_t bytes_remaining          = size;

  // Get a stream for the current CUDA context and thread
  CUstream stream = StreamCachePerThreadAndContext::get();

  while (bytes_remaining > 0) {
    // Each iteration re-derives alignment from the current file offset rather than assuming the
    // previous read left us page-aligned. This is necessary because Direct I/O short reads may
    // return a sector-aligned count (e.g. a multiple of 512 bytes) rather than a page-aligned one,
    // which would leave cur_file_offset unaligned after advancing by nbytes_processed. Re-aligning
    // on every iteration ensures we always issue page-aligned Direct I/O requests, at the cost of
    // potentially re-reading a small prefix that overlaps the previous iteration.
    std::size_t aligned_offset  = align_down(cur_file_offset, page_size);
    std::size_t prefix          = cur_file_offset - aligned_offset;
    std::size_t nbytes_expected = std::min(bytes_remaining, bounce_buffer_size - prefix);
    std::size_t aligned_size    = align_up(prefix + nbytes_expected, page_size);

    // Pure Direct I/O is expected, with aligned offset, aligned buffer, aligned size
    // Note: Use PartialIO::YES for posix_host_io, because the requested read size aligned_size may
    // extend past EOF. With PartialIO::NO, posix_host_io would loop, and eventually hit EOF on
    // ::pread.
    ssize_t nbytes_io =
      posix_host_io<IOOperationType::READ, PartialIO::YES>(fd_direct_off,
                                                           bounce_buffer.get(),
                                                           aligned_size,
                                                           convert_size2off(aligned_offset),
                                                           fd_direct_on);
    KVIKIO_EXPECT(nbytes_io > static_cast<ssize_t>(prefix),
                  "pread(O_DIRECT): unexpected EOF within the requested range");

    std::size_t nbytes_processed =
      std::min(nbytes_expected, static_cast<std::size_t>(nbytes_io) - prefix);

    CUDA_DRIVER_TRY(cudaAPI::instance().MemcpyHtoDAsync(
      devPtr, static_cast<std::byte*>(bounce_buffer.get()) + prefix, nbytes_processed, stream));
    CUDA_DRIVER_TRY(cudaAPI::instance().StreamSynchronize(stream));

    cur_file_offset += nbytes_processed;
    devPtr += nbytes_processed;
    bytes_remaining -= nbytes_processed;
  }
  return size;
}

std::size_t posix_device_read(int fd_direct_off,
                              void const* devPtr_base,
                              std::size_t size,
                              std::size_t file_offset,
                              std::size_t devPtr_offset,
                              int fd_direct_on)
{
  KVIKIO_NVTX_FUNC_RANGE(size);
  // The bounce buffer must hold at least 2 pages so that a read straddling two adjacent pages can
  // be satisfied in a single aligned pread.
  static std::size_t const lower_bound = 2 * get_page_size();
  // If Direct I/O is supported and requested and bounce buffer is at least two pages
  if (fd_direct_on != -1 && defaults::auto_direct_io_read() &&
      defaults::bounce_buffer_size() >= lower_bound) {
    if (defaults::auto_direct_io_read_overread()) {
      return posix_device_read_aligned(
        fd_direct_off, devPtr_base, size, file_offset, devPtr_offset, fd_direct_on);
    }
    return posix_device_io<IOOperationType::READ, CudaPageAlignedPinnedBounceBufferPool>(
      fd_direct_off, devPtr_base, size, file_offset, devPtr_offset, fd_direct_on);
  } else {
    return posix_device_io<IOOperationType::READ>(
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
  // Note: Unlike reads, writes cannot use an "over-write" approach because writing beyond the
  // requested range would corrupt adjacent file data. Instead, we use the opportunistic Direct I/O
  // path which falls back to buffered I/O for unaligned prefix/suffix portions.
  if (fd_direct_on != -1 && defaults::auto_direct_io_write()) {
    return posix_device_io<IOOperationType::WRITE, CudaPageAlignedPinnedBounceBufferPool>(
      fd_direct_off, devPtr_base, size, file_offset, devPtr_offset, fd_direct_on);
  } else {
    return posix_device_io<IOOperationType::WRITE>(
      fd_direct_off, devPtr_base, size, file_offset, devPtr_offset);
  }
}

}  // namespace kvikio::detail
