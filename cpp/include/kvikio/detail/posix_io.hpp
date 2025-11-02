/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <unistd.h>
#include <cstddef>
#include <cstdlib>
#include <map>
#include <optional>
#include <thread>

#include <kvikio/bounce_buffer.hpp>
#include <kvikio/defaults.hpp>
#include <kvikio/detail/nvtx.hpp>
#include <kvikio/detail/utils.hpp>
#include <kvikio/error.hpp>
#include <kvikio/shim/cuda.hpp>
#include <kvikio/utils.hpp>

namespace kvikio::detail {

/**
 * @brief Type of the IO operation.
 */
enum class IOOperationType : uint8_t {
  READ,   ///< POSIX read.
  WRITE,  ///< POSIX write.
};

/**
 * @brief Specifies whether all requested bytes are to be processed or not.
 */
enum class PartialIO : uint8_t {
  YES,  ///< POSIX read/write is called only once, which may not process all bytes requested.
  NO,   ///< POSIX read/write is called repeatedly until all requested bytes are processed.
};

/**
 * @brief Singleton class to retrieve a CUDA stream for device-host copying
 *
 * Call `StreamsByThread::get` to get the CUDA stream assigned to the current
 * CUDA context and thread.
 */
class StreamsByThread {
 private:
  std::map<std::pair<CUcontext, std::thread::id>, CUstream> _streams;

 public:
  StreamsByThread() = default;

  // Here we intentionally do not destroy in the destructor the CUDA resources
  // (e.g. CUstream) with static storage duration, but instead let them leak
  // on program termination. This is to prevent undefined behavior in CUDA. See
  // <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#initialization>
  // This also prevents crash (segmentation fault) if clients call
  // cuDevicePrimaryCtxReset() or cudaDeviceReset() before program termination.
  ~StreamsByThread() = default;

  KVIKIO_EXPORT static CUstream get(CUcontext ctx, std::thread::id thd_id);

  static CUstream get();

  StreamsByThread(StreamsByThread const&)            = delete;
  StreamsByThread& operator=(StreamsByThread const&) = delete;
  StreamsByThread(StreamsByThread&& o)               = delete;
  StreamsByThread& operator=(StreamsByThread&& o)    = delete;
};

/**
 * @brief Read or write host memory to or from disk using POSIX
 *
 * @tparam Operation Whether the operation is a read or a write.
 * @tparam PartialIOStatus Whether all requested data are processed or not. If `FULL`, all of
 * `count` bytes are read or written.
 * @param fd_direct_off File descriptor without Direct I/O
 * @param buf Buffer to write
 * @param count Number of bytes to write
 * @param offset File offset
 * @param fd_direct_on Optional file descriptor with Direct I/O
 * @return The number of bytes read or written (always gather than zero)
 */
template <IOOperationType Operation,
          PartialIO PartialIOStatus,
          typename BounceBufferPoolType = PageAlignedBounceBufferPool>
ssize_t posix_host_io(
  int fd_direct_off, void const* buf, size_t count, off_t offset, int fd_direct_on = -1)
{
  auto pread_or_write = [](int fd, void* buf, size_t count, off_t offset) -> ssize_t {
    ssize_t nbytes{};
    if constexpr (Operation == IOOperationType::READ) {
      nbytes = ::pread(fd, buf, count, offset);
    } else {
      nbytes = ::pwrite(fd, buf, count, offset);
    }
    return nbytes;
  };

  off_t cur_offset       = offset;
  size_t bytes_remaining = count;
  char* buffer           = const_cast<char*>(static_cast<char const*>(buf));
  auto const page_size   = get_page_size();

  while (bytes_remaining > 0) {
    ssize_t nbytes_processed{};

    if (fd_direct_on == -1) {
      nbytes_processed = pread_or_write(fd_direct_off, buffer, bytes_remaining, cur_offset);
    } else {
      // Make earnest attempt to perform Direct I/O
      auto const is_cur_offset_aligned = detail::is_aligned(cur_offset, page_size);

      if (!is_cur_offset_aligned) {
        auto const aligned_cur_offset = detail::align_up(cur_offset, page_size);
        auto const bytes_requested    = std::min(aligned_cur_offset - cur_offset, bytes_remaining);
        // Use non-Direct I/O from the current file offset to a page boundary
        nbytes_processed = pread_or_write(fd_direct_off, buffer, bytes_requested, cur_offset);
      } else {
        if (bytes_remaining < page_size) {
          nbytes_processed = pread_or_write(fd_direct_off, buffer, bytes_remaining, cur_offset);
        } else {
          auto aligned_bytes_remaining = detail::align_down(bytes_remaining, page_size);
          auto const is_buf_aligned    = detail::is_aligned(buffer, page_size);
          auto bytes_requested         = aligned_bytes_remaining;
          if (!is_buf_aligned) {
            auto bounce_buffer = BounceBufferPoolType::instance().get();
            auto* aligned_buf  = bounce_buffer.get();
            bytes_requested    = std::min(bytes_requested, bounce_buffer.size());

            if constexpr (Operation == IOOperationType::WRITE) {
              std::memcpy(aligned_buf, buffer, bytes_requested);
            }

            // Perform Direct I/O
            nbytes_processed =
              pread_or_write(fd_direct_on, aligned_buf, bytes_requested, cur_offset);

            if constexpr (Operation == IOOperationType::READ) {
              std::memcpy(buffer, aligned_buf, nbytes_processed);
            }
          } else {
            // Perform Direct I/O
            nbytes_processed = pread_or_write(fd_direct_on, buffer, bytes_requested, cur_offset);
          }
        }
      }
    }

    if (nbytes_processed == -1) {
      std::string const name = (Operation == IOOperationType::READ) ? "pread" : "pwrite";
      KVIKIO_EXPECT(errno != EBADF, "POSIX error: Operation not permitted");
      KVIKIO_FAIL("POSIX error on " + name + ": " + strerror(errno));
    }
    if constexpr (Operation == IOOperationType::READ) {
      KVIKIO_EXPECT(nbytes_processed != 0, "POSIX error on pread: EOF");
    }
    if constexpr (PartialIOStatus == PartialIO::YES) { return nbytes_processed; }
    buffer += nbytes_processed;  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    cur_offset += nbytes_processed;
    bytes_remaining -= nbytes_processed;
  }
  return convert_size2ssize(count);
}

/**
 * @brief Read or write device memory to or from disk using POSIX
 *
 * @tparam Operation Whether the operation is a read or a write.
 * @param fd File descriptor
 * @param devPtr_base Device pointer to read or write to.
 * @param size Number of bytes to read or write.
 * @param file_offset Byte offset to the start of the file.
 * @param devPtr_offset Byte offset to the start of the device pointer.
 * @return Number of bytes read or written.
 */
template <IOOperationType Operation, typename BounceBufferPoolType = CudaPinnedBounceBufferPool>
std::size_t posix_device_io(int fd_direct_off,
                            void const* devPtr_base,
                            std::size_t size,
                            std::size_t file_offset,
                            std::size_t devPtr_offset,
                            int fd_direct_on = -1)
{
  auto bounce_buffer      = BounceBufferPoolType::instance().get();
  CUdeviceptr devPtr      = convert_void2deviceptr(devPtr_base) + devPtr_offset;
  off_t cur_file_offset   = convert_size2off(file_offset);
  off_t bytes_remaining   = convert_size2off(size);
  off_t const chunk_size2 = convert_size2off(bounce_buffer.size());

  // Get a stream for the current CUDA context and thread
  CUstream stream = StreamsByThread::get();

  while (bytes_remaining > 0) {
    off_t const nbytes_requested = std::min(chunk_size2, bytes_remaining);
    ssize_t nbytes_got           = nbytes_requested;
    if constexpr (Operation == IOOperationType::READ) {
      nbytes_got = posix_host_io<IOOperationType::READ, PartialIO::YES>(
        fd_direct_off, bounce_buffer.get(), nbytes_requested, cur_file_offset, fd_direct_on);
      CUDA_DRIVER_TRY(
        cudaAPI::instance().MemcpyHtoDAsync(devPtr, bounce_buffer.get(), nbytes_got, stream));
      CUDA_DRIVER_TRY(cudaAPI::instance().StreamSynchronize(stream));
    } else {  // Is a write operation
      CUDA_DRIVER_TRY(
        cudaAPI::instance().MemcpyDtoHAsync(bounce_buffer.get(), devPtr, nbytes_requested, stream));
      CUDA_DRIVER_TRY(cudaAPI::instance().StreamSynchronize(stream));
      posix_host_io<IOOperationType::WRITE, PartialIO::NO>(
        fd_direct_off, bounce_buffer.get(), nbytes_requested, cur_file_offset, fd_direct_on);
    }
    cur_file_offset += nbytes_got;
    devPtr += nbytes_got;
    bytes_remaining -= nbytes_got;
  }
  return size;
}

/**
 * @brief Read from disk to host memory using POSIX
 *
 * If `size` or `file_offset` isn't aligned with `page_size` then
 * `fd` cannot have been opened with the `O_DIRECT` flag.
 *
 * @tparam PartialIOStatus Whether all requested data are processed or not. If `FULL`, all of
 * `count` bytes are read.
 * @param fd File descriptor
 * @param buf Base address of buffer in host memory.
 * @param size Size in bytes to read.
 * @param file_offset Offset in the file to read from.
 * @return Size of bytes that were successfully read.
 */
template <PartialIO PartialIOStatus>
std::size_t posix_host_read(
  int fd_direct_off, void* buf, std::size_t size, std::size_t file_offset, int fd_direct_on = -1)
{
  KVIKIO_NVTX_FUNC_RANGE(size);
  return detail::posix_host_io<IOOperationType::READ, PartialIOStatus>(
    fd_direct_off, buf, size, convert_size2off(file_offset), fd_direct_on);
}

/**
 * @brief Write host memory to disk using POSIX
 *
 * If `size` or `file_offset` isn't aligned with `page_size` then
 * `fd` cannot have been opened with the `O_DIRECT` flag.
 *
 * @tparam ioDataCompletionLevel Whether all requested data are processed or not. If `FULL`, all
 * of `count` bytes are written.
 * @param fd File descriptor
 * @param buf Base address of buffer in host memory.
 * @param size Size in bytes to write.
 * @param file_offset Offset in the file to write to.
 * @return Size of bytes that were successfully read.
 */
template <PartialIO PartialIOStatus>
std::size_t posix_host_write(int fd_direct_off,
                             void const* buf,
                             std::size_t size,
                             std::size_t file_offset,
                             int fd_direct_on = -1)
{
  KVIKIO_NVTX_FUNC_RANGE(size);
  return detail::posix_host_io<IOOperationType::WRITE, PartialIOStatus>(
    fd_direct_off, buf, size, convert_size2off(file_offset), fd_direct_on);
}

/**
 * @brief Read from disk to device memory using POSIX
 *
 * If `size` or `file_offset` isn't aligned with `page_size` then
 * `fd` cannot have been opened with the `O_DIRECT` flag.
 *
 * @param fd File descriptor
 * @param devPtr_base Base address of buffer in device memory.
 * @param size Size in bytes to read.
 * @param file_offset Offset in the file to read from.
 * @param devPtr_offset Offset relative to the `devPtr_base` pointer to read into.
 * @return Size of bytes that were successfully read.
 */
std::size_t posix_device_read(int fd_direct_off,
                              void const* devPtr_base,
                              std::size_t size,
                              std::size_t file_offset,
                              std::size_t devPtr_offset,
                              int fd_direct_on = -1);

/**
 * @brief Write device memory to disk using POSIX
 *
 * If `size` or `file_offset` isn't aligned with `page_size` then
 * `fd` cannot have been opened with the `O_DIRECT` flag.
 *
 * @param fd File descriptor
 * @param devPtr_base Base address of buffer in device memory.
 * @param size Size in bytes to write.
 * @param file_offset Offset in the file to write to.
 * @param devPtr_offset Offset relative to the `devPtr_base` pointer to write into.
 * @return Size of bytes that were successfully written.
 */
std::size_t posix_device_write(int fd_direct_off,
                               void const* devPtr_base,
                               std::size_t size,
                               std::size_t file_offset,
                               std::size_t devPtr_offset,
                               int fd_direct_on = -1);

}  // namespace kvikio::detail
