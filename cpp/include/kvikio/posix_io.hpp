/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

#include <unistd.h>
#include <cstddef>
#include <cstdlib>
#include <map>
#include <thread>

#include <kvikio/bounce_buffer.hpp>
#include <kvikio/error.hpp>
#include <kvikio/nvtx.hpp>
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
 * @param fd File descriptor
 * @param buf Buffer to write
 * @param count Number of bytes to write
 * @param offset File offset
 * @return The number of bytes read or written (always gather than zero)
 */
template <IOOperationType Operation, PartialIO PartialIOStatus>
ssize_t posix_host_io(int fd, void const* buf, size_t count, off_t offset)
{
  off_t cur_offset      = offset;
  size_t byte_remaining = count;
  char* buffer          = const_cast<char*>(static_cast<char const*>(buf));
  while (byte_remaining > 0) {
    ssize_t nbytes = 0;
    if constexpr (Operation == IOOperationType::READ) {
      nbytes = ::pread(fd, buffer, byte_remaining, cur_offset);
    } else {
      nbytes = ::pwrite(fd, buffer, byte_remaining, cur_offset);
    }
    if (nbytes == -1) {
      std::string const name = (Operation == IOOperationType::READ) ? "pread" : "pwrite";
      KVIKIO_EXPECT(errno != EBADF, "POSIX error: Operation not permitted");
      KVIKIO_FAIL("POSIX error on " + name + ": " + strerror(errno));
    }
    if constexpr (Operation == IOOperationType::READ) {
      KVIKIO_EXPECT(nbytes != 0, "POSIX error on pread: EOF");
    }
    if constexpr (PartialIOStatus == PartialIO::YES) { return nbytes; }
    buffer += nbytes;  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    cur_offset += nbytes;
    byte_remaining -= nbytes;
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
template <IOOperationType Operation>
std::size_t posix_device_io(int fd,
                            void const* devPtr_base,
                            std::size_t size,
                            std::size_t file_offset,
                            std::size_t devPtr_offset)
{
  auto alloc              = AllocRetain::instance().get();
  CUdeviceptr devPtr      = convert_void2deviceptr(devPtr_base) + devPtr_offset;
  off_t cur_file_offset   = convert_size2off(file_offset);
  off_t byte_remaining    = convert_size2off(size);
  off_t const chunk_size2 = convert_size2off(alloc.size());

  // Get a stream for the current CUDA context and thread
  CUstream stream = StreamsByThread::get();

  while (byte_remaining > 0) {
    off_t const nbytes_requested = std::min(chunk_size2, byte_remaining);
    ssize_t nbytes_got           = nbytes_requested;
    if constexpr (Operation == IOOperationType::READ) {
      nbytes_got = posix_host_io<IOOperationType::READ, PartialIO::YES>(
        fd, alloc.get(), nbytes_requested, cur_file_offset);
      CUDA_DRIVER_TRY(cudaAPI::instance().MemcpyHtoDAsync(devPtr, alloc.get(), nbytes_got, stream));
      CUDA_DRIVER_TRY(cudaAPI::instance().StreamSynchronize(stream));
    } else {  // Is a write operation
      CUDA_DRIVER_TRY(
        cudaAPI::instance().MemcpyDtoHAsync(alloc.get(), devPtr, nbytes_requested, stream));
      CUDA_DRIVER_TRY(cudaAPI::instance().StreamSynchronize(stream));
      posix_host_io<IOOperationType::WRITE, PartialIO::NO>(
        fd, alloc.get(), nbytes_requested, cur_file_offset);
    }
    cur_file_offset += nbytes_got;
    devPtr += nbytes_got;
    byte_remaining -= nbytes_got;
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
std::size_t posix_host_read(int fd, void* buf, std::size_t size, std::size_t file_offset)
{
  KVIKIO_NVTX_FUNC_RANGE(size);
  return detail::posix_host_io<IOOperationType::READ, PartialIOStatus>(
    fd, buf, size, convert_size2off(file_offset));
}

/**
 * @brief Write host memory to disk using POSIX
 *
 * If `size` or `file_offset` isn't aligned with `page_size` then
 * `fd` cannot have been opened with the `O_DIRECT` flag.
 *
 * @tparam ioDataCompletionLevel Whether all requested data are processed or not. If `FULL`, all of
 * `count` bytes are written.
 * @param fd File descriptor
 * @param buf Base address of buffer in host memory.
 * @param size Size in bytes to write.
 * @param file_offset Offset in the file to write to.
 * @return Size of bytes that were successfully read.
 */
template <PartialIO PartialIOStatus>
std::size_t posix_host_write(int fd, void const* buf, std::size_t size, std::size_t file_offset)
{
  KVIKIO_NVTX_FUNC_RANGE(size);
  return detail::posix_host_io<IOOperationType::WRITE, PartialIOStatus>(
    fd, buf, size, convert_size2off(file_offset));
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
std::size_t posix_device_read(int fd,
                              void const* devPtr_base,
                              std::size_t size,
                              std::size_t file_offset,
                              std::size_t devPtr_offset);

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
std::size_t posix_device_write(int fd,
                               void const* devPtr_base,
                               std::size_t size,
                               std::size_t file_offset,
                               std::size_t devPtr_offset);

}  // namespace kvikio::detail
