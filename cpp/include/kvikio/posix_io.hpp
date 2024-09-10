/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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
#include <kvikio/shim/cuda.hpp>
#include <kvikio/utils.hpp>

namespace kvikio {

namespace detail {

class cuda_context_checker {
 public:
  static bool is_primary_context_active() noexcept
  {
    try {
      CUcontext current_ctx{};
      CUdevice device_handle{};

      CUDA_DRIVER_TRY(cudaAPI::instance().CtxGetCurrent(&current_ctx));
      if (current_ctx == nullptr) {
        int device_idx{0};
        CUDA_DRIVER_TRY(cudaAPI::instance().DeviceGet(&device_handle, device_idx));
      } else {
        CUDA_DRIVER_TRY(cudaAPI::instance().CtxGetDevice(&device_handle));
      }

      // Whether the current context, if it exists, is a primary or user-created context,
      // here we use the primary context to determine if cudaDeviceReset() has been called.
      [[maybe_unused]] unsigned int flags{};
      int active_state{};
      CUDA_DRIVER_TRY(
        cudaAPI::instance().DevicePrimaryCtxGetState(device_handle, &flags, &active_state));

      return static_cast<bool>(active_state);
    } catch (const CUfileException& e) {
      std::cerr << e.what() << std::endl;
    }
    return false;
  }
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
  ~StreamsByThread() noexcept
  {
    if (cuda_context_checker::is_primary_context_active()) {
      for (auto& [_, stream] : _streams) {
        try {
          CUDA_DRIVER_TRY(cudaAPI::instance().StreamDestroy(stream));
        } catch (const CUfileException& e) {
          std::cerr << e.what() << std::endl;
        }
      }
    }
  }

  static CUstream get(CUcontext ctx, std::thread::id thd_id)
  {
    static StreamsByThread _instance;

    // If no current context, we return the null/default stream
    if (ctx == nullptr) { return nullptr; }
    auto key = std::make_pair(ctx, thd_id);

    // Create a new stream if `ctx` doesn't have one.
    if (_instance._streams.find(key) == _instance._streams.end()) {
      CUstream stream{};
      CUDA_DRIVER_TRY(cudaAPI::instance().StreamCreate(&stream, CU_STREAM_DEFAULT));
      _instance._streams[key] = stream;
    }
    return _instance._streams.at(key);
  }

  static CUstream get()
  {
    CUcontext ctx{nullptr};
    CUDA_DRIVER_TRY(cudaAPI::instance().CtxGetCurrent(&ctx));
    return get(ctx, std::this_thread::get_id());
  }

  StreamsByThread(const StreamsByThread&)            = delete;
  StreamsByThread& operator=(StreamsByThread const&) = delete;
  StreamsByThread(StreamsByThread&& o)               = delete;
  StreamsByThread& operator=(StreamsByThread&& o)    = delete;
};

/**
 * @brief Read or write host memory to or from disk using POSIX
 *
 * @tparam IsReadOperation Whether the operation is a read or a write
 * @param fd File descriptor
 * @param buf Buffer to write
 * @param count Number of bytes to write
 * @param offset File offset
 * @param partial If false, all of `count` bytes are read or written.
 * @return The number of bytes read or written (always gather than zero)
 */
template <bool IsReadOperation>
ssize_t posix_host_io(int fd, const void* buf, size_t count, off_t offset, bool partial)
{
  off_t cur_offset      = offset;
  size_t byte_remaining = count;
  char* buffer          = const_cast<char*>(static_cast<const char*>(buf));
  while (byte_remaining > 0) {
    ssize_t nbytes = 0;
    if constexpr (IsReadOperation) {
      nbytes = ::pread(fd, buffer, byte_remaining, cur_offset);
    } else {
      nbytes = ::pwrite(fd, buffer, byte_remaining, cur_offset);
    }
    if (nbytes == -1) {
      const std::string name = IsReadOperation ? "pread" : "pwrite";
      if (errno == EBADF) {
        throw CUfileException{std::string{"POSIX error on " + name + " at: "} + __FILE__ + ":" +
                              KVIKIO_STRINGIFY(__LINE__) + ": Operation not permitted"};
      }
      throw CUfileException{std::string{"POSIX error on " + name + " at: "} + __FILE__ + ":" +
                            KVIKIO_STRINGIFY(__LINE__) + ": " + strerror(errno)};
    }
    if constexpr (IsReadOperation) {
      if (nbytes == 0) {
        throw CUfileException{std::string{"POSIX error on pread at: "} + __FILE__ + ":" +
                              KVIKIO_STRINGIFY(__LINE__) + ": EOF"};
      }
    }
    if (partial) { return nbytes; }
    buffer += nbytes;  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    cur_offset += nbytes;
    byte_remaining -= nbytes;
  }
  return convert_size2ssize(count);
}

/**
 * @brief Read or write device memory to or from disk using POSIX
 *
 * @tparam IsReadOperation Whether the operation is a read or a write
 * @param fd File descriptor
 * @param devPtr_base Device pointer to read or write to.
 * @param size Number of bytes to read or write.
 * @param file_offset Byte offset to the start of the file.
 * @param devPtr_offset Byte offset to the start of the device pointer.
 * @return Number of bytes read or written.
 */
template <bool IsReadOperation>
std::size_t posix_device_io(int fd,
                            const void* devPtr_base,
                            std::size_t size,
                            std::size_t file_offset,
                            std::size_t devPtr_offset)
{
  auto alloc              = AllocRetain::instance().get();
  CUdeviceptr devPtr      = convert_void2deviceptr(devPtr_base) + devPtr_offset;
  off_t cur_file_offset   = convert_size2off(file_offset);
  off_t byte_remaining    = convert_size2off(size);
  const off_t chunk_size2 = convert_size2off(alloc.size());

  // Get a stream for the current CUDA context and thread
  CUstream stream = StreamsByThread::get();

  while (byte_remaining > 0) {
    const off_t nbytes_requested = std::min(chunk_size2, byte_remaining);
    ssize_t nbytes_got           = nbytes_requested;
    if constexpr (IsReadOperation) {
      nbytes_got = posix_host_io<true>(fd, alloc.get(), nbytes_requested, cur_file_offset, true);
      CUDA_DRIVER_TRY(cudaAPI::instance().MemcpyHtoDAsync(devPtr, alloc.get(), nbytes_got, stream));
      CUDA_DRIVER_TRY(cudaAPI::instance().StreamSynchronize(stream));
    } else {  // Is a write operation
      CUDA_DRIVER_TRY(
        cudaAPI::instance().MemcpyDtoHAsync(alloc.get(), devPtr, nbytes_requested, stream));
      CUDA_DRIVER_TRY(cudaAPI::instance().StreamSynchronize(stream));
      posix_host_io<false>(fd, alloc.get(), nbytes_requested, cur_file_offset, false);
    }
    cur_file_offset += nbytes_got;
    devPtr += nbytes_got;
    byte_remaining -= nbytes_got;
  }
  return size;
}

}  // namespace detail

/**
 * @brief Read from disk to host memory using POSIX
 *
 * If `size` or `file_offset` isn't aligned with `page_size` then
 * `fd` cannot have been opened with the `O_DIRECT` flag.
 *
 * @param fd File descriptor
 * @param buf Base address of buffer in host memory.
 * @param size Size in bytes to read.
 * @param file_offset Offset in the file to read from.
 * @param partial If false, all of `size` bytes are read.
 * @return Size of bytes that were successfully read.
 */
inline std::size_t posix_host_read(
  int fd, void* buf, std::size_t size, std::size_t file_offset, bool partial)
{
  KVIKIO_NVTX_FUNC_RANGE("posix_host_read()", size);
  return detail::posix_host_io<true>(fd, buf, size, convert_size2off(file_offset), partial);
}

/**
 * @brief Write host memory to disk using POSIX
 *
 * If `size` or `file_offset` isn't aligned with `page_size` then
 * `fd` cannot have been opened with the `O_DIRECT` flag.
 *
 * @param fd File descriptor
 * @param buf Base address of buffer in host memory.
 * @param size Size in bytes to write.
 * @param file_offset Offset in the file to write to.
 * @param partial If false, all of `size` bytes are written.
 * @return Size of bytes that were successfully read.
 */
inline std::size_t posix_host_write(
  int fd, const void* buf, std::size_t size, std::size_t file_offset, bool partial)
{
  KVIKIO_NVTX_FUNC_RANGE("posix_host_write()", size);
  return detail::posix_host_io<false>(fd, buf, size, convert_size2off(file_offset), partial);
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
inline std::size_t posix_device_read(int fd,
                                     const void* devPtr_base,
                                     std::size_t size,
                                     std::size_t file_offset,
                                     std::size_t devPtr_offset)
{
  KVIKIO_NVTX_FUNC_RANGE("posix_device_read()", size);
  return detail::posix_device_io<true>(fd, devPtr_base, size, file_offset, devPtr_offset);
}

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
inline std::size_t posix_device_write(int fd,
                                      const void* devPtr_base,
                                      std::size_t size,
                                      std::size_t file_offset,
                                      std::size_t devPtr_offset)
{
  KVIKIO_NVTX_FUNC_RANGE("posix_device_write()", size);
  return detail::posix_device_io<false>(fd, devPtr_base, size, file_offset, devPtr_offset);
}

}  // namespace kvikio
