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

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <optional>
#include <system_error>
#include <utility>

#include <kvikio/buffer.hpp>
#include <kvikio/defaults.hpp>
#include <kvikio/error.hpp>
#include <kvikio/parallel_operation.hpp>
#include <kvikio/posix_io.hpp>
#include <kvikio/shim/cufile.hpp>
#include <kvikio/utils.hpp>
#include <kvikio/shim/cuda.hpp>

namespace kvikio {

/**
 * @brief Handle of an open file registered with cufile.
 *
 * In order to utilize cufile and GDS, a file must be registered with cufile.
 */
class StreamHandle {
 private:
  int _fd{-1};
  bool _initialized{false};
  bool _compat_mode{false};
  mutable std::size_t _nbytes{0};  // The size of the underlying file, zero means unknown.
  CUfileHandle_t _handle{};
  CUstream _stream;
  unsigned _stream_flags;

 public:
  static constexpr mode_t m644 = S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH;
  StreamHandle() noexcept        = default;

  /**
   * @brief Construct a stream handle from a file path
   *
   * @param file_path File path to the file
   * @param flags Open flags (see also `fopen(3)`):
   *   "r" -> "open for reading (default)"
   *   "w" -> "open for writing, truncating the file first"
   *   "a" -> "open for writing, appending to the end of file if it exists"
   *   "+" -> "open for updating (reading and writing)"
   * @param mode Access modes (see `open(2)`).
   * @param compat_mode Enable KvikIO's compatibility mode for this file.
   * @param stream_flags for cufile stream register call.
   */
  StreamHandle(const std::string& file_path,
             const std::string& flags = "r",
             mode_t mode              = m644,
             bool compat_mode         = defaults::compat_mode(),
	     unsigned stream_flags    = 0xf)
    : _initialized{true},
      _compat_mode{compat_mode},
      _stream_flags{stream_flags}
  {
    try {
      _fd = detail::open_fd(file_path, flags, true, mode);
    } catch (const std::system_error&) {
      _compat_mode = true;  // Fall back to compat mode if we cannot open the file with O_DIRECT
    }

    if (_compat_mode) { return; }

    CUDA_DRIVER_TRY(cudaAPI::instance().StreamCreate(&_stream, CU_STREAM_DEFAULT));
#ifdef KVIKIO_CUFILE_EXIST
    CUfileDescr_t desc{};  // It is important to set to zero!
    desc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-union-access)
    desc.handle.fd = _fd;
    CUFILE_TRY(cuFileAPI::instance().HandleRegister(&_handle, &desc));
    if (_stream != nullptr) {
	    CUFILE_TRY(cuFileAPI::instance().StreamRegister(_stream, _stream_flags));
    }
#endif
  }

  /**
   * @brief FileHandle support move semantic but isn't copyable
   */
  StreamHandle(const StreamHandle&)            = delete;
  StreamHandle& operator=(StreamHandle const&) = delete;
  StreamHandle(StreamHandle&& o) noexcept
    : _fd{std::exchange(o._fd, -1)},
      _initialized{std::exchange(o._initialized, false)},
      _compat_mode{std::exchange(o._compat_mode, false)},
      _nbytes{std::exchange(o._nbytes, 0)},
      _handle{std::exchange(o._handle, CUfileHandle_t{})},
      _stream{std::exchange(o._stream, CUstream{})},
      _stream_flags{std::exchange(o._stream_flags, 0xf)}
  {
  }
  StreamHandle& operator=(StreamHandle&& o) noexcept
  {
    _fd = std::exchange(o._fd, -1);
    _initialized   = std::exchange(o._initialized, false);
    _compat_mode   = std::exchange(o._compat_mode, false);
    _nbytes        = std::exchange(o._nbytes, 0);
    _handle        = std::exchange(o._handle, CUfileHandle_t{});
    _stream_flags  = std::exchange(o._stream_flags, 0xf);
    _stream	   = std::exchange(o._stream, CUstream{});
    return *this;
  }
  ~StreamHandle() noexcept { close(); }

  [[nodiscard]] bool closed() const noexcept { return !_initialized; }

  /**
   * @brief Deregister the file and close the stream. 
   */
  void close()
  {
    if (closed()) { return; }

    if (!_compat_mode) {
#ifdef KVIKIO_CUFILE_EXIST
      cuFileAPI::instance().HandleDeregister(_handle);
#endif
    }
    if (_stream != nullptr) {
#ifdef KVIKIO_CUFILE_EXIST
	CUFILE_TRY(cuFileAPI::instance().StreamDeregister(_stream));
#endif
    	CUDA_DRIVER_TRY(cudaAPI::instance().StreamDestroy(_stream));
    	_stream = nullptr;
    }

    if (_fd != -1) { ::close(_fd); }
    _fd = -1;
    _stream_flags = 0;
    _initialized   = false;
  }

  /**
   * @brief Get the underlying cuFile file handle
   *
   * The file handle must be open and not in compatibility mode i.e.
   * both `.closed()` and `.is_compat_mode_on()` must be return false.
   *
   * @return cuFile's file handle
   */
  [[nodiscard]] CUfileHandle_t handle()
  {
    if (closed()) { throw CUfileException("File handle is closed"); }
    if (_compat_mode) {
      throw CUfileException("The underlying cuFile handle isn't available in compatibility mode");
    }
    return _handle;
  }

  /**
   * @brief Get the file descriptors
   *
   * @return File descriptor
   */
  [[nodiscard]] int fd() const noexcept { return _fd; }

  /**
   * @brief Get the flags of one of the file descriptors (see open(2))
   *
   * @return File descriptor
   */
  [[nodiscard]] int fd_open_flags() const { return detail::open_flags(_fd); }

  /**
   * @brief Get the number of bytes
   *
   * The value are cached.
   *
   * @return The number of bytes
   */
  [[nodiscard]] inline std::size_t nbytes() const
  {
    if (closed()) { return 0; }
    if (_nbytes == 0) { _nbytes = detail::get_file_size(_fd); }
    return _nbytes;
  }
  /**
   * @brief Get the stream associated with the handle. 
   *
   * @return CUstream 
   */
  [[nodiscard]] CUstream get_stream() const noexcept { return _stream; }

  void stream_synchronize()
  {
	CUDA_DRIVER_TRY(cudaAPI::instance().StreamSynchronize(_stream));
  }

  /**
   * @brief Reads specified bytes from the file into the device memory.
   *
   * This API reads asynchronously the data from the GPU memory to the file at a specified offset
   * and size bytes by using GDS functionality. The API works correctly for unaligned
   * offset and data sizes, although the performance is not on-par with aligned read.
   * This is an asynchronous call and will be executed in sequence for the specified stream.
   *
   * @note For the `devPtr_offset`, if data will be read starting exactly from the
   * `devPtr_base` that is registered with `buffer_register`, `devPtr_offset` should
   * be set to 0. To read starting from an offset in the registered buffer range,
   * the relative offset should be specified in the `devPtr_offset`, and the
   * `devPtr_base` must remain set to the base address that was used in the
   * `buffer_register` call.
   *
   * @param devPtr_base Base address of buffer in device memory. For registered buffers,
   * `devPtr_base` must remain set to the base address used in the `buffer_register` call.
   * @param size Size in bytes to read.
   * @param file_offset Offset in the file to read from.
   * @param devPtr_offset Offset relative to the `devPtr_base` pointer to read into.
   * This parameter should be used only with registered buffers.
   * @return Size of bytes that were successfully read.
   */
  std::size_t read_async(void* devPtr_base,
                   std::size_t *size,
                   off_t *file_offset,
                   off_t *devPtr_offset)
  {
#ifdef KVIKIO_CUFILE_EXIST
    ssize_t ret;
    CUFILE_TRY(cuFileAPI::instance().ReadAsync( _handle, devPtr_base, size,
		file_offset, devPtr_offset, &ret, _stream));
    if (ret == -1) {
      throw std::system_error(errno, std::generic_category(), "Unable to submit stream read");
    }
    if (ret < -1) {
      throw CUfileException(std::string{"cuFile error at: "} + __FILE__ + ":" +
                            KVIKIO_STRINGIFY(__LINE__) + ": " + CUFILE_ERRSTR(ret));
    }
    return ret;
#else
    throw CUfileException("KvikIO not compiled with cuFile.h");
#endif
  }

  /**
   * @brief Writes specified bytes from the device memory into the file.
   *
   * This API writes asynchronously the data from the GPU memory to the file at a specified offset
   * and size bytes by using GDS functionality. The API works correctly for unaligned
   * offset and data sizes, although the performance is not on-par with aligned writes.
   * This is an asynchronous call and will be executed in sequence for the specified stream.
   *
   * @note  GDS functionality modified the standard file system metadata in SysMem.
   * However, GDS functionality does not take any special responsibility for writing
   * that metadata back to permanent storage. The data is not guaranteed to be present
   * after a system crash unless the application uses an explicit `fsync(2)` call. If the
   * file is opened with an `O_SYNC` flag, the metadata will be written to the disk before
   * the call is complete.
   * Refer to the note in read for more information about `devPtr_offset`.
   *
   * @param devPtr_base Base address of buffer in device memory. For registered buffers,
   * `devPtr_base` must remain set to the base address used in the `buffer_register` call.
   * @param size Size in bytes to write.
   * @param file_offset Offset in the file to write from.
   * @param devPtr_offset Offset relative to the `devPtr_base` pointer to write into.
   * This parameter should be used only with registered buffers.
   * @return Size of bytes that were successfully written.
   */
  std::size_t write_async(void* devPtr_base,
                    std::size_t *size,
                    off_t *file_offset,
                    off_t *devPtr_offset)
  {
    _nbytes = 0;  // Invalidate the computed file size


#ifdef KVIKIO_CUFILE_EXIST
    ssize_t ret; 
    CUFILE_TRY(cuFileAPI::instance().WriteAsync(_handle, devPtr_base, size,
		file_offset, devPtr_offset, &ret, _stream));
    if (ret == -1) {
      throw std::system_error(errno, std::generic_category(), "Unable to submit stream write");
    }
    if (ret < -1) {
      throw CUfileException(std::string{"cuFile error at: "} + __FILE__ + ":" +
                            KVIKIO_STRINGIFY(__LINE__) + ": " + CUFILE_ERRSTR(ret));
    }
    return ret;
#else
    throw CUfileException("KvikIO not compiled with cuFile.h");
#endif
  }

};
}  // namespace kvikio
