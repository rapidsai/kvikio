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
#include <kvikio/config.hpp>
#include <kvikio/error.hpp>
#include <kvikio/parallel_operation.hpp>
#include <kvikio/posix_io.hpp>
#include <kvikio/shim/cufile.hpp>
#include <kvikio/thread_pool/default.hpp>
#include <kvikio/utils.hpp>

namespace kvikio {
namespace {

/**
 * @brief Parse open file flags given as a string and return oflags
 *
 * @param flags The flags
 * @return oflags
 */
inline int open_fd_parse_flags(const std::string& flags)
{
  int file_flags = -1;
  if (flags.empty()) { throw std::invalid_argument("Unknown file open flag"); }
  switch (flags[0]) {
    case 'r':
      file_flags = O_RDONLY;
      if (flags[1] == '+') { file_flags = O_RDWR; }
      break;
    case 'w':
      file_flags = O_WRONLY;
      if (flags[1] == '+') { file_flags = O_RDWR; }
      file_flags |= O_CREAT | O_TRUNC;
      break;
    case 'a': throw std::invalid_argument("Open flag 'a' isn't supported");
    default: throw std::invalid_argument("Unknown file open flag");
  }
  file_flags |= O_CLOEXEC;
  if (!config::get_global_compat_mode()) { file_flags |= O_DIRECT; }
  return file_flags;
}

/**
 * @brief Open file using `open(2)`
 *
 * @param flags Open flags given as a string
 * @param mode Access modes
 * @return File descriptor
 */
inline int open_fd(const std::string& file_path, const std::string& flags, mode_t mode)
{
  /*NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)*/
  int fd = ::open(file_path.c_str(), open_fd_parse_flags(flags), mode);
  if (fd == -1) { throw std::system_error(errno, std::generic_category(), "Unable to open file"); }
  return fd;
}

/**
 * @brief Get file size from file descriptor `fstat(3)`
 *
 * @param file_descriptor Open file descriptor
 * @return The number of bytes
 */
[[nodiscard]] inline std::size_t get_file_size(int file_descriptor)
{
  struct stat st {
  };
  int ret = fstat(file_descriptor, &st);
  if (ret == -1) {
    throw std::system_error(errno, std::generic_category(), "Unable to query file size");
  }
  return static_cast<std::size_t>(st.st_size);
}

}  // namespace

/**
 * @brief Handle of an open file registred with cufile.
 *
 * In order to utilize cufile and GDS, a file must be registred with cufile.
 */
class FileHandle {
 private:
  int _fd{-1};
  bool _own_fd{false};
  bool _closed{true};
  mutable std::size_t _nbytes{0};  // The size of the underlying file, zero means unknown.
  CUfileHandle_t _handle{};

 public:
  static constexpr mode_t m644 = S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH;
  FileHandle()                 = default;

  /**
   * @brief Construct a file handle from an existing file decriptor
   *
   * @param fd File decriptor
   * @param steal_fd When true, the handle owns the file descriptor and will close it
   * on destruction.
   */
  FileHandle(int fd, bool steal_fd = false) : _fd{fd}, _own_fd{steal_fd}, _closed{false}
  {
    if (config::get_global_compat_mode()) { return; }
#ifdef KVIKIO_CUFILE_EXIST
    CUfileDescr_t desc{};  // It is important to set zero!
    desc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    /*NOLINTNEXTLINE(cppcoreguidelines-pro-type-union-access)*/
    desc.handle.fd = fd;
    CUFILE_TRY(cuFileAPI::instance()->HandleRegister(&_handle, &desc));
#endif
  }

  /**
   * @brief Construct a file handle from a file path
   *
   * @param file_path File path to the file
   * @param steal_fd When true, the handle owns the file descriptor and will close it
   * on destruction. When false, the file is deregistered but not closed.
   */
  FileHandle(const std::string& file_path, const std::string& flags = "r", mode_t mode = m644)
    : FileHandle(open_fd(file_path, flags, mode), true)
  {
  }

  /**
   * @brief FileHandle support move semantic but isn't copyable
   */
  FileHandle(const FileHandle&) = delete;
  FileHandle& operator=(FileHandle const&) = delete;
  FileHandle(FileHandle&& o) noexcept
    : _fd{std::exchange(o._fd, -1)},
      _own_fd{std::exchange(o._own_fd, false)},
      _closed{std::exchange(o._closed, true)},
      _nbytes{std::exchange(o._nbytes, 0)},
      _handle{std::exchange(o._handle, CUfileHandle_t{})}
  {
  }
  FileHandle& operator=(FileHandle&& o) noexcept
  {
    _fd     = std::exchange(o._fd, -1);
    _own_fd = std::exchange(o._own_fd, false);
    _closed = std::exchange(o._closed, true);
    _nbytes = std::exchange(o._nbytes, 0);
    _handle = std::exchange(o._handle, CUfileHandle_t{});
    return *this;
  }

  /**
   * @brief FileHandle support move semantic but isn't copyable
   */
  ~FileHandle() noexcept
  {
    if (!_closed) { this->close(); }
  }

  [[nodiscard]] bool closed() const noexcept { return _closed; }

  /**
   * @brief Deregister the file and close the file if created with `steal_fd=true`
   */
  void close() noexcept
  {
    _closed = true;
#ifdef KVIKIO_CUFILE_EXIST
    if (!config::get_global_compat_mode()) { cuFileAPI::instance()->HandleDeregister(_handle); }
#endif
    if (_own_fd) { ::close(_fd); }
  }

  /**
   * @brief Get the file descripter of the open file
   * @return File descripter
   */
  [[nodiscard]] int fd() const noexcept { return _fd; }

  /**
   * @brief Get the flags of the file descripter (see open(2))
   * @return File descripter
   */
  [[nodiscard]] int fd_open_flags() const
  {
    /*NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)*/
    int ret = fcntl(_fd, F_GETFL);
    if (ret == -1) {
      throw std::system_error(errno, std::generic_category(), "Unable to retrieve open flags");
    }
    return ret;
  }

  /**
   * @brief Get the file size
   *
   * @return The number of bytes
   */
  [[nodiscard]] inline std::size_t nbytes() const
  {
    if (_nbytes == 0 && _fd > 0) { _nbytes = get_file_size(_fd); }
    return _nbytes;
  }

  /**
   * @brief Reads specified bytes from the file into the device memory.
   *
   * This API reads the data from the GPU memory to the file at a specified offset
   * and size bytes by using GDS functionality. The API works correctly for unaligned
   * offset and data sizes, although the performance is not on-par with aligned read.
   * This is a synchronous call and will block until the IO is complete.
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
  std::size_t read(void* devPtr_base,
                   std::size_t size,
                   std::size_t file_offset,
                   std::size_t devPtr_offset)
  {
    if (config::get_global_compat_mode()) {
      return posix_read(_fd, devPtr_base, size, file_offset, devPtr_offset);
    }
#ifdef KVIKIO_CUFILE_EXIST
    ssize_t ret = cuFileAPI::instance()->Read(
      _handle, devPtr_base, size, convert_size2off(file_offset), convert_size2off(devPtr_offset));
    if (ret == -1) {
      throw std::system_error(errno, std::generic_category(), "Unable to read file");
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
   * This API writes the data from the GPU memory to the file at a specified offset
   * and size bytes by using GDS functionality. The API works correctly for unaligned
   * offset and data sizes, although the performance is not on-par with aligned writes.
   * This is a synchronous call and will block until the IO is complete.
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
  std::size_t write(const void* devPtr_base,
                    std::size_t size,
                    std::size_t file_offset,
                    std::size_t devPtr_offset)
  {
    _nbytes = 0;  // Invalidate the computed file size

    if (config::get_global_compat_mode()) {
      return posix_write(_fd, devPtr_base, size, file_offset, devPtr_offset);
    }
#ifdef KVIKIO_CUFILE_EXIST
    ssize_t ret = cuFileAPI::instance()->Write(
      _handle, devPtr_base, size, convert_size2off(file_offset), convert_size2off(devPtr_offset));
    if (ret == -1) {
      throw std::system_error(errno, std::generic_category(), "Unable to write file");
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
   * @brief Reads specified bytes from the file into the device memory in parallel.
   *
   * This API is a parallel async version of `.read()` that create `ntasks` tasks
   * for the thread pool to execute.
   *
   * @note `pread` use the base address of the allocation `devPtr` is part of. This means
   * that when registering buffers, use the base address of the allocation. This is what
   * `memory_register` and `memory_deregister` do automatically.
   *
   * @param devPtr Address to device memory.
   * @param size Size in bytes to read.
   * @param file_offset Offset in the file to read from.
   * @param ntasks Number of tasks to use.
   * @return Future that on completion returns the size of bytes that were successfully read.
   */
  std::future<std::size_t> pread(void* devPtr,
                                 std::size_t size,
                                 std::size_t file_offset = 0,
                                 std::size_t ntasks      = default_thread_pool::nthreads())
  {
    // Lambda that calls this->read()
    auto op = [this](void* devPtr_base,
                     std::size_t size,
                     std::size_t file_offset,
                     std::size_t devPtr_offset) -> std::size_t {
      return read(devPtr_base, size, file_offset, devPtr_offset);
    };
    return parallel_io(op, devPtr, size, file_offset, ntasks);
  }

  /**
   * @brief Writes specified bytes from the device memory into the file in parallel.
   *
   * This API is a parallel async version of `.write()` that create `ntasks` tasks
   * for the thread pool to execute.
   *
   * @note `pwrite` use the base address of the allocation `devPtr` is part of. This means
   * that when registering buffers, use the base address of the allocation. This is what
   * `memory_register` and `memory_deregister` do automatically.
   *
   * @param devPtr Address to device memory.
   * @param size Size in bytes to write.
   * @param file_offset Offset in the file to write from.
   * @param ntasks Number of tasks to use.
   * @return Future that on completion returns the size of bytes that were successfully written.
   */
  std::future<std::size_t> pwrite(const void* devPtr,
                                  std::size_t size,
                                  std::size_t file_offset = 0,
                                  std::size_t ntasks      = default_thread_pool::nthreads())
  {
    // Lambda that calls this->write()
    auto op = [this](const void* devPtr_base,
                     std::size_t size,
                     std::size_t file_offset,
                     std::size_t devPtr_offset) -> std::size_t {
      return write(devPtr_base, size, file_offset, devPtr_offset);
    };
    return parallel_io(op, devPtr, size, file_offset, ntasks);
  }
};

}  // namespace kvikio
