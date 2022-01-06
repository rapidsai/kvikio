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
#include <unistd.h>

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <optional>
#include <system_error>
#include <utility>

#include <cufile.h>
#include <cufile/buffer.hpp>
#include <cufile/error.hpp>
#include <cufile/parallel_operation.hpp>
#include <cufile/thread_pool/default.hpp>
#include <cufile/utils.hpp>

namespace cufile {
namespace {

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
    case 'a':
      throw std::invalid_argument("Open flag 'a' isn't supported");
      file_flags = O_RDWR | O_CREAT;
      break;
    default: throw std::invalid_argument("Unknown file open flag");
  }
  file_flags |= O_CLOEXEC;
  file_flags |= O_DIRECT;
  return file_flags;
}

inline int open_fd(const std::string& file_path, const std::string& flags, mode_t mode)
{
  /*NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)*/
  int fd = ::open(file_path.c_str(), open_fd_parse_flags(flags), mode);
  if (fd == -1) { throw std::system_error(errno, std::generic_category(), "Unable to open file"); }
  return fd;
}

}  // namespace

class FileHandle {
 private:
  int _fd{-1};
  bool _own_fd{false};
  bool _closed{true};
  CUfileHandle_t _handle{};

 public:
  static constexpr mode_t m644 = S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH;
  FileHandle()                 = default;

  FileHandle(int fd, bool steal_fd = false) : _fd{fd}, _own_fd{steal_fd}, _closed{false}
  {
    CUfileDescr_t desc{};  // It is important to set zero!
    desc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    /*NOLINTNEXTLINE(cppcoreguidelines-pro-type-union-access)*/
    desc.handle.fd = fd;
    CUFILE_TRY(cuFileHandleRegister(&_handle, &desc));
  }

  FileHandle(const std::string& file_path, const std::string& flags = "r", mode_t mode = m644)
    : FileHandle(open_fd(file_path, flags, mode), true)
  {
  }

  // We implement move semantic only
  FileHandle(const FileHandle&) = delete;
  FileHandle& operator=(FileHandle const&) = delete;

  FileHandle(FileHandle&& o) noexcept
    : _fd{std::exchange(o._fd, -1)},
      _own_fd{std::exchange(o._own_fd, false)},
      _closed{std::exchange(o._closed, true)},
      _handle{std::exchange(o._handle, CUfileHandle_t{})}
  {
  }

  FileHandle& operator=(FileHandle&& o) noexcept
  {
    _fd     = std::exchange(o._fd, -1);
    _own_fd = std::exchange(o._own_fd, false);
    _closed = std::exchange(o._closed, true);
    _handle = std::exchange(o._handle, CUfileHandle_t{});
    return *this;
  }

  ~FileHandle() noexcept
  {
    if (!_closed) { this->close(); }
  }

  [[nodiscard]] bool closed() const noexcept { return _closed; }

  void close() noexcept
  {
    _closed = true;
    cuFileHandleDeregister(_handle);
    if (_own_fd) { ::close(_fd); }
  }

  [[nodiscard]] int fd() const noexcept { return _fd; }

  [[nodiscard]] int fd_open_flags() const
  {
    /*NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)*/
    int ret = fcntl(_fd, F_GETFL);
    if (ret == -1) {
      throw std::system_error(errno, std::generic_category(), "Unable to retrieve open flags");
    }
    return ret;
  }

  std::size_t read(void* devPtr_base,
                   std::size_t size,
                   std::size_t file_offset,
                   std::size_t devPtr_offset)
  {
    ssize_t ret = cuFileRead(
      _handle, devPtr_base, size, convert_size2off(file_offset), convert_size2off(devPtr_offset));
    if (ret == -1) {
      throw std::system_error(errno, std::generic_category(), "Unable to read file");
    }
    if (ret < -1) {
      throw CUfileException(std::string{"cuFile error at: "} + __FILE__ + ":" +
                            CUFILE_STRINGIFY(__LINE__) + ": " + CUFILE_ERRSTR(ret));
    }
    return ret;
  }

  std::size_t write(const void* devPtr_base,
                    std::size_t size,
                    std::size_t file_offset,
                    std::size_t devPtr_offset)
  {
    ssize_t ret = cuFileWrite(
      _handle, devPtr_base, size, convert_size2off(file_offset), convert_size2off(devPtr_offset));
    if (ret == -1) {
      throw std::system_error(errno, std::generic_category(), "Unable to write file");
    }
    if (ret < -1) {
      throw CUfileException(std::string{"cuFile error at: "} + __FILE__ + ":" +
                            CUFILE_STRINGIFY(__LINE__) + ": " + CUFILE_ERRSTR(ret));
    }
    return ret;
  }

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

}  // namespace cufile
