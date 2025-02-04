/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdexcept>
#include <system_error>
#include <utility>

#include <kvikio/file_utils.hpp>
#include <kvikio/shim/cufile.hpp>

namespace kvikio {

FileWrapper::FileWrapper(std::string const& file_path,
                         std::string const& flags,
                         bool o_direct,
                         mode_t mode)
{
  open(file_path, flags, o_direct, mode);
}

FileWrapper::~FileWrapper() noexcept { close(); }

FileWrapper::FileWrapper(FileWrapper&& o) noexcept : _fd(std::exchange(o._fd, -1)) {}

FileWrapper& FileWrapper::operator=(FileWrapper&& o) noexcept
{
  _fd = std::exchange(o._fd, -1);
  return *this;
}

void FileWrapper::open(std::string const& file_path,
                       std::string const& flags,
                       bool o_direct,
                       mode_t mode)
{
  if (!opened()) { _fd = open_fd(file_path, flags, o_direct, mode); }
}

bool FileWrapper::opened() const noexcept { return _fd != -1; }

void FileWrapper::close() noexcept
{
  if (opened()) {
    ::close(_fd);
    _fd = -1;
  }
}

int FileWrapper::fd() const noexcept { return _fd; }

CUFileHandleWrapper::~CUFileHandleWrapper() noexcept { unregister_handle(); }

CUFileHandleWrapper::CUFileHandleWrapper(CUFileHandleWrapper&& o) noexcept
  : _handle{std::exchange(o._handle, {})}, _registered{std::exchange(o._registered, false)}
{
}

CUFileHandleWrapper& CUFileHandleWrapper::operator=(CUFileHandleWrapper&& o) noexcept
{
  _handle     = std::exchange(o._handle, {});
  _registered = std::exchange(o._registered, false);
  return *this;
}

std::optional<CUfileError_t> CUFileHandleWrapper::register_handle(int fd) noexcept
{
  std::optional<CUfileError_t> error_code;
  if (registered()) { return error_code; }

  // Create a cuFile handle, if not in compatibility mode
  CUfileDescr_t desc{};  // It is important to set to zero!
  desc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-union-access)
  desc.handle.fd = fd;
  error_code     = cuFileAPI::instance().HandleRegister(&_handle, &desc);
  if (error_code.value().err == CU_FILE_SUCCESS) { _registered = true; }
  return error_code;
}

bool CUFileHandleWrapper::registered() const noexcept { return _registered; }

CUfileHandle_t CUFileHandleWrapper::handle() const noexcept { return _handle; }

void CUFileHandleWrapper::unregister_handle() noexcept
{
  if (registered()) {
    cuFileAPI::instance().HandleDeregister(_handle);
    _registered = false;
  }
}

int open_fd_parse_flags(std::string const& flags, bool o_direct)
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
  if (o_direct) {
#if defined(O_DIRECT)
    file_flags |= O_DIRECT;
#else
    throw std::invalid_argument("'o_direct' flag unsupported on this platform");
#endif
  }
  return file_flags;
}

int open_fd(std::string const& file_path, std::string const& flags, bool o_direct, mode_t mode)
{
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
  int fd = ::open(file_path.c_str(), open_fd_parse_flags(flags, o_direct), mode);
  if (fd == -1) { throw std::system_error(errno, std::generic_category(), "Unable to open file"); }
  return fd;
}

[[nodiscard]] int open_flags(int fd)
{
  int ret = fcntl(fd, F_GETFL);  // NOLINT(cppcoreguidelines-pro-type-vararg)
  if (ret == -1) {
    throw std::system_error(errno, std::generic_category(), "Unable to retrieve open flags");
  }
  return ret;
}

[[nodiscard]] std::size_t get_file_size(int file_descriptor)
{
  struct stat st {};
  int ret = fstat(file_descriptor, &st);
  if (ret == -1) {
    throw std::system_error(errno, std::generic_category(), "Unable to query file size");
  }
  return static_cast<std::size_t>(st.st_size);
}

}  // namespace kvikio
