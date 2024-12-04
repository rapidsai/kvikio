/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
#include <sys/types.h>
#include <unistd.h>
#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <system_error>

#include <kvikio/defaults.hpp>
#include <kvikio/file_handle.hpp>

namespace kvikio {

namespace {

/**
 * @brief Parse open file flags given as a string and return oflags
 *
 * @param flags The flags
 * @param o_direct Append O_DIRECT to the open flags
 * @return oflags
 *
 * @throw std::invalid_argument if the specified flags are not supported.
 * @throw std::invalid_argument if `o_direct` is true, but `O_DIRECT` is not supported.
 */
int open_fd_parse_flags(const std::string& flags, bool o_direct)
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

/**
 * @brief Open file using `open(2)`
 *
 * @param flags Open flags given as a string
 * @param o_direct Append O_DIRECT to `flags`
 * @param mode Access modes
 * @return File descriptor
 */
int open_fd(const std::string& file_path, const std::string& flags, bool o_direct, mode_t mode)
{
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
  int fd = ::open(file_path.c_str(), open_fd_parse_flags(flags, o_direct), mode);
  if (fd == -1) { throw std::system_error(errno, std::generic_category(), "Unable to open file"); }
  return fd;
}

/**
 * @brief Get the flags of the file descriptor (see `open(2)`)
 *
 * @return Open flags
 */
[[nodiscard]] int open_flags(int fd)
{
  int ret = fcntl(fd, F_GETFL);  // NOLINT(cppcoreguidelines-pro-type-vararg)
  if (ret == -1) {
    throw std::system_error(errno, std::generic_category(), "Unable to retrieve open flags");
  }
  return ret;
}

/**
 * @brief Get file size from file descriptor `fstat(3)`
 *
 * @param file_descriptor Open file descriptor
 * @return The number of bytes
 */
[[nodiscard]] std::size_t get_file_size(int file_descriptor)
{
  struct stat st {};
  int ret = fstat(file_descriptor, &st);
  if (ret == -1) {
    throw std::system_error(errno, std::generic_category(), "Unable to query file size");
  }
  return static_cast<std::size_t>(st.st_size);
}

}  // namespace

FileHandle::FileHandle(const std::string& file_path,
                       const std::string& flags,
                       mode_t mode,
                       CompatMode compat_mode)
  : _fd_direct_off{open_fd(file_path, flags, false, mode)},
    _initialized{true},
    _compat_mode{compat_mode}
{
  if (is_compat_mode_preferred()) {
    return;  // Nothing to do in compatibility mode
  }

  // Try to open the file with the O_DIRECT flag. Fall back to compatibility mode, if it fails.
  auto handle_0_direct_except = [this] {
    if (_compat_mode == CompatMode::AUTO) {
      _compat_mode = CompatMode::ON;
    } else {  // CompatMode::OFF
      throw;
    }
  };

  try {
    _fd_direct_on = open_fd(file_path, flags, true, mode);
  } catch (const std::system_error&) {
    handle_0_direct_except();
  } catch (const std::invalid_argument&) {
    handle_0_direct_except();
  }

  if (_compat_mode == CompatMode::ON) { return; }

  // Create a cuFile handle, if not in compatibility mode
  CUfileDescr_t desc{};  // It is important to set to zero!
  desc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-union-access)
  desc.handle.fd = _fd_direct_on;

  auto error_code = cuFileAPI::instance().HandleRegister(&_handle, &desc);
  // For the AUTO mode, if the first cuFile API call fails, fall back to the compatibility
  // mode.
  if (_compat_mode == CompatMode::AUTO && error_code.err != CU_FILE_SUCCESS) {
    _compat_mode = CompatMode::ON;
  } else {
    CUFILE_TRY(error_code);
  }
}

[[nodiscard]] int FileHandle::fd_open_flags() const { return open_flags(_fd_direct_off); }

[[nodiscard]] std::size_t FileHandle::nbytes() const
{
  if (closed()) { return 0; }
  if (_nbytes == 0) { _nbytes = get_file_size(_fd_direct_off); }
  return _nbytes;
}

}  // namespace kvikio
