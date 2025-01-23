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

#include <kvikio/detail/file_handle_dep.hpp>
#include <kvikio/file_handle.hpp>
#include "kvikio/cufile/config.hpp"
#include "kvikio/shim/cufile.hpp"

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

}  // namespace

namespace kvikio {

namespace detail {
void FileHandleDependencyBase::set_file_handle(FileHandle* file_handle)
{
  _file_handle = file_handle;
}

bool FileHandleDependencyProduction::is_compat_mode_preferred() const noexcept
{
  return defaults::is_compat_mode_preferred(_file_handle->_compat_mode);
}

bool FileHandleDependencyProduction::is_compat_mode_preferred(CompatMode compat_mode) const noexcept
{
  return defaults::is_compat_mode_preferred(compat_mode);
}

int FileHandleDependencyProduction::open_fd(const std::string& file_path,
                                            const std::string& flags,
                                            bool o_direct,
                                            mode_t mode)
{
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
  int fd = ::open(file_path.c_str(), open_fd_parse_flags(flags, o_direct), mode);
  if (fd == -1) { throw std::system_error(errno, std::generic_category(), "Unable to open file"); }
  return fd;
}

void FileHandleDependencyProduction::close_fd(int fd) { ::close(fd); }

CUfileError_t FileHandleDependencyProduction::cuFile_handle_register(CUfileHandle_t* fh,
                                                                     CUfileDescr_t* descr)
{
  return cuFileAPI::instance().HandleRegister(fh, descr);
}

void FileHandleDependencyProduction::cuFile_handle_deregister(CUfileHandle_t fh)
{
  cuFileAPI::instance().HandleDeregister(fh);
}

bool FileHandleDependencyProduction::is_stream_api_available() noexcept
{
  return kvikio::is_stream_api_available();
}

const std::string& FileHandleDependencyProduction::config_path() { return kvikio::config_path(); }

}  // namespace detail
}  // namespace kvikio
