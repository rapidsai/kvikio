/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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
#include <kvikio/file_utils.hpp>
#include "kvikio/utils.hpp"

namespace kvikio {

FileHandle::FileHandle(std::string const& file_path,
                       std::string const& flags,
                       mode_t mode,
                       CompatMode compat_mode)
  : _fd_direct_off{file_path, flags, false, mode}, _initialized{true}, _compat_mode{compat_mode}
{
  if (is_compat_mode_preferred()) {
    return;  // Nothing to do in compatibility mode
  }

  // Try to open the file with the O_DIRECT flag. Fall back to compatibility mode, if it fails.
  auto handle_o_direct_except = [this] {
    if (_compat_mode == CompatMode::AUTO) {
      _compat_mode = CompatMode::ON;
    } else {  // CompatMode::OFF
      throw;
    }
  };

  try {
    _fd_direct_on.open(file_path, flags, true, mode);
  } catch (std::system_error const&) {
    handle_o_direct_except();
  } catch (std::invalid_argument const&) {
    handle_o_direct_except();
  }

  if (_compat_mode == CompatMode::ON) { return; }

  CUFileHandleWrapper handle;
  auto error_code = handle.register_handle(_fd_direct_on.fd());
  assert(error_code.has_value());

  // For the AUTO mode, if the first cuFile API call fails, fall back to the compatibility
  // mode.
  if (_compat_mode == CompatMode::AUTO && error_code.value().err != CU_FILE_SUCCESS) {
    _compat_mode = CompatMode::ON;
  } else {
    CUFILE_TRY(error_code.value());
  }
}

FileHandle::FileHandle(FileHandle&& o) noexcept
  : _fd_direct_on{std::exchange(o._fd_direct_on, {})},
    _fd_direct_off{std::exchange(o._fd_direct_off, {})},
    _initialized{std::exchange(o._initialized, false)},
    _compat_mode{std::exchange(o._compat_mode, CompatMode::AUTO)},
    _nbytes{std::exchange(o._nbytes, 0)},
    _cufile_handle{std::exchange(o._cufile_handle, {})}
{
}

FileHandle& FileHandle::operator=(FileHandle&& o) noexcept
{
  _fd_direct_on  = std::exchange(o._fd_direct_on, {});
  _fd_direct_off = std::exchange(o._fd_direct_off, {});
  _initialized   = std::exchange(o._initialized, false);
  _compat_mode   = std::exchange(o._compat_mode, CompatMode::AUTO);
  _nbytes        = std::exchange(o._nbytes, 0);
  _cufile_handle = std::exchange(o._cufile_handle, {});
  return *this;
}

FileHandle::~FileHandle() noexcept { close(); }

bool FileHandle::closed() const noexcept { return !_initialized; }

void FileHandle::close() noexcept
{
  try {
    if (closed()) { return; }

    _cufile_handle.unregister_handle();
    _compat_mode = CompatMode::AUTO;
    _fd_direct_off.close();
    _fd_direct_on.close();
    _initialized = false;
  } catch (...) {
  }
}

CUfileHandle_t FileHandle::handle()
{
  if (closed()) { throw CUfileException("File handle is closed"); }
  if (is_compat_mode_preferred()) {
    throw CUfileException("The underlying cuFile handle isn't available in compatibility mode");
  }
  return _cufile_handle.handle();
}

int FileHandle::fd(bool o_direct) const noexcept
{
  return o_direct ? _fd_direct_on.fd() : _fd_direct_off.fd();
}

int FileHandle::fd_open_flags(bool o_direct) const { return open_flags(fd(o_direct)); }

std::size_t FileHandle::nbytes() const
{
  if (closed()) { return 0; }
  if (_nbytes == 0) { _nbytes = get_file_size(_fd_direct_off.fd()); }
  return _nbytes;
}

std::size_t FileHandle::read(void* devPtr_base,
                             std::size_t size,
                             std::size_t file_offset,
                             std::size_t devPtr_offset,
                             bool sync_default_stream)
{
  if (is_compat_mode_preferred()) {
    return detail::posix_device_read(
      _fd_direct_off.fd(), devPtr_base, size, file_offset, devPtr_offset);
  }
  if (sync_default_stream) { CUDA_DRIVER_TRY(cudaAPI::instance().StreamSynchronize(nullptr)); }

  KVIKIO_NVTX_SCOPED_RANGE("cufileRead()", size);
  ssize_t ret = cuFileAPI::instance().Read(_cufile_handle.handle(),
                                           devPtr_base,
                                           size,
                                           convert_size2off(file_offset),
                                           convert_size2off(devPtr_offset));
  CUFILE_CHECK_BYTES_DONE(ret);
  return ret;
}

std::size_t FileHandle::write(void const* devPtr_base,
                              std::size_t size,
                              std::size_t file_offset,
                              std::size_t devPtr_offset,
                              bool sync_default_stream)
{
  _nbytes = 0;  // Invalidate the computed file size

  if (is_compat_mode_preferred()) {
    return detail::posix_device_write(
      _fd_direct_off.fd(), devPtr_base, size, file_offset, devPtr_offset);
  }
  if (sync_default_stream) { CUDA_DRIVER_TRY(cudaAPI::instance().StreamSynchronize(nullptr)); }

  KVIKIO_NVTX_SCOPED_RANGE("cufileWrite()", size);
  ssize_t ret = cuFileAPI::instance().Write(_cufile_handle.handle(),
                                            devPtr_base,
                                            size,
                                            convert_size2off(file_offset),
                                            convert_size2off(devPtr_offset));
  if (ret == -1) {
    throw std::system_error(errno, std::generic_category(), "Unable to write file");
  }
  if (ret < -1) {
    throw CUfileException(std::string{"cuFile error at: "} + __FILE__ + ":" +
                          KVIKIO_STRINGIFY(__LINE__) + ": " + CUFILE_ERRSTR(ret));
  }
  return ret;
}

std::future<std::size_t> FileHandle::pread(void* buf,
                                           std::size_t size,
                                           std::size_t file_offset,
                                           std::size_t task_size,
                                           std::size_t gds_threshold,
                                           bool sync_default_stream)
{
  KVIKIO_NVTX_MARKER("FileHandle::pread()", size);
  if (is_host_memory(buf)) {
    auto op = [this](void* hostPtr_base,
                     std::size_t size,
                     std::size_t file_offset,
                     std::size_t hostPtr_offset) -> std::size_t {
      char* buf = static_cast<char*>(hostPtr_base) + hostPtr_offset;
      return detail::posix_host_read<detail::PartialIO::NO>(
        _fd_direct_off.fd(), buf, size, file_offset);
    };

    return parallel_io(op, buf, size, file_offset, task_size, 0);
  }

  CUcontext ctx = get_context_from_pointer(buf);

  // Shortcut that circumvent the threadpool and use the POSIX backend directly.
  if (size < gds_threshold) {
    PushAndPopContext c(ctx);
    auto bytes_read = detail::posix_device_read(_fd_direct_off.fd(), buf, size, file_offset, 0);
    // Maintain API consistency while making this trivial case synchronous.
    // The result in the future is immediately available after the call.
    return make_ready_future(bytes_read);
  }

  // Let's synchronize once instead of in each task.
  if (sync_default_stream && !is_compat_mode_preferred()) {
    PushAndPopContext c(ctx);
    CUDA_DRIVER_TRY(cudaAPI::instance().StreamSynchronize(nullptr));
  }

  // Regular case that use the threadpool and run the tasks in parallel
  auto task = [this, ctx](void* devPtr_base,
                          std::size_t size,
                          std::size_t file_offset,
                          std::size_t devPtr_offset) -> std::size_t {
    PushAndPopContext c(ctx);
    return read(devPtr_base, size, file_offset, devPtr_offset, /* sync_default_stream = */ false);
  };
  auto [devPtr_base, base_size, devPtr_offset] = get_alloc_info(buf, &ctx);
  return parallel_io(task, devPtr_base, size, file_offset, task_size, devPtr_offset);
}

std::future<std::size_t> FileHandle::pwrite(void const* buf,
                                            std::size_t size,
                                            std::size_t file_offset,
                                            std::size_t task_size,
                                            std::size_t gds_threshold,
                                            bool sync_default_stream)
{
  KVIKIO_NVTX_MARKER("FileHandle::pwrite()", size);
  if (is_host_memory(buf)) {
    auto op = [this](void const* hostPtr_base,
                     std::size_t size,
                     std::size_t file_offset,
                     std::size_t hostPtr_offset) -> std::size_t {
      char const* buf = static_cast<char const*>(hostPtr_base) + hostPtr_offset;
      return detail::posix_host_write<detail::PartialIO::NO>(
        _fd_direct_off.fd(), buf, size, file_offset);
    };

    return parallel_io(op, buf, size, file_offset, task_size, 0);
  }

  CUcontext ctx = get_context_from_pointer(buf);

  // Shortcut that circumvent the threadpool and use the POSIX backend directly.
  if (size < gds_threshold) {
    PushAndPopContext c(ctx);
    auto bytes_write = detail::posix_device_write(_fd_direct_off.fd(), buf, size, file_offset, 0);
    // Maintain API consistency while making this trivial case synchronous.
    // The result in the future is immediately available after the call.
    return make_ready_future(bytes_write);
  }

  // Let's synchronize once instead of in each task.
  if (sync_default_stream && !is_compat_mode_preferred()) {
    PushAndPopContext c(ctx);
    CUDA_DRIVER_TRY(cudaAPI::instance().StreamSynchronize(nullptr));
  }

  // Regular case that use the threadpool and run the tasks in parallel
  auto op = [this, ctx](void const* devPtr_base,
                        std::size_t size,
                        std::size_t file_offset,
                        std::size_t devPtr_offset) -> std::size_t {
    PushAndPopContext c(ctx);
    return write(devPtr_base, size, file_offset, devPtr_offset, /* sync_default_stream = */ false);
  };
  auto [devPtr_base, base_size, devPtr_offset] = get_alloc_info(buf, &ctx);
  return parallel_io(op, devPtr_base, size, file_offset, task_size, devPtr_offset);
}

void FileHandle::read_async(void* devPtr_base,
                            std::size_t* size_p,
                            off_t* file_offset_p,
                            off_t* devPtr_offset_p,
                            ssize_t* bytes_read_p,
                            CUstream stream)
{
  if (is_compat_mode_preferred_for_async(_compat_mode)) {
    CUDA_DRIVER_TRY(cudaAPI::instance().StreamSynchronize(stream));
    *bytes_read_p =
      static_cast<ssize_t>(read(devPtr_base, *size_p, *file_offset_p, *devPtr_offset_p));
  } else {
    CUFILE_TRY(cuFileAPI::instance().ReadAsync(_cufile_handle.handle(),
                                               devPtr_base,
                                               size_p,
                                               file_offset_p,
                                               devPtr_offset_p,
                                               bytes_read_p,
                                               stream));
  }
}

StreamFuture FileHandle::read_async(
  void* devPtr_base, std::size_t size, off_t file_offset, off_t devPtr_offset, CUstream stream)
{
  StreamFuture ret(devPtr_base, size, file_offset, devPtr_offset, stream);
  auto [devPtr_base_, size_p, file_offset_p, devPtr_offset_p, bytes_read_p, stream_] =
    ret.get_args();
  read_async(devPtr_base_, size_p, file_offset_p, devPtr_offset_p, bytes_read_p, stream_);
  return ret;
}

void FileHandle::write_async(void* devPtr_base,
                             std::size_t* size_p,
                             off_t* file_offset_p,
                             off_t* devPtr_offset_p,
                             ssize_t* bytes_written_p,
                             CUstream stream)
{
  if (is_compat_mode_preferred_for_async(_compat_mode)) {
    CUDA_DRIVER_TRY(cudaAPI::instance().StreamSynchronize(stream));
    *bytes_written_p =
      static_cast<ssize_t>(write(devPtr_base, *size_p, *file_offset_p, *devPtr_offset_p));
  } else {
    CUFILE_TRY(cuFileAPI::instance().WriteAsync(_cufile_handle.handle(),
                                                devPtr_base,
                                                size_p,
                                                file_offset_p,
                                                devPtr_offset_p,
                                                bytes_written_p,
                                                stream));
  }
}

StreamFuture FileHandle::write_async(
  void* devPtr_base, std::size_t size, off_t file_offset, off_t devPtr_offset, CUstream stream)
{
  StreamFuture ret(devPtr_base, size, file_offset, devPtr_offset, stream);
  auto [devPtr_base_, size_p, file_offset_p, devPtr_offset_p, bytes_written_p, stream_] =
    ret.get_args();
  write_async(devPtr_base_, size_p, file_offset_p, devPtr_offset_p, bytes_written_p, stream_);
  return ret;
}

bool FileHandle::is_compat_mode_preferred() const noexcept
{
  return defaults::is_compat_mode_preferred(_compat_mode);
}

bool FileHandle::is_compat_mode_preferred_for_async() const noexcept
{
  static bool is_extra_symbol_available = is_stream_api_available();
  static bool is_config_path_empty      = config_path().empty();
  return is_compat_mode_preferred() || !is_extra_symbol_available || is_config_path_empty;
}

bool FileHandle::is_compat_mode_preferred_for_async(CompatMode requested_compat_mode)
{
  if (defaults::is_compat_mode_preferred(requested_compat_mode)) { return true; }

  if (!is_stream_api_available()) {
    if (requested_compat_mode == CompatMode::AUTO) { return true; }
    throw std::runtime_error("Missing the cuFile stream api.");
  }

  // When checking for availability, we also check if cuFile's config file exists. This is
  // because even when the stream API is available, it doesn't work if no config file exists.
  if (config_path().empty()) {
    if (requested_compat_mode == CompatMode::AUTO) { return true; }
    throw std::runtime_error("Missing cuFile configuration file.");
  }
  return false;
}

}  // namespace kvikio
