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
#include <utility>

#include <kvikio/compat_mode.hpp>
#include <kvikio/defaults.hpp>
#include <kvikio/file_handle.hpp>
#include <kvikio/file_utils.hpp>

namespace kvikio {

FileHandle::FileHandle(std::string const& file_path,
                       std::string const& flags,
                       mode_t mode,
                       CompatMode compat_mode)
  : _initialized{true},
    _compat_mode_manager{
      file_path, flags, mode, compat_mode, _file_direct_on, _file_direct_off, _cufile_handle}
{
}

FileHandle::FileHandle(FileHandle&& o) noexcept
  : _file_direct_on{std::exchange(o._file_direct_on, {})},
    _file_direct_off{std::exchange(o._file_direct_off, {})},
    _initialized{std::exchange(o._initialized, false)},
    _nbytes{std::exchange(o._nbytes, 0)},
    _cufile_handle{std::exchange(o._cufile_handle, {})},
    _compat_mode_manager{std::move(o._compat_mode_manager)}
{
}

FileHandle& FileHandle::operator=(FileHandle&& o) noexcept
{
  _file_direct_on      = std::exchange(o._file_direct_on, {});
  _file_direct_off     = std::exchange(o._file_direct_off, {});
  _initialized         = std::exchange(o._initialized, false);
  _nbytes              = std::exchange(o._nbytes, 0);
  _cufile_handle       = std::exchange(o._cufile_handle, {});
  _compat_mode_manager = std::move(o._compat_mode_manager);
  return *this;
}

FileHandle::~FileHandle() noexcept { close(); }

bool FileHandle::closed() const noexcept { return !_initialized; }

void FileHandle::close() noexcept
{
  try {
    if (closed()) { return; }
    _cufile_handle.unregister_handle();
    _file_direct_off.close();
    _file_direct_on.close();
    _nbytes      = 0;
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
  return o_direct ? _file_direct_on.fd() : _file_direct_off.fd();
}

int FileHandle::fd_open_flags(bool o_direct) const { return open_flags(fd(o_direct)); }

std::size_t FileHandle::nbytes() const
{
  if (closed()) { return 0; }
  if (_nbytes == 0) { _nbytes = get_file_size(_file_direct_off.fd()); }
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
      _file_direct_off.fd(), devPtr_base, size, file_offset, devPtr_offset);
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
      _file_direct_off.fd(), devPtr_base, size, file_offset, devPtr_offset);
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
        _file_direct_off.fd(), buf, size, file_offset);
    };

    return parallel_io(op, buf, size, file_offset, task_size, 0);
  }

  CUcontext ctx = get_context_from_pointer(buf);

  // Shortcut that circumvent the threadpool and use the POSIX backend directly.
  if (size < gds_threshold) {
    auto task = [this, ctx, buf, size, file_offset]() -> std::size_t {
      PushAndPopContext c(ctx);
      return detail::posix_device_read(_file_direct_off.fd(), buf, size, file_offset, 0);
    };
    return std::async(std::launch::deferred, task);
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
        _file_direct_off.fd(), buf, size, file_offset);
    };

    return parallel_io(op, buf, size, file_offset, task_size, 0);
  }

  CUcontext ctx = get_context_from_pointer(buf);

  // Shortcut that circumvent the threadpool and use the POSIX backend directly.
  if (size < gds_threshold) {
    auto task = [this, ctx, buf, size, file_offset]() -> std::size_t {
      PushAndPopContext c(ctx);
      return detail::posix_device_write(_file_direct_off.fd(), buf, size, file_offset, 0);
    };
    return std::async(std::launch::deferred, task);
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
  _compat_mode_manager.validate_compat_mode_for_async();
  if (is_compat_mode_preferred_for_async()) {
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
  _compat_mode_manager.validate_compat_mode_for_async();
  if (is_compat_mode_preferred_for_async()) {
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

CompatMode FileHandle::compat_mode_requested() const noexcept
{
  return _compat_mode_manager.compat_mode_requested();
}

bool FileHandle::is_compat_mode_preferred() const noexcept
{
  return _compat_mode_manager.is_compat_mode_preferred();
}

bool FileHandle::is_compat_mode_preferred_for_async() const noexcept
{
  return _compat_mode_manager.is_compat_mode_preferred_for_async();
}

}  // namespace kvikio
