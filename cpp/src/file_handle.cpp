/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <unordered_map>
#include <utility>

#include <kvikio/compat_mode.hpp>
#include <kvikio/defaults.hpp>
#include <kvikio/detail/nvtx.hpp>
#include <kvikio/detail/parallel_operation.hpp>
#include <kvikio/detail/posix_io.hpp>
#include <kvikio/error.hpp>
#include <kvikio/file_handle.hpp>
#include <kvikio/file_utils.hpp>
#include <kvikio/threadpool_wrapper.hpp>

namespace kvikio {

namespace {

/**
 * @brief Get a thread pool specific to the block device hosting the given file.
 *
 * Thread pools are created lazily on first access and cached for subsequent lookups.
 * Two levels of caching are used:
 * - File path --> thread pool (fast path for repeated access to the same file)
 * - Block device ID --> thread pool (groups different files on the same device)
 *
 * @param file_path Path to the file where a thread pool is requested for the underlying block
 * device.
 * @return Pointer to the appropriate thread pool. The pointer remains valid for the lifetime of the
 * program (static storage duration).
 *
 * @note If device detection fails for any reason (e.g., unsupported filesystem, permission issues),
 * the default global thread pool is returned and an error is logged.
 */
ThreadPool* get_thread_pool_per_block_device(std::string const& file_path)
{
  KVIKIO_NVTX_FUNC_RANGE();

  if (!defaults::thread_pool_per_block_device()) { return &defaults::thread_pool(); }

  static std::mutex mtx;
  static std::unordered_map<std::string, std::shared_ptr<ThreadPool>> file_path_to_thread_pool_map;
  static std::unordered_map<dev_t, std::shared_ptr<ThreadPool>> dev_id_to_thread_pool_map;

  try {
    // Fast path: check if this exact file path has been seen before
    {
      std::lock_guard lock(mtx);
      if (auto it = file_path_to_thread_pool_map.find(file_path);
          it != file_path_to_thread_pool_map.end()) {
        return it->second.get();
      }
    }

    // Resolve file path to its underlying block device
    auto block_dev_info = get_block_device_info(file_path);

    // Check if we already have a thread pool for this block device
    std::lock_guard lock(mtx);
    if (auto it = dev_id_to_thread_pool_map.find(block_dev_info.id);
        it != dev_id_to_thread_pool_map.end()) {
      // Cache the file path mapping for future fast-path lookups
      file_path_to_thread_pool_map.emplace(file_path, it->second);
      return it->second.get();
    }

    // First file on this block device: create a new dedicated thread pool
    auto thread_pool = std::make_shared<ThreadPool>(defaults::num_threads());
    dev_id_to_thread_pool_map.emplace(block_dev_info.id, thread_pool);
    file_path_to_thread_pool_map.emplace(file_path, thread_pool);
    return thread_pool.get();
  } catch (std::exception const& ex) {
    std::string const& msg = std::string(ex.what()) + " Falling back to the default thread pool.";
    KVIKIO_LOG_ERROR(msg);
    return &defaults::thread_pool();
  }
}
}  // namespace

FileHandle::FileHandle(std::string const& file_path,
                       std::string const& flags,
                       mode_t mode,
                       CompatMode compat_mode)
  : _initialized{true}, _compat_mode_manager{file_path, flags, mode, compat_mode, this}
{
  KVIKIO_NVTX_FUNC_RANGE();
  _thread_pool = get_thread_pool_per_block_device(file_path);
}

FileHandle::FileHandle(FileHandle&& o) noexcept
  : _file_direct_on{std::exchange(o._file_direct_on, {})},
    _file_direct_off{std::exchange(o._file_direct_off, {})},
    _initialized{std::exchange(o._initialized, false)},
    _nbytes{std::exchange(o._nbytes, 0)},
    _cufile_handle{std::exchange(o._cufile_handle, {})},
    _compat_mode_manager{std::move(o._compat_mode_manager)},
    _thread_pool{std::exchange(o._thread_pool, {})}
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
  _thread_pool         = std::exchange(o._thread_pool, {});
  return *this;
}

FileHandle::~FileHandle() noexcept
{
  KVIKIO_NVTX_FUNC_RANGE();
  close();
}

bool FileHandle::closed() const noexcept { return !_initialized; }

void FileHandle::close() noexcept
{
  KVIKIO_NVTX_FUNC_RANGE();
  try {
    if (closed()) { return; }
    _cufile_handle.unregister_handle();
    _file_direct_off.close();
    _file_direct_on.close();
    _nbytes      = 0;
    _initialized = false;
    _thread_pool = nullptr;
  } catch (...) {
  }
}

CUfileHandle_t FileHandle::handle()
{
  KVIKIO_EXPECT(!closed(), "File handle is closed");
  KVIKIO_EXPECT(!get_compat_mode_manager().is_compat_mode_preferred(),
                "The underlying cuFile handle isn't available in compatibility mode");
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
  KVIKIO_NVTX_FUNC_RANGE(size);
  if (get_compat_mode_manager().is_compat_mode_preferred()) {
    return detail::posix_device_read(
      _file_direct_off.fd(), devPtr_base, size, file_offset, devPtr_offset, _file_direct_on.fd());
  }
  if (sync_default_stream) { CUDA_DRIVER_TRY(cudaAPI::instance().StreamSynchronize(nullptr)); }

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
  KVIKIO_NVTX_FUNC_RANGE(size);
  _nbytes = 0;  // Invalidate the computed file size

  if (get_compat_mode_manager().is_compat_mode_preferred()) {
    return detail::posix_device_write(
      _file_direct_off.fd(), devPtr_base, size, file_offset, devPtr_offset, _file_direct_on.fd());
  }
  if (sync_default_stream) { CUDA_DRIVER_TRY(cudaAPI::instance().StreamSynchronize(nullptr)); }

  ssize_t ret = cuFileAPI::instance().Write(_cufile_handle.handle(),
                                            devPtr_base,
                                            size,
                                            convert_size2off(file_offset),
                                            convert_size2off(devPtr_offset));
  KVIKIO_EXPECT(ret != -1, "Unable to write file", GenericSystemError);
  KVIKIO_EXPECT(ret >= 0, std::string{"cuFile error:"} + CUFILE_ERRSTR(ret));
  return ret;
}

std::future<std::size_t> FileHandle::pread(void* buf,
                                           std::size_t size,
                                           std::size_t file_offset,
                                           std::size_t task_size,
                                           std::size_t gds_threshold,
                                           bool sync_default_stream,
                                           ThreadPool* thread_pool)
{
  KVIKIO_EXPECT(thread_pool != nullptr, "The thread pool must not be nullptr");
  auto* actual_thread_pool{thread_pool};
  // Use the block-device-specific pool only if it exists and the user didn't explicitly provide a
  // custom pool
  if (_thread_pool != nullptr && thread_pool == &defaults::thread_pool()) {
    actual_thread_pool = _thread_pool;
  }

  auto& [nvtx_color, call_idx] = detail::get_next_color_and_call_idx();
  KVIKIO_NVTX_FUNC_RANGE(size, nvtx_color);
  if (is_host_memory(buf)) {
    auto op = [this](void* hostPtr_base,
                     std::size_t size,
                     std::size_t file_offset,
                     std::size_t hostPtr_offset) -> std::size_t {
      char* buf = static_cast<char*>(hostPtr_base) + hostPtr_offset;
      return detail::posix_host_read<detail::PartialIO::NO>(
        _file_direct_off.fd(), buf, size, file_offset, _file_direct_on.fd());
    };

    return parallel_io(
      op, buf, size, file_offset, task_size, 0, actual_thread_pool, call_idx, nvtx_color);
  }

  CUcontext ctx = get_context_from_pointer(buf);

  // Shortcut that circumvent the threadpool and use the POSIX backend directly.
  if (size < gds_threshold) {
    PushAndPopContext c(ctx);
    auto bytes_read = detail::posix_device_read(
      _file_direct_off.fd(), buf, size, file_offset, 0, _file_direct_on.fd());
    // Maintain API consistency while making this trivial case synchronous.
    // The result in the future is immediately available after the call.
    return make_ready_future(bytes_read);
  }

  // Let's synchronize once instead of in each task.
  if (sync_default_stream && !get_compat_mode_manager().is_compat_mode_preferred()) {
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
  return parallel_io(task,
                     devPtr_base,
                     size,
                     file_offset,
                     task_size,
                     devPtr_offset,
                     actual_thread_pool,
                     call_idx,
                     nvtx_color);
}

std::future<std::size_t> FileHandle::pwrite(void const* buf,
                                            std::size_t size,
                                            std::size_t file_offset,
                                            std::size_t task_size,
                                            std::size_t gds_threshold,
                                            bool sync_default_stream,
                                            ThreadPool* thread_pool)
{
  KVIKIO_EXPECT(thread_pool != nullptr, "The thread pool must not be nullptr");
  auto* actual_thread_pool{thread_pool};
  // Use the block-device-specific pool only if it exists and the user didn't explicitly provide a
  // custom pool
  if (_thread_pool != nullptr && thread_pool == &defaults::thread_pool()) {
    actual_thread_pool = _thread_pool;
  }

  auto& [nvtx_color, call_idx] = detail::get_next_color_and_call_idx();
  KVIKIO_NVTX_FUNC_RANGE(size, nvtx_color);
  if (is_host_memory(buf)) {
    auto op = [this](void const* hostPtr_base,
                     std::size_t size,
                     std::size_t file_offset,
                     std::size_t hostPtr_offset) -> std::size_t {
      char const* buf = static_cast<char const*>(hostPtr_base) + hostPtr_offset;
      return detail::posix_host_write<detail::PartialIO::NO>(
        _file_direct_off.fd(), buf, size, file_offset, _file_direct_on.fd());
    };

    return parallel_io(
      op, buf, size, file_offset, task_size, 0, actual_thread_pool, call_idx, nvtx_color);
  }

  CUcontext ctx = get_context_from_pointer(buf);

  // Shortcut that circumvent the threadpool and use the POSIX backend directly.
  if (size < gds_threshold) {
    PushAndPopContext c(ctx);
    auto bytes_write = detail::posix_device_write(
      _file_direct_off.fd(), buf, size, file_offset, 0, _file_direct_on.fd());
    // Maintain API consistency while making this trivial case synchronous.
    // The result in the future is immediately available after the call.
    return make_ready_future(bytes_write);
  }

  // Let's synchronize once instead of in each task.
  if (sync_default_stream && !get_compat_mode_manager().is_compat_mode_preferred()) {
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
  return parallel_io(op,
                     devPtr_base,
                     size,
                     file_offset,
                     task_size,
                     devPtr_offset,
                     actual_thread_pool,
                     call_idx,
                     nvtx_color);
}

void FileHandle::read_async(void* devPtr_base,
                            std::size_t* size_p,
                            off_t* file_offset_p,
                            off_t* devPtr_offset_p,
                            ssize_t* bytes_read_p,
                            CUstream stream)
{
  KVIKIO_NVTX_FUNC_RANGE();
  get_compat_mode_manager().validate_compat_mode_for_async();
  if (get_compat_mode_manager().is_compat_mode_preferred_for_async()) {
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
  KVIKIO_NVTX_FUNC_RANGE();
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
  KVIKIO_NVTX_FUNC_RANGE();
  get_compat_mode_manager().validate_compat_mode_for_async();
  if (get_compat_mode_manager().is_compat_mode_preferred_for_async()) {
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
  KVIKIO_NVTX_FUNC_RANGE();
  StreamFuture ret(devPtr_base, size, file_offset, devPtr_offset, stream);
  auto [devPtr_base_, size_p, file_offset_p, devPtr_offset_p, bytes_written_p, stream_] =
    ret.get_args();
  write_async(devPtr_base_, size_p, file_offset_p, devPtr_offset_p, bytes_written_p, stream_);
  return ret;
}

const CompatModeManager& FileHandle::get_compat_mode_manager() const noexcept
{
  return _compat_mode_manager;
}

bool FileHandle::is_direct_io_supported() const noexcept { return _file_direct_on.fd() != -1; }

}  // namespace kvikio
