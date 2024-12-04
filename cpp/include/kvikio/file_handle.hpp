/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

#include <sys/stat.h>
#include <sys/types.h>

#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <system_error>
#include <utility>

#include <kvikio/buffer.hpp>
#include <kvikio/cufile/config.hpp>
#include <kvikio/defaults.hpp>
#include <kvikio/error.hpp>
#include <kvikio/parallel_operation.hpp>
#include <kvikio/posix_io.hpp>
#include <kvikio/shim/cufile.hpp>
#include <kvikio/stream.hpp>
#include <kvikio/utils.hpp>

namespace kvikio {

/**
 * @brief Handle of an open file registered with cufile.
 *
 * In order to utilize cufile and GDS, a file must be registered with cufile.
 */
class FileHandle {
 private:
  // We use two file descriptors, one opened with the O_DIRECT flag and one without.
  int _fd_direct_on{-1};
  int _fd_direct_off{-1};
  bool _initialized{false};
  CompatMode _compat_mode{CompatMode::AUTO};
  mutable std::size_t _nbytes{0};  // The size of the underlying file, zero means unknown.
  CUfileHandle_t _handle{};

  /**
   * @brief Given a requested compatibility mode, whether it is expected to reduce to `ON` for
   * asynchronous I/O.
   *
   * @param requested_compat_mode Requested compatibility mode.
   * @return True if POSIX I/O fallback will be used; false for cuFile I/O.
   * @exception std::runtime_error When the requested compatibility mode is `OFF`, but cuFile
   * batch/stream library symbol is missing, or cuFile configuration file is missing.
   */
  bool is_compat_mode_preferred_for_async(CompatMode requested_compat_mode)
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

 public:
  static constexpr mode_t m644 = S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH;
  FileHandle() noexcept        = default;

  /**
   * @brief Construct a file handle from a file path
   *
   * FileHandle opens the file twice and maintains two file descriptors.
   * One file is opened with the specified `flags` and the other file is
   * opened with the `flags` plus the `O_DIRECT` flag.
   *
   * @param file_path File path to the file
   * @param flags Open flags (see also `fopen(3)`):
   *   "r" -> "open for reading (default)"
   *   "w" -> "open for writing, truncating the file first"
   *   "a" -> "open for writing, appending to the end of file if it exists"
   *   "+" -> "open for updating (reading and writing)"
   * @param mode Access modes (see `open(2)`).
   * @param compat_mode Set KvikIO's compatibility mode for this file.
   */
  FileHandle(const std::string& file_path,
             const std::string& flags = "r",
             mode_t mode              = m644,
             CompatMode compat_mode   = defaults::compat_mode());

  /**
   * @brief FileHandle support move semantic but isn't copyable
   */
  FileHandle(const FileHandle&)            = delete;
  FileHandle& operator=(FileHandle const&) = delete;
  FileHandle(FileHandle&& o) noexcept
    : _fd_direct_on{std::exchange(o._fd_direct_on, -1)},
      _fd_direct_off{std::exchange(o._fd_direct_off, -1)},
      _initialized{std::exchange(o._initialized, false)},
      _compat_mode{std::exchange(o._compat_mode, CompatMode::AUTO)},
      _nbytes{std::exchange(o._nbytes, 0)},
      _handle{std::exchange(o._handle, CUfileHandle_t{})}
  {
  }
  FileHandle& operator=(FileHandle&& o) noexcept
  {
    _fd_direct_on  = std::exchange(o._fd_direct_on, -1);
    _fd_direct_off = std::exchange(o._fd_direct_off, -1);
    _initialized   = std::exchange(o._initialized, false);
    _compat_mode   = std::exchange(o._compat_mode, CompatMode::AUTO);
    _nbytes        = std::exchange(o._nbytes, 0);
    _handle        = std::exchange(o._handle, CUfileHandle_t{});
    return *this;
  }
  ~FileHandle() noexcept { close(); }

  /**
   * @brief Whether the file is closed according to its initialization status.
   *
   * @return Boolean answer.
   */
  [[nodiscard]] bool closed() const noexcept { return !_initialized; }

  /**
   * @brief Deregister the file and close the two files
   */
  void close() noexcept
  {
    if (closed()) { return; }

    if (!is_compat_mode_preferred()) { cuFileAPI::instance().HandleDeregister(_handle); }
    _compat_mode = CompatMode::AUTO;
    ::close(_fd_direct_off);
    if (_fd_direct_on != -1) { ::close(_fd_direct_on); }
    _fd_direct_on  = -1;
    _fd_direct_off = -1;
    _initialized   = false;
  }

  /**
   * @brief Get the underlying cuFile file handle
   *
   * The file handle must be open and not in compatibility mode i.e.
   * both `closed()` and `is_compat_mode_preferred()` must be false.
   *
   * @return cuFile's file handle
   */
  [[nodiscard]] CUfileHandle_t handle()
  {
    if (closed()) { throw CUfileException("File handle is closed"); }
    if (is_compat_mode_preferred()) {
      throw CUfileException("The underlying cuFile handle isn't available in compatibility mode");
    }
    return _handle;
  }

  /**
   * @brief Get one of the file descriptors
   *
   * Notice, FileHandle maintains two file descriptors - one opened with the
   * `O_DIRECT` flag and one without. This function returns one of them but
   * it is unspecified which one.
   *
   * @return File descriptor
   */
  [[nodiscard]] int fd() const noexcept { return _fd_direct_off; }

  /**
   * @brief Get the flags of one of the file descriptors (see open(2))
   *
   * Notice, FileHandle maintains two file descriptors - one opened with the
   * `O_DIRECT` flag and one without. This function returns the flags of one of
   * them but it is unspecified which one.
   *
   * @return File descriptor
   */
  [[nodiscard]] int fd_open_flags() const;

  /**
   * @brief Get the file size
   *
   * The value are cached.
   *
   * @return The number of bytes
   */
  [[nodiscard]] std::size_t nbytes() const;

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
   * @param sync_default_stream Synchronize the CUDA default (null) stream prior to calling cuFile.
   * Contrary to most of the non-async CUDA API, cuFile does not have the semantic of being ordered
   * with respect to other non-cuFile work in the default stream. By enabling `sync_default_stream`,
   * KvikIO will synchronize the default stream and order the operation with respect to other work
   * in the null stream. When in KvikIO's compatibility mode or when accessing host memory, the
   * operation is always default stream ordered like the rest of the non-async CUDA API. In this
   * case, the value of `sync_default_stream` is ignored.
   * @return Size of bytes that were successfully read.
   */
  std::size_t read(void* devPtr_base,
                   std::size_t size,
                   std::size_t file_offset,
                   std::size_t devPtr_offset,
                   bool sync_default_stream = true)
  {
    if (is_compat_mode_preferred()) {
      return detail::posix_device_read(
        _fd_direct_off, devPtr_base, size, file_offset, devPtr_offset);
    }
    if (sync_default_stream) { CUDA_DRIVER_TRY(cudaAPI::instance().StreamSynchronize(nullptr)); }

    KVIKIO_NVTX_SCOPED_RANGE("cufileRead()", size);
    ssize_t ret = cuFileAPI::instance().Read(
      _handle, devPtr_base, size, convert_size2off(file_offset), convert_size2off(devPtr_offset));
    CUFILE_CHECK_BYTES_DONE(ret);
    return ret;
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
   * @param devPtr_offset Offset relative to the `devPtr_base` pointer to write from. This parameter
   * should be used only with registered buffers.
   * @param sync_default_stream Synchronize the CUDA default (null) stream prior to calling cuFile.
   * Contrary to most of the non-async CUDA API, cuFile does not have the semantic of being ordered
   * with respect to other non-cuFile work in the default stream. By enabling `sync_default_stream`,
   * KvikIO will synchronize the default stream and order the operation with respect to other work
   * in the null stream. When in KvikIO's compatibility mode or when accessing host memory, the
   * operation is always default stream ordered like the rest of the non-async CUDA API. In this
   * case, the value of `sync_default_stream` is ignored.
   * @return Size of bytes that were successfully written.
   */
  std::size_t write(const void* devPtr_base,
                    std::size_t size,
                    std::size_t file_offset,
                    std::size_t devPtr_offset,
                    bool sync_default_stream = true)
  {
    _nbytes = 0;  // Invalidate the computed file size

    if (is_compat_mode_preferred()) {
      return detail::posix_device_write(
        _fd_direct_off, devPtr_base, size, file_offset, devPtr_offset);
    }
    if (sync_default_stream) { CUDA_DRIVER_TRY(cudaAPI::instance().StreamSynchronize(nullptr)); }

    KVIKIO_NVTX_SCOPED_RANGE("cufileWrite()", size);
    ssize_t ret = cuFileAPI::instance().Write(
      _handle, devPtr_base, size, convert_size2off(file_offset), convert_size2off(devPtr_offset));
    if (ret == -1) {
      throw std::system_error(errno, std::generic_category(), "Unable to write file");
    }
    if (ret < -1) {
      throw CUfileException(std::string{"cuFile error at: "} + __FILE__ + ":" +
                            KVIKIO_STRINGIFY(__LINE__) + ": " + CUFILE_ERRSTR(ret));
    }
    return ret;
  }

  /**
   * @brief Reads specified bytes from the file into the device or host memory in parallel.
   *
   * This API is a parallel async version of `.read()` that partition the operation
   * into tasks of size `task_size` for execution in the default thread pool.
   *
   * In order to improve performance of small buffers, when `size < gds_threshold` a shortcut
   * that circumvent the threadpool and use the POSIX backend directly is used.
   *
   * @note For cuFile reads, the base address of the allocation `buf` is part of is used.
   * This means that when registering buffers, use the base address of the allocation.
   * This is what `memory_register` and `memory_deregister` do automatically.
   *
   * @param buf Address to device or host memory.
   * @param size Size in bytes to read.
   * @param file_offset Offset in the file to read from.
   * @param task_size Size of each task in bytes.
   * @param gds_threshold Minimum buffer size to use GDS and the thread pool.
   * @param sync_default_stream Synchronize the CUDA default (null) stream prior to calling cuFile.
   * Contrary to most of the non-async CUDA API, cuFile does not have the semantic of being ordered
   * with respect to other non-cuFile work in the default stream. By enabling `sync_default_stream`,
   * KvikIO will synchronize the default stream and order the operation with respect to other work
   * in the null stream. When in KvikIO's compatibility mode or when accessing host memory, the
   * operation is always default stream ordered like the rest of the non-async CUDA API. In this
   * case, the value of `sync_default_stream` is ignored.
   * @return Future that on completion returns the size of bytes that were successfully read.
   */
  std::future<std::size_t> pread(void* buf,
                                 std::size_t size,
                                 std::size_t file_offset   = 0,
                                 std::size_t task_size     = defaults::task_size(),
                                 std::size_t gds_threshold = defaults::gds_threshold(),
                                 bool sync_default_stream  = true)
  {
    KVIKIO_NVTX_MARKER("FileHandle::pread()", size);
    if (is_host_memory(buf)) {
      auto op = [this](void* hostPtr_base,
                       std::size_t size,
                       std::size_t file_offset,
                       std::size_t hostPtr_offset) -> std::size_t {
        char* buf = static_cast<char*>(hostPtr_base) + hostPtr_offset;
        return detail::posix_host_read<detail::PartialIO::NO>(
          _fd_direct_off, buf, size, file_offset);
      };

      return parallel_io(op, buf, size, file_offset, task_size, 0);
    }

    CUcontext ctx = get_context_from_pointer(buf);

    // Shortcut that circumvent the threadpool and use the POSIX backend directly.
    if (size < gds_threshold) {
      auto task = [this, ctx, buf, size, file_offset]() -> std::size_t {
        PushAndPopContext c(ctx);
        return detail::posix_device_read(_fd_direct_off, buf, size, file_offset, 0);
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

  /**
   * @brief Writes specified bytes from device or host memory into the file in parallel.
   *
   * This API is a parallel async version of `.write()` that partition the operation
   * into tasks of size `task_size` for execution in the default thread pool.
   *
   * In order to improve performance of small buffers, when `size < gds_threshold` a shortcut
   * that circumvent the threadpool and use the POSIX backend directly is used.
   *
   * @note For cuFile reads, the base address of the allocation `buf` is part of is used.
   * This means that when registering buffers, use the base address of the allocation.
   * This is what `memory_register` and `memory_deregister` do automatically.
   *
   * @param buf Address to device or host memory.
   * @param size Size in bytes to write.
   * @param file_offset Offset in the file to write from.
   * @param task_size Size of each task in bytes.
   * @param gds_threshold Minimum buffer size to use GDS and the thread pool.
   * @param sync_default_stream Synchronize the CUDA default (null) stream prior to calling cuFile.
   * Contrary to most of the non-async CUDA API, cuFile does not have the semantic of being ordered
   * with respect to other non-cuFile work in the default stream. By enabling `sync_default_stream`,
   * KvikIO will synchronize the default stream and order the operation with respect to other work
   * in the null stream. When in KvikIO's compatibility mode or when accessing host memory, the
   * operation is always default stream ordered like the rest of the non-async CUDA API. In this
   * case, the value of `sync_default_stream` is ignored.
   * @return Future that on completion returns the size of bytes that were successfully written.
   */
  std::future<std::size_t> pwrite(const void* buf,
                                  std::size_t size,
                                  std::size_t file_offset   = 0,
                                  std::size_t task_size     = defaults::task_size(),
                                  std::size_t gds_threshold = defaults::gds_threshold(),
                                  bool sync_default_stream  = true)
  {
    KVIKIO_NVTX_MARKER("FileHandle::pwrite()", size);
    if (is_host_memory(buf)) {
      auto op = [this](const void* hostPtr_base,
                       std::size_t size,
                       std::size_t file_offset,
                       std::size_t hostPtr_offset) -> std::size_t {
        const char* buf = static_cast<const char*>(hostPtr_base) + hostPtr_offset;
        return detail::posix_host_write<detail::PartialIO::NO>(
          _fd_direct_off, buf, size, file_offset);
      };

      return parallel_io(op, buf, size, file_offset, task_size, 0);
    }

    CUcontext ctx = get_context_from_pointer(buf);

    // Shortcut that circumvent the threadpool and use the POSIX backend directly.
    if (size < gds_threshold) {
      auto task = [this, ctx, buf, size, file_offset]() -> std::size_t {
        PushAndPopContext c(ctx);
        return detail::posix_device_write(_fd_direct_off, buf, size, file_offset, 0);
      };
      return std::async(std::launch::deferred, task);
    }

    // Let's synchronize once instead of in each task.
    if (sync_default_stream && !is_compat_mode_preferred()) {
      PushAndPopContext c(ctx);
      CUDA_DRIVER_TRY(cudaAPI::instance().StreamSynchronize(nullptr));
    }

    // Regular case that use the threadpool and run the tasks in parallel
    auto op = [this, ctx](const void* devPtr_base,
                          std::size_t size,
                          std::size_t file_offset,
                          std::size_t devPtr_offset) -> std::size_t {
      PushAndPopContext c(ctx);
      return write(
        devPtr_base, size, file_offset, devPtr_offset, /* sync_default_stream = */ false);
    };
    auto [devPtr_base, base_size, devPtr_offset] = get_alloc_info(buf, &ctx);
    return parallel_io(op, devPtr_base, size, file_offset, task_size, devPtr_offset);
  }

  /**
   * @brief Reads specified bytes from the file into the device memory asynchronously.
   *
   * This is an asynchronous version of `.read()`, which will be executed in sequence
   * for the specified stream.
   *
   * When running CUDA v12.1 or older, this function falls back to use `.read()` after
   * `stream` has been synchronized.
   *
   * The arguments have the same meaning as in `.read()` but some of them are deferred.
   * That is, the values pointed to by `size_p`, `file_offset_p` and `devPtr_offset_p`
   * will not be evaluated until execution time. Notice, this behavior can be changed
   * using cuFile's cuFileStreamRegister API.
   *
   * @param devPtr_base Base address of buffer in device memory. For registered buffers,
   * `devPtr_base` must remain set to the base address used in the `buffer_register` call.
   * @param size_p Pointer to size in bytes to read. If the exact size is not known at the time of
   * I/O submission, then you must set it to the maximum possible I/O size for that stream I/O.
   * Later the actual size can be set prior to the stream I/O execution.
   * @param file_offset_p Pointer to offset in the file from which to read. Unless otherwise set
   * using cuFileStreamRegister API, this value will not be evaluated until execution time.
   * @param devPtr_offset_p Pointer to the offset relative to the bufPtr_base from which to write.
   * Unless otherwise set using cuFileStreamRegister API, this value will not be evaluated until
   * execution time.
   * @param bytes_read_p Pointer to the bytes read from file. This pointer should be a non-NULL
   * value and *bytes_read_p set to 0. The bytes_read_p memory should be allocated with
   * cuMemHostAlloc/malloc/mmap or registered with cuMemHostRegister. After successful execution of
   * the operation in the stream, the value *bytes_read_p will contain either:
   *     - The number of bytes successfully read.
   *     - -1 on IO errors.
   *     - All other errors return a negative integer value of the CUfileOpError enum value.
   * @param stream CUDA stream in which to enqueue the operation. If NULL, make this operation
   * synchronous.
   */
  void read_async(void* devPtr_base,
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
      CUFILE_TRY(cuFileAPI::instance().ReadAsync(
        _handle, devPtr_base, size_p, file_offset_p, devPtr_offset_p, bytes_read_p, stream));
    }
  }

  /**
   * @brief Reads specified bytes from the file into the device memory asynchronously.
   *
   * This is an asynchronous version of `.read()`, which will be executed in sequence
   * for the specified stream.
   *
   * When running CUDA v12.1 or older, this function falls back to use `.read()` after
   * `stream` has been synchronized.
   *
   * The arguments have the same meaning as in `.read()` but returns a `StreamFuture` object
   * that the caller must keep alive until all data has been read from disk. One way to do this,
   * is by calling `StreamFuture.check_bytes_done()`, which will synchronize the associated stream
   * and return the number of bytes read.
   *
   * @param devPtr_base Base address of buffer in device memory. For registered buffers,
   * `devPtr_base` must remain set to the base address used in the `buffer_register` call.
   * @param size Size in bytes to read.
   * @param file_offset Offset in the file to read from.
   * @param devPtr_offset Offset relative to the `devPtr_base` pointer to read into. This parameter
   * should be used only with registered buffers.
   * @param stream CUDA stream in which to enqueue the operation. If NULL, make this operation
   * synchronous.
   * @return A future object that must be kept alive until all data has been read to disk e.g.
   * by synchronizing `stream`.
   */
  [[nodiscard]] StreamFuture read_async(void* devPtr_base,
                                        std::size_t size,
                                        off_t file_offset   = 0,
                                        off_t devPtr_offset = 0,
                                        CUstream stream     = nullptr)
  {
    StreamFuture ret(devPtr_base, size, file_offset, devPtr_offset, stream);
    auto [devPtr_base_, size_p, file_offset_p, devPtr_offset_p, bytes_read_p, stream_] =
      ret.get_args();
    read_async(devPtr_base_, size_p, file_offset_p, devPtr_offset_p, bytes_read_p, stream_);
    return ret;
  }

  /**
   * @brief Writes specified bytes from the device memory into the file asynchronously.
   *
   * This is an asynchronous version of `.write()`, which will be executed in sequence
   * for the specified stream.
   *
   * When running CUDA v12.1 or older, this function falls back to use `.read()` after
   * `stream` has been synchronized.
   *
   * The arguments have the same meaning as in `.write()` but some of them are deferred.
   * That is, the values pointed to by `size_p`, `file_offset_p` and `devPtr_offset_p`
   * will not be evaluated until execution time. Notice, this behavior can be changed
   * using cuFile's cuFileStreamRegister API.
   *
   * @param devPtr_base Base address of buffer in device memory. For registered buffers,
   * `devPtr_base` must remain set to the base address used in the `buffer_register` call.
   * @param size_p Pointer to size in bytes to read. If the exact size is not known at the time of
   * I/O submission, then you must set it to the maximum possible I/O size for that stream I/O.
   * Later the actual size can be set prior to the stream I/O execution.
   * @param file_offset_p Pointer to offset in the file from which to read. Unless otherwise set
   * using cuFileStreamRegister API, this value will not be evaluated until execution time.
   * @param devPtr_offset_p Pointer to the offset relative to the bufPtr_base from which to read.
   * Unless otherwise set using cuFileStreamRegister API, this value will not be evaluated until
   * execution time.
   * @param bytes_written_p Pointer to the bytes read from file. This pointer should be a non-NULL
   * value and *bytes_written_p set to 0. The bytes_written_p memory should be allocated with
   * cuMemHostAlloc/malloc/mmap or registered with cuMemHostRegister.
   * After successful execution of the operation in the stream, the value *bytes_written_p will
   * contain either:
   *     - The number of bytes successfully read.
   *     - -1 on IO errors.
   *     - All other errors return a negative integer value of the CUfileOpError enum value.
   * @param stream CUDA stream in which to enqueue the operation. If NULL, make this operation
   * synchronous.
   */
  void write_async(void* devPtr_base,
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
      CUFILE_TRY(cuFileAPI::instance().WriteAsync(
        _handle, devPtr_base, size_p, file_offset_p, devPtr_offset_p, bytes_written_p, stream));
    }
  }

  /**
   * @brief Writes specified bytes from the device memory into the file asynchronously.
   *
   * This is an asynchronous version of `.write()`, which will be executed in sequence
   * for the specified stream.
   *
   * When running CUDA v12.1 or older, this function falls back to use `.read()` after
   * `stream` has been synchronized.
   *
   * The arguments have the same meaning as in `.write()` but returns a `StreamFuture` object
   * that the caller must keep alive until all data has been written to disk. One way to do this,
   * is by calling `StreamFuture.check_bytes_done()`, which will synchronize the associated stream
   * and return the number of bytes written.
   *
   * @param devPtr_base Base address of buffer in device memory. For registered buffers,
   * `devPtr_base` must remain set to the base address used in the `buffer_register` call.
   * @param size Size in bytes to write.
   * @param file_offset Offset in the file to write from.
   * @param devPtr_offset Offset relative to the `devPtr_base` pointer to write from. This parameter
   * should be used only with registered buffers.
   * @param stream CUDA stream in which to enqueue the operation. If NULL, make this operation
   * synchronous.
   * @return A future object that must be kept alive until all data has been written to disk e.g.
   * by synchronizing `stream`.
   */
  [[nodiscard]] StreamFuture write_async(void* devPtr_base,
                                         std::size_t size,
                                         off_t file_offset   = 0,
                                         off_t devPtr_offset = 0,
                                         CUstream stream     = nullptr)
  {
    StreamFuture ret(devPtr_base, size, file_offset, devPtr_offset, stream);
    auto [devPtr_base_, size_p, file_offset_p, devPtr_offset_p, bytes_written_p, stream_] =
      ret.get_args();
    write_async(devPtr_base_, size_p, file_offset_p, devPtr_offset_p, bytes_written_p, stream_);
    return ret;
  }

  /**
   * @brief Returns `true` if the compatibility mode is expected to be `ON` for this file.
   *
   * Compatibility mode can be explicitly enabled in object creation. The mode is also enabled
   * automatically, if file cannot be opened with the `O_DIRECT` flag, or if the system does not
   * meet the requirements for the cuFile library under the `AUTO` compatibility mode.
   *
   * @return Boolean answer.
   */
  [[nodiscard]] bool is_compat_mode_preferred() const noexcept
  {
    return defaults::is_compat_mode_preferred(_compat_mode);
  }

  /**
   * @brief Returns `true` if the compatibility mode is expected to be `ON` for the asynchronous I/O
   * on this file.
   *
   * For asynchronous I/O, the compatibility mode can be automatically enabled if the cuFile batch
   * and stream symbols are missing, or if the cuFile configuration file is missing, or if
   * `is_compat_mode_preferred()` returns true.
   *
   * @return Boolean answer.
   */
  [[nodiscard]] bool is_compat_mode_preferred_for_async() const noexcept
  {
    static bool is_extra_symbol_available = is_stream_api_available();
    static bool is_config_path_empty      = config_path().empty();
    return is_compat_mode_preferred() || !is_extra_symbol_available || is_config_path_empty;
  }
};

}  // namespace kvikio
