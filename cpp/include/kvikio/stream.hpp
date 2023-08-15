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
#include <kvikio/file_handle.hpp>
#include <kvikio/parallel_operation.hpp>
#include <kvikio/posix_io.hpp>
#include <kvikio/shim/cuda.hpp>
#include <kvikio/shim/cufile.hpp>
#include <kvikio/utils.hpp>

namespace kvikio {

/**
 * @brief Handle of an open file registered with cufile.
 *
 * In order to utilize cufile and GDS, a file must be registered with cufile.
 */
class StreamHandle : public FileHandle {
 private:
  CUstream _stream;
  unsigned _stream_flags;
  bool _registered;

 public:
  StreamHandle() noexcept = default;

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
               CUstream stream,
               const std::string& flags = "r",
               mode_t mode              = m644,
               bool compat_mode         = defaults::compat_mode(),
               unsigned stream_flags    = 0xf)
    : FileHandle(file_path, flags, mode, compat_mode)
  {
    _stream       = stream;
    _stream_flags = stream_flags;
    _registered   = false;
  }

  /**
   * @brief FileHandle support move semantic but isn't copyable
   */
  StreamHandle(const StreamHandle&)            = delete;
  StreamHandle& operator=(StreamHandle const&) = delete;
  StreamHandle(StreamHandle&& o) noexcept
    : _stream{std::exchange(o._stream, CUstream{})},
      _stream_flags{std::exchange(o._stream_flags, 0xf)},
      _registered{std::exchange(o._registered, false)}
  {
  }
  StreamHandle& operator=(StreamHandle&& o) noexcept
  {
    _stream_flags = std::exchange(o._stream_flags, 0xf);
    _stream       = std::exchange(o._stream, CUstream{});
    _registered   = std::exchange(o._registered, false);
    return *this;
  }
  ~StreamHandle() noexcept { close(); }

  /**
   * @brief Deregister the file and close the stream.
   */
  void close()
  {
    if (closed()) { return; }

    if (_stream != nullptr) {
      if (_registered) {
        CUFILE_TRY(cuFileAPI::instance().StreamDeregister(_stream));
        _registered = false;
      }
      CUDA_DRIVER_TRY(cudaAPI::instance().StreamDestroy(_stream));
      _stream = nullptr;
    }

    _stream_flags = 0;
  }

  /**
   * @brief Get the stream associated with the handle.
   *
   * @return CUstream
   */
  [[nodiscard]] CUstream get_stream() const noexcept { return _stream; }

  void stream_register(unsigned stream_flags)
  {
    if (_stream != nullptr && !_registered) {
      CUFILE_TRY(cuFileAPI::instance().StreamRegister(_stream, stream_flags));
      _registered   = true;
      _stream_flags = stream_flags;
    }
  }

  void stream_deregister(CUstream stream)
  {
    if (stream != nullptr && _registered) {
      CUFILE_TRY(cuFileAPI::instance().StreamDeregister(stream));
      _registered   = false;
      _stream_flags = 0;
    }
  }
  void stream_synchronize() { CUDA_DRIVER_TRY(cudaAPI::instance().StreamSynchronize(_stream)); }
};
}  // namespace kvikio
