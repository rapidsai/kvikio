/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <sys/types.h>
#include <algorithm>
#include <cstdlib>
#include <kvikio/error.hpp>
#include <kvikio/shim/cuda.hpp>
#include <kvikio/shim/cufile.hpp>
#include <tuple>

namespace kvikio {

/**
 * @brief Future of an asynchronous IO operation
 *
 * This class shouldn't be used directly, instead some stream operations such as
 * `FileHandle.read_async` and `FileHandle.write_async` returns an instance of this class. Use
 * `.check_bytes_done()` to synchronize the associated CUDA stream and return the number of bytes
 * read or written by the operation.
 *
 * The goal of this class is twofold:
 *   - Have `read_async` and `write_async` return an object that clearly associates the function
 *     arguments with the CUDA stream used. This is useful because the current validity of the
 *     arguments depends on the stream.
 *   - Support of by-value arguments. In many cases, a user will use `read_async` and `write_async`
 *     like most other asynchronous CUDA functions that take by-value arguments.
 *
 * To support by-value arguments, we allocate the arguments on the heap (malloc `ArgByVal`) and have
 * the by-reference arguments (`ArgByRef`) points into `ArgByVal`. This way, the `read_async` and
 * `write_async` can call `.get_args()` to get the by-reference arguments required by cuFile's
 * stream API.
 */
class StreamFuture {
 public:
  struct ArgByVal {
    std::size_t size;
    off_t file_offset;
    off_t devPtr_offset;
    ssize_t bytes_done;
  };
  struct ArgByRef {
    std::size_t* size_p;
    off_t* file_offset_p;
    off_t* devPtr_offset_p;
    ssize_t* bytes_done_p;
  };

 private:
  void* _devPtr_base{nullptr};
  CUstream _stream{nullptr};
  ArgByVal* _val{nullptr};
  ArgByRef _ref{nullptr};

 public:
  StreamFuture() noexcept = default;

  StreamFuture(
    void* devPtr_base, std::size_t size, off_t file_offset, off_t devPtr_offset, CUstream stream)
    : _devPtr_base{devPtr_base}, _stream{stream}
  {
    // Notice, we allocate the arguments using malloc() as specified in the cuFile docs:
    // <https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#cufilewriteasync>
    if ((_val = static_cast<ArgByVal*>(std::malloc(sizeof(ArgByVal)))) == nullptr) {
      throw std::bad_alloc{};
    }
    *_val = {
      .size = size, .file_offset = file_offset, .devPtr_offset = devPtr_offset, .bytes_done = 0};
    _ref = {.size_p          = &_val->size,
            .file_offset_p   = &_val->file_offset,
            .devPtr_offset_p = &_val->devPtr_offset,
            .bytes_done_p    = &_val->bytes_done};
  }

  /**
   * @brief StreamFuture support move semantic but isn't copyable
   */
  StreamFuture(const StreamFuture&)        = delete;
  StreamFuture& operator=(StreamFuture& o) = delete;
  StreamFuture(StreamFuture&& o) noexcept
    : _devPtr_base{std::exchange(o._devPtr_base, nullptr)},
      _stream{std::exchange(o._stream, nullptr)},
      _val{std::exchange(o._val, nullptr)},
      _ref{o._ref}
  {
  }
  StreamFuture& operator=(StreamFuture&& o) noexcept
  {
    _devPtr_base = std::exchange(o._devPtr_base, nullptr);
    _stream      = std::exchange(o._stream, nullptr);
    _val         = std::exchange(o._val, nullptr);
    _ref         = o._ref;
    return *this;
  }

  ~StreamFuture() noexcept
  {
    if (_val != nullptr) { free(_val); }
  }

  /**
   * @brief Return the arguments of the future call
   *
   * @return Tuple of the arguments in the order matching `FileHandle.read()` and
   * `FileHandle.write()`
   */
  std::tuple<void*, std::size_t*, off_t*, off_t*, ssize_t*, CUstream> get_args() const noexcept
  {
    return std::make_tuple(_devPtr_base,
                           _ref.size_p,
                           _ref.file_offset_p,
                           _ref.devPtr_offset_p,
                           _ref.bytes_done_p,
                           _stream);
  }

  /**
   * @brief Return the number of bytes read or written by the future operation.
   *
   * Synchronize the associated CUDA stream.
   *
   * @return Number of bytes read or written by the future operation.
   */
  std::size_t check_bytes_done() const
  {
    CUDA_DRIVER_TRY(cudaAPI::instance().StreamSynchronize(_stream));
    CUFILE_CHECK_STREAM_IO(_ref.bytes_done_p);
    // At this point, we know `*_bytes_done_p` is positive otherwise
    // CUFILE_CHECK_STREAM_IO() would have raised an exception.
    return static_cast<std::size_t>(*_ref.bytes_done_p);
  }
};

}  // namespace kvikio
