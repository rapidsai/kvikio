/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.
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
#include <cstdlib>
#include <tuple>
#include <utility>

#include <kvikio/shim/cuda.hpp>

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
 * the by-reference arguments points into `ArgByVal`. This way, the `read_async` and `write_async`
 * can call `.get_args()` to get the by-reference arguments required by cuFile's stream API.
 */
class StreamFuture {
 private:
  struct ArgByVal {
    std::size_t size;
    off_t file_offset;
    off_t devPtr_offset;
    ssize_t bytes_done;
  };

  void* _devPtr_base{nullptr};
  CUstream _stream{nullptr};
  ArgByVal* _val{nullptr};
  bool _stream_synchronized{false};

 public:
  StreamFuture() noexcept = default;

  StreamFuture(
    void* devPtr_base, std::size_t size, off_t file_offset, off_t devPtr_offset, CUstream stream);

  /**
   * @brief StreamFuture support move semantic but isn't copyable
   */
  StreamFuture(StreamFuture const&)        = delete;
  StreamFuture& operator=(StreamFuture& o) = delete;
  StreamFuture(StreamFuture&& o) noexcept;
  StreamFuture& operator=(StreamFuture&& o) noexcept;

  /**
   * @brief Return the arguments of the future call
   *
   * @return Tuple of the arguments in the order matching `FileHandle.read()` and
   * `FileHandle.write()`
   */
  std::tuple<void*, std::size_t*, off_t*, off_t*, ssize_t*, CUstream> get_args() const;

  /**
   * @brief Return the number of bytes read or written by the future operation.
   *
   * Synchronize the associated CUDA stream.
   *
   * @return Number of bytes read or written by the future operation.
   */
  std::size_t check_bytes_done();

  /**
   * @brief Free the by-value arguments and make sure the associated CUDA stream has been
   * synchronized.
   */
  ~StreamFuture() noexcept;
};

}  // namespace kvikio
