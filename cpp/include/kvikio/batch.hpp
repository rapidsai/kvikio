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

#include <cstddef>
#include <ctime>
#include <utility>
#include <vector>

#include <kvikio/error.hpp>
#include <kvikio/file_handle.hpp>
#include <kvikio/shim/cufile.hpp>

namespace kvikio {

/**
 * @brief IO operation used when submitting batches
 */
struct BatchOp {
  // The file handle of the file to read or write
  FileHandle& file_handle;
  // Base address of buffer in device memory (host memory not supported).
  void* devPtr_base;
  // Offset in the file to read from or write to.
  off_t file_offset;
  // Offset relative to the `devPtr_base` pointer to write into or read from.
  off_t devPtr_offset;
  // Size in bytes to read or write.
  size_t size;
  // The operation type: CUFILE_READ or CUFILE_WRITE.
  CUfileOpcode_t opcode;
};

#ifdef KVIKIO_CUFILE_BATCH_API_FOUND

/**
 * @brief Handle of an cuFile batch using  semantic.
 *
 * The workflow is as follows:
 *  1) Create a batch with a large enough `max_num_events`.
 *  2) Call `.submit()` with a vector of operations (`vector.size() <= max_num_events`).
 *  3) Call `.status()` to wait on the operations to finish, or
 *  3) Call `.cancel()` to cancel the operations.
 *  4) Go to step 2 or call `.close()` to free up resources.
 *
 * Notice, a batch handle can only handle one "submit" at a time and is closed
 * in the destructor automatically.
 */
class BatchHandle {
 private:
  bool _initialized{false};
  int _max_num_events{};
  CUfileBatchHandle_t _handle{};

 public:
  BatchHandle() noexcept = default;

  /**
   * @brief Construct a batch handle
   *
   * @param max_num_events The maximum number of operations supported by this instance.
   */
  BatchHandle(int max_num_events) : _initialized{true}, _max_num_events{max_num_events}
  {
    CUFILE_TRY(cuFileAPI::instance().BatchIOSetUp(&_handle, max_num_events));
  }

  /**
   * @brief BatchHandle support move semantic but isn't copyable
   */
  BatchHandle(const BatchHandle&)            = delete;
  BatchHandle& operator=(BatchHandle const&) = delete;
  BatchHandle(BatchHandle&& o) noexcept
    : _initialized{std::exchange(o._initialized, false)},
      _max_num_events{std::exchange(o._max_num_events, 0)}
  {
    _handle = std::exchange(o._handle, CUfileBatchHandle_t{});
  }
  ~BatchHandle() noexcept { close(); }

  [[nodiscard]] bool closed() const noexcept { return !_initialized; }

  /**
   * @brief Destroy the batch handle and free up resources
   */
  void close() noexcept
  {
    if (closed()) { return; }
    _initialized = false;

    cuFileAPI::instance().BatchIODestroy(_handle);
  }

  /**
   * @brief Submit a vector of batch operations
   *
   * @param operations The vector of batch operations, which must not exceed the
   * `max_num_events`.
   */
  void submit(const std::vector<BatchOp>& operations)
  {
    if (convert_size2ssize(operations.size()) > _max_num_events) {
      throw CUfileException("Cannot submit more than the max_num_events)");
    }
    std::vector<CUfileIOParams_t> io_batch_params;
    io_batch_params.reserve(operations.size());
    for (const auto& op : operations) {
      if (op.file_handle.is_compat_mode_on()) {
        throw CUfileException("Cannot submit a FileHandle opened in compatibility mode");
      }

      io_batch_params.push_back(CUfileIOParams_t{.mode   = CUFILE_BATCH,
                                                 .u      = {.batch = {.devPtr_base   = op.devPtr_base,
                                                                      .file_offset   = op.file_offset,
                                                                      .devPtr_offset = op.devPtr_offset,
                                                                      .size          = op.size}},
                                                 .fh     = op.file_handle.handle(),
                                                 .opcode = op.opcode,
                                                 .cookie = nullptr});
    }

    CUFILE_TRY(cuFileAPI::instance().BatchIOSubmit(
      _handle, io_batch_params.size(), io_batch_params.data(), 0));
  }

  /**
   * @brief Get status of submitted operations
   *
   * @param min_nr The minimum number of IO entries for which status is requested.
   * @param max_nr The maximum number of IO requests to poll for.
   * @param timeout This parameter is used to specify the amount of time to wait for
   * in this API, even if the minimum number of requests have not completed. If the
   * timeout hits, it is possible that the number of returned IOs can be less than `min_nr`
   * @return Vector of the status of the completed I/Os in the batch.
   */
  std::vector<CUfileIOEvents_t> status(unsigned min_nr,
                                       unsigned max_nr,
                                       struct timespec* timeout = nullptr)
  {
    std::vector<CUfileIOEvents_t> ret;
    ret.resize(_max_num_events);
    CUFILE_TRY(cuFileAPI::instance().BatchIOGetStatus(_handle, min_nr, &max_nr, &ret[0], timeout));
    ret.resize(max_nr);
    return ret;
  }

  void cancel() { CUFILE_TRY(cuFileAPI::instance().BatchIOCancel(_handle)); }
};

#else

class BatchHandle {
 public:
  BatchHandle() noexcept = default;

  BatchHandle(int max_num_events)
  {
    throw CUfileException("BatchHandle requires cuFile's batch API, please build with CUDA v12.1+");
  }

  [[nodiscard]] bool closed() const noexcept { return true; }

  void close() noexcept {}

  void submit(const std::vector<BatchOp>& operations) {}

  std::vector<CUfileIOEvents_t> status(unsigned min_nr,
                                       unsigned max_nr,
                                       struct timespec* timeout = nullptr)
  {
    return std::vector<CUfileIOEvents_t>{};
  }
  void cancel() {}
};

#endif

}  // namespace kvikio
