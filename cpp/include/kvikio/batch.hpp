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

struct BatchOp {
  FileHandle& file_handle;
  void* devPtr_base;
  off_t file_offset;
  off_t devPtr_offset;
  size_t size;
  CUfileOpcode_t opcode;
};

#ifdef CUFILE_BATCH_API_FOUND

/**
 * @brief Handle of an cuFile batch.
 *
 * In order to utilize cufile and GDS, a file must be registered with cufile.
 */
class BatchHandle {
 private:
  bool _initialized{false};
  int _max_num_events{};
  CUfileBatchHandle_t _handle{};

 public:
  BatchHandle() noexcept = default;

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

  void close() noexcept
  {
    if (closed()) { return; }
    _initialized = false;

    cuFileAPI::instance().BatchIODestroy(_handle);
  }

  void submit(const std::vector<BatchOp>& operations)
  {
    std::vector<CUfileIOParams_t> io_batch_params;
    io_batch_params.reserve(operations.size());
    for (const auto& op : operations) {
      io_batch_params.push_back(CUfileIOParams_t{.mode   = CUFILE_BATCH,
                                                 .u      = {.batch = {.devPtr_base   = op.devPtr_base,
                                                                      .file_offset   = op.file_offset,
                                                                      .devPtr_offset = op.devPtr_offset,
                                                                      .size          = op.size}},
                                                 .fh     = op.file_handle.handle(),
                                                 .opcode = op.opcode,
                                                 .cookie = nullptr});
    }

    CUFILE_TRY(
      cuFileAPI::instance().BatchIOSubmit(_handle, io_batch_params.size(), &io_batch_params[0], 0));
  }

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
};

#endif

}  // namespace kvikio
