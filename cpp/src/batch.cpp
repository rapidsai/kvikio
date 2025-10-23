/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstddef>
#include <ctime>
#include <utility>
#include <vector>

#include <kvikio/batch.hpp>
#include <kvikio/error.hpp>
#include <kvikio/file_handle.hpp>
#include <kvikio/shim/cufile.hpp>

namespace kvikio {

#ifdef KVIKIO_CUFILE_BATCH_API_FOUND

BatchHandle::BatchHandle(int max_num_events) : _initialized{true}, _max_num_events{max_num_events}
{
  CUFILE_TRY(cuFileAPI::instance().BatchIOSetUp(&_handle, max_num_events));
}

BatchHandle::BatchHandle(BatchHandle&& o) noexcept
  : _initialized{std::exchange(o._initialized, false)},
    _max_num_events{std::exchange(o._max_num_events, 0)}
{
  _handle = std::exchange(o._handle, CUfileBatchHandle_t{});
}

BatchHandle::~BatchHandle() noexcept { close(); }

bool BatchHandle::closed() const noexcept { return !_initialized; }

void BatchHandle::close() noexcept
{
  if (closed()) { return; }
  _initialized = false;

  cuFileAPI::instance().BatchIODestroy(_handle);
}

void BatchHandle::submit(std::vector<BatchOp> const& operations)
{
  KVIKIO_EXPECT(convert_size2ssize(operations.size()) <= _max_num_events,
                "Cannot submit more than the max_num_events)");
  std::vector<CUfileIOParams_t> io_batch_params;
  io_batch_params.reserve(operations.size());
  for (auto const& op : operations) {
    KVIKIO_EXPECT(!op.file_handle.get_compat_mode_manager().is_compat_mode_preferred(),
                  "Cannot submit a FileHandle opened in compatibility mode");

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

std::vector<CUfileIOEvents_t> BatchHandle::status(unsigned min_nr,
                                                  unsigned max_nr,
                                                  struct timespec* timeout)
{
  std::vector<CUfileIOEvents_t> ret;
  ret.resize(_max_num_events);
  CUFILE_TRY(cuFileAPI::instance().BatchIOGetStatus(_handle, min_nr, &max_nr, &ret[0], timeout));
  ret.resize(max_nr);
  return ret;
}

void BatchHandle::cancel() { CUFILE_TRY(cuFileAPI::instance().BatchIOCancel(_handle)); }

#else

BatchHandle::BatchHandle(int max_num_events)
{
  KVIKIO_FAIL("BatchHandle requires cuFile's batch API, please build with CUDA v12.1+");
}

bool BatchHandle::closed() const noexcept { return true; }

void BatchHandle::close() noexcept {}

void BatchHandle::submit(std::vector<BatchOp> const& operations) {}

std::vector<CUfileIOEvents_t> BatchHandle::status(unsigned min_nr,
                                                  unsigned max_nr,
                                                  struct timespec* timeout)
{
  return std::vector<CUfileIOEvents_t>{};
}

void BatchHandle::cancel() {}

#endif

}  // namespace kvikio
