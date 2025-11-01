/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <unistd.h>
#include <cstddef>
#include <cstdlib>
#include <map>
#include <thread>

#include <kvikio/bounce_buffer.hpp>
#include <kvikio/detail/nvtx.hpp>
#include <kvikio/detail/posix_io.hpp>
#include <kvikio/error.hpp>
#include <kvikio/shim/cuda.hpp>
#include <kvikio/utils.hpp>

namespace kvikio::detail {

CUstream StreamsByThread::get(CUcontext ctx, std::thread::id thd_id)
{
  KVIKIO_NVTX_FUNC_RANGE();
  static StreamsByThread _instance;

  // If no current context, we return the null/default stream
  if (ctx == nullptr) { return nullptr; }
  auto key = std::make_pair(ctx, thd_id);

  // Create a new stream if `ctx` doesn't have one.
  if (auto search = _instance._streams.find(key); search == _instance._streams.end()) {
    CUstream stream{};
    CUDA_DRIVER_TRY(cudaAPI::instance().StreamCreate(&stream, CU_STREAM_DEFAULT));
    _instance._streams[key] = stream;
    return stream;
  } else {
    return search->second;
  }
}

CUstream StreamsByThread::get()
{
  KVIKIO_NVTX_FUNC_RANGE();
  CUcontext ctx{nullptr};
  CUDA_DRIVER_TRY(cudaAPI::instance().CtxGetCurrent(&ctx));
  return get(ctx, std::this_thread::get_id());
}

std::size_t posix_device_read(int fd_direct_off,
                              void const* devPtr_base,
                              std::size_t size,
                              std::size_t file_offset,
                              std::size_t devPtr_offset,
                              std::optional<int> fd_direct_on)
{
  KVIKIO_NVTX_FUNC_RANGE(size);
  if (defaults::posix_direct_io_enabled()) {
    return detail::posix_device_io<IOOperationType::READ, CudaPageAlignedPinnedBounceBufferPool>(
      fd_direct_off, devPtr_base, size, file_offset, devPtr_offset, fd_direct_on);
  } else {
    return detail::posix_device_io<IOOperationType::READ>(
      fd_direct_off, devPtr_base, size, file_offset, devPtr_offset, fd_direct_on);
  }
}

std::size_t posix_device_write(int fd_direct_off,
                               void const* devPtr_base,
                               std::size_t size,
                               std::size_t file_offset,
                               std::size_t devPtr_offset,
                               std::optional<int> fd_direct_on)
{
  KVIKIO_NVTX_FUNC_RANGE(size);
  if (defaults::posix_direct_io_enabled()) {
    return detail::posix_device_io<IOOperationType::WRITE, CudaPageAlignedPinnedBounceBufferPool>(
      fd_direct_off, devPtr_base, size, file_offset, devPtr_offset, fd_direct_on);
  } else {
    return detail::posix_device_io<IOOperationType::WRITE>(
      fd_direct_off, devPtr_base, size, file_offset, devPtr_offset, fd_direct_on);
  }
}

}  // namespace kvikio::detail
