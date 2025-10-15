/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

std::size_t posix_device_read(int fd,
                              void const* devPtr_base,
                              std::size_t size,
                              std::size_t file_offset,
                              std::size_t devPtr_offset)
{
  KVIKIO_NVTX_FUNC_RANGE(size);
  return detail::posix_device_io<IOOperationType::READ>(
    fd, devPtr_base, size, file_offset, devPtr_offset);
}

std::size_t posix_device_write(int fd,
                               void const* devPtr_base,
                               std::size_t size,
                               std::size_t file_offset,
                               std::size_t devPtr_offset)
{
  KVIKIO_NVTX_FUNC_RANGE(size);
  return detail::posix_device_io<IOOperationType::WRITE>(
    fd, devPtr_base, size, file_offset, devPtr_offset);
}

}  // namespace kvikio::detail
