
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <kvikio/detail/nvtx.hpp>
#include <kvikio/detail/stream.hpp>

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
}  // namespace kvikio::detail
