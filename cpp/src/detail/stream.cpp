
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <mutex>

#include <kvikio/detail/nvtx.hpp>
#include <kvikio/detail/stream.hpp>

namespace kvikio::detail {

CUstream StreamCachePerThreadAndContext::get(CUcontext ctx, std::thread::id thd_id)
{
  static StreamCachePerThreadAndContext _instance;

  // If no current context, we return the null/default stream
  if (ctx == nullptr) { return nullptr; }

  std::lock_guard const lock(_instance._mutex);
  auto key = std::make_pair(ctx, thd_id);

  // Create a new stream if this (context, thread) pair doesn't have one.
  if (auto search = _instance._streams.find(key); search == _instance._streams.end()) {
    CUstream stream{};
    CUDA_DRIVER_TRY(cudaAPI::instance().StreamCreate(&stream, CU_STREAM_DEFAULT));
    _instance._streams[key] = stream;
    return stream;
  } else {
    return search->second;
  }
}

CUstream StreamCachePerThreadAndContext::get()
{
  KVIKIO_NVTX_FUNC_RANGE();
  CUcontext ctx{nullptr};
  CUDA_DRIVER_TRY(cudaAPI::instance().CtxGetCurrent(&ctx));
  return get(ctx, std::this_thread::get_id());
}
}  // namespace kvikio::detail
