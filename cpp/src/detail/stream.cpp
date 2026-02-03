
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <mutex>

#include <kvikio/detail/nvtx.hpp>
#include <kvikio/detail/stream.hpp>

namespace kvikio::detail {

CUstream StreamCachePerThreadAndContext::get()
{
  KVIKIO_NVTX_FUNC_RANGE();

  CUcontext ctx{nullptr};
  CUDA_DRIVER_TRY(cudaAPI::instance().CtxGetCurrent(&ctx));
  // If no current context, we return the null/default stream
  if (ctx == nullptr) { return nullptr; }

  static StreamCachePerThreadAndContext _instance;
  auto key = std::make_pair(ctx, std::this_thread::get_id());

  std::lock_guard const lock(_instance._mutex);

  // Create a new stream if the (context, thread) pair doesn't have one.
  if (auto search = _instance._streams.find(key); search == _instance._streams.end()) {
    CUstream stream{};
    CUDA_DRIVER_TRY(cudaAPI::instance().StreamCreate(&stream, CU_STREAM_DEFAULT));
    _instance._streams[key] = stream;
    return stream;
  } else {
    return search->second;
  }
}
}  // namespace kvikio::detail
