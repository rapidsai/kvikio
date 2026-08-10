
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <mutex>
#include <utility>

#include <kvikio/detail/nvtx.hpp>
#include <kvikio/detail/stream.hpp>
#include <kvikio/error.hpp>

namespace kvikio::detail {

CUstream StreamCachePerThreadAndContext::get()
{
  KVIKIO_NVTX_FUNC_RANGE();

  CUcontext ctx{nullptr};
  KVIKIO_CUDA_DRIVER_TRY(cudaAPI::instance().CtxGetCurrent(&ctx));
  // If no current context, we return the null/default stream
  if (ctx == nullptr) { return nullptr; }

  static StreamCachePerThreadAndContext _instance;
  auto key = std::make_pair(ctx, std::this_thread::get_id());

  {
    std::lock_guard const lock(_instance._mutex);
    if (auto it = _instance._streams.find(key); it != _instance._streams.end()) {
      return it->second;
    }
  }

  // Create a new stream if the (context, thread) pair doesn't have one.
  CUstream stream{};
  KVIKIO_CUDA_DRIVER_TRY(cudaAPI::instance().StreamCreate(&stream, CU_STREAM_DEFAULT));

  {
    std::lock_guard const lock(_instance._mutex);
    auto const [it, inserted] = _instance._streams.emplace(key, stream);
    KVIKIO_EXPECT(inserted, "New stream insertion failed unexpectedly.");
  }
  return stream;
}
}  // namespace kvikio::detail
