/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <map>
#include <mutex>
#include <thread>

#include <kvikio/shim/cuda.hpp>

namespace kvikio::detail {
/**
 * @brief Singleton cache that provides one CUDA stream per (context, thread) pair.
 *
 * This class manages CUDA streams used for host-device memory transfers. Each unique combination of
 * CUDA context and calling thread is assigned a dedicated stream, which is created lazily on first
 * access and reused for subsequent calls.
 *
 * The cache is thread-safe and handles concurrent access from multiple threads.
 *
 * @note CUDA streams are intentionally leaked on program termination rather than destroyed in the
 * destructor. This avoids undefined behavior that can occur when destroying CUDA resources during
 * static destruction, and prevents crashes (segmentation faults) if clients call
 * cuDevicePrimaryCtxReset() or cudaDeviceReset() before program termination. See:
 * https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#initialization
 */
class StreamCachePerThreadAndContext {
 private:
  std::map<std::pair<CUcontext, std::thread::id>, CUstream> _streams;
  std::mutex mutable _mutex;

 private:
  StreamCachePerThreadAndContext()  = default;
  ~StreamCachePerThreadAndContext() = default;

  /**
   * @brief Get or create a CUDA stream for the specified context and thread.
   *
   * If a stream already exists for the given (context, thread) pair, it is returned. Otherwise, a
   * new stream is created, cached, and returned.
   *
   * @param ctx The CUDA context. If null, the null stream is returned.
   * @param thd_id The thread identifier.
   * @return The CUDA stream associated with this (context, thread) pair, or nullptr if @p ctx is
   * null.
   */
  KVIKIO_EXPORT static CUstream get(CUcontext ctx, std::thread::id thd_id);

 public:
  /**
   * @brief Get or create a CUDA stream for the current context and thread.
   *
   * Convenience overload that uses the current CUDA context and calling thread's ID.
   *
   * @return The CUDA stream associated with the current (context, thread) pair, or nullptr if no
   * CUDA context is current.
   */
  static CUstream get();

  StreamCachePerThreadAndContext(StreamCachePerThreadAndContext const&)            = delete;
  StreamCachePerThreadAndContext& operator=(StreamCachePerThreadAndContext const&) = delete;
  StreamCachePerThreadAndContext(StreamCachePerThreadAndContext&& o)               = delete;
  StreamCachePerThreadAndContext& operator=(StreamCachePerThreadAndContext&& o)    = delete;
};
}  // namespace kvikio::detail
