/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <map>
#include <thread>

#include <kvikio/shim/cuda.hpp>

namespace kvikio::detail {
/**
 * @brief Singleton class to retrieve a CUDA stream for device-host copying
 *
 * Call `StreamsByThread::get` to get the CUDA stream assigned to the current
 * CUDA context and thread.
 */
class StreamsByThread {
 private:
  std::map<std::pair<CUcontext, std::thread::id>, CUstream> _streams;

 public:
  StreamsByThread() = default;

  // Here we intentionally do not destroy in the destructor the CUDA resources
  // (e.g. CUstream) with static storage duration, but instead let them leak
  // on program termination. This is to prevent undefined behavior in CUDA. See
  // <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#initialization>
  // This also prevents crash (segmentation fault) if clients call
  // cuDevicePrimaryCtxReset() or cudaDeviceReset() before program termination.
  ~StreamsByThread() = default;

  KVIKIO_EXPORT static CUstream get(CUcontext ctx, std::thread::id thd_id);

  static CUstream get();

  StreamsByThread(StreamsByThread const&)            = delete;
  StreamsByThread& operator=(StreamsByThread const&) = delete;
  StreamsByThread(StreamsByThread&& o)               = delete;
  StreamsByThread& operator=(StreamsByThread&& o)    = delete;
};
}  // namespace kvikio::detail
