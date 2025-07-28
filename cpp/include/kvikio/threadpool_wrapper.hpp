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

#pragma once

#include <functional>

#include <BS_thread_pool.hpp>

#include <kvikio/nvtx.hpp>

namespace kvikio {

template <typename pool_type>
class thread_pool_wrapper : public pool_type {
 public:
  /**
   * @brief Construct a new thread pool wrapper, and invoke a pre-defined initialization function in
   * each worker thread.
   *
   * @param nthreads The number of threads to use.
   */
  thread_pool_wrapper(unsigned int nthreads) : pool_type{nthreads, worker_thread_init_func}
  {
    KVIKIO_NVTX_FUNC_RANGE();
  }

  /**
   * @brief Reset the number of threads in the thread pool, and invoke a pre-defined initialization
   * function in each worker thread.
   *
   * @param nthreads The number of threads to use.
   */
  void reset(unsigned int nthreads)
  {
    KVIKIO_NVTX_FUNC_RANGE();
    pool_type::reset(nthreads, worker_thread_init_func);
  }

 private:
  inline static std::function<void()> worker_thread_init_func{[] {
    KVIKIO_NVTX_FUNC_RANGE();
    // Rename the worker thread in the thread pool to improve clarity from nsys-ui.
    // Note: This NVTX feature is currently not supported by nsys-ui.
    NvtxManager::rename_current_thread("thread pool");
  }};
};

using BS_thread_pool = thread_pool_wrapper<BS::thread_pool>;

}  // namespace kvikio
