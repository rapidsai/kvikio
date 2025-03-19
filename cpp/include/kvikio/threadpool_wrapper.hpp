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

#include <kvikio/bounce_buffer.hpp>
#include <kvikio/error.hpp>
#include <kvikio/nvtx.hpp>
#include <kvikio/shim/cuda.hpp>
#include <kvikio/utils.hpp>

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
  thread_pool_wrapper(unsigned int nthreads,
                      std::size_t bounce_buffer_size,
                      std::size_t bounce_buffer_group_size)
    : pool_type(nthreads, preinitialize(nthreads, bounce_buffer_size, bounce_buffer_group_size))
  {
  }

  std::function<void()> preinitialize(unsigned int nthreads,
                                      std::size_t bounce_buffer_size,
                                      std::size_t bounce_buffer_group_size)
  {
    auto ctx = ensure_valid_current_context();
    BounceBuffer::preinitialize_for_pool(nthreads, bounce_buffer_size, bounce_buffer_group_size);

    auto worker_thread_init_func = [=] {
      CUDA_DRIVER_TRY(cudaAPI::instance().CtxPushCurrent(ctx));

      KVIKIO_NVTX_SCOPED_RANGE("worker thread init", 0, NvtxManager::default_color());
      // Rename the worker thread in the thread pool to improve clarity from nsys-ui.
      // Note: This NVTX feature is currently not supported by nsys-ui.
      NvtxManager::rename_current_thread("thread pool");

      BounceBuffer::instance().initialize_per_thread(bounce_buffer_size, bounce_buffer_group_size);
    };

    return worker_thread_init_func;
  }

  /**
   * @brief Reset the number of threads in the thread pool, and invoke a pre-defined initialization
   * function in each worker thread.
   *
   * @param nthreads The number of threads to use.
   */
  void reset(unsigned int nthreads,
             std::size_t bounce_buffer_size,
             std::size_t bounce_buffer_group_size)
  {
    // Block the calling thread until existing tasks in the thread pool are done.
    // This avoids race condition where data (such as BounceBuffer::block_pool) still being used by
    // the worker threads are modified in preinitialize().
    pool_type::wait();

    auto worker_thread_init_func =
      preinitialize(nthreads, bounce_buffer_size, bounce_buffer_group_size);

    pool_type::reset(nthreads, worker_thread_init_func);
  }
};

using BS_thread_pool = thread_pool_wrapper<BS::thread_pool>;

namespace this_thread {
template <typename T>
bool is_from_pool();

template <>
bool is_from_pool<BS_thread_pool>();

template <typename T>
std::optional<std::size_t> index();

template <>
std::optional<std::size_t> index<BS_thread_pool>();

}  // namespace this_thread

}  // namespace kvikio
