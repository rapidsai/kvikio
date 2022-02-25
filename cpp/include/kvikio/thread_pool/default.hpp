/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <cstdlib>
#include <stdexcept>
#include <string>

#include "thread_pool.hpp"

namespace kvikio::default_thread_pool {  // TODO: should this be a singletone class instead?
namespace {
inline unsigned int get_num_threads_from_env()
{
  const char* nthreads = std::getenv("KVIKIO_NTHREADS");
  if (nthreads == nullptr) { return 1; }
  const int n = std::stoi(nthreads);
  if (n <= 0) { throw std::invalid_argument("KVIKIO_NTHREADS has to be a positive integer"); }
  return std::stoi(nthreads);
}
/*NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)*/
inline kvikio::third_party::thread_pool _current_default_thread_pool{get_num_threads_from_env()};

}  // namespace

/**
 * @brief Get the default thread pool.
 *
 * @return The the default thread pool instance.
 */
inline kvikio::third_party::thread_pool& get() { return _current_default_thread_pool; }

/**
 * @brief Reset the number of threads in the default thread pool. Waits for all currently running
 * tasks to be completed, then destroys all threads in the pool and creates a new thread pool with
 * the new number of threads. Any tasks that were waiting in the queue before the pool was reset
 * will then be executed by the new threads. If the pool was paused before resetting it, the new
 * pool will be paused as well.
 *
 * @param nthreads The number of threads to use. The default value can be specified by setting
 * the `KVIKIO_NTHREADS` environment variable. If not set, the default value is 1.
 */
inline void reset(unsigned int nthreads = get_num_threads_from_env())
{
  _current_default_thread_pool.reset(nthreads);
}

/**
 * @brief Get the number of threads of the default thread pool.
 *
 * @return The number of threads in the current default thread pool.
 */
inline unsigned int nthreads() noexcept { return _current_default_thread_pool.get_thread_count(); }

}  // namespace kvikio::default_thread_pool
