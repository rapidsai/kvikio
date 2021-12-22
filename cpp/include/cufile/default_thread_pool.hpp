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

#include <cufile/thread_pool.hpp>

namespace cufile::default_thread_pool {
namespace {
inline unsigned int get_num_threads_from_env()
{
  const char* nthreads = std::getenv("CUFILE_NTHREADS");
  if (nthreads == nullptr) { return 1; }
  const int n = std::stoi(nthreads);
  if (n <= 0) { throw std::invalid_argument("CUFILE_NTHREADS has to be a positive integer"); }
  return std::stoi(nthreads);
}
/*NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)*/
inline thread_pool _current_default_thread_pool{get_num_threads_from_env()};

}  // namespace

inline thread_pool& get() { return _current_default_thread_pool; }
inline void reset(unsigned int nthreads = get_num_threads_from_env())
{
  _current_default_thread_pool.reset(nthreads);
}
inline unsigned int nthreads() { return _current_default_thread_pool.get_thread_count(); }

}  // namespace cufile::default_thread_pool
