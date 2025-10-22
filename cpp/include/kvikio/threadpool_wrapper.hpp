/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <BS_thread_pool.hpp>

namespace kvikio {

template <typename pool_type>
class thread_pool_wrapper : public pool_type {
 public:
  /**
   * @brief Construct a new thread pool wrapper.
   *
   * @param nthreads The number of threads to use.
   */
  thread_pool_wrapper(unsigned int nthreads) : pool_type{nthreads} {}

  /**
   * @brief Reset the number of threads in the thread pool.
   *
   * @param nthreads The number of threads to use.
   */
  void reset(unsigned int nthreads) { pool_type::reset(nthreads); }
};

using BS_thread_pool = thread_pool_wrapper<BS::thread_pool>;

}  // namespace kvikio
