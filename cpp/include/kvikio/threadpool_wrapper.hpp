/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <atomic>
#include <cstdio>
#include <functional>
#include <memory>
#include <string>

#include <BS_thread_pool.hpp>

#include <pthread.h>

namespace kvikio {

/**
 * @brief Thread pool type used for parallel I/O operations.
 */
using ThreadPool = BS::thread_pool;

/**
 * @brief Build a `BS::thread_pool` init task that names each worker thread.
 *
 * The returned functor is intended to be passed to `BS::thread_pool`'s
 * constructor or `reset()` overloads that accept an `init_task`. It runs
 * once per worker as the pool starts each thread, setting the OS-level
 * thread name (`comm` on Linux) to `"<prefix>-<index>"` so profilers such
 * as nsys, `top -H`, and `/proc/<pid>/task/<tid>/comm` show meaningful
 * names instead of the parent process name.
 *
 * The per-thread index is assigned via an atomic counter captured by the
 * returned functor, so different pools can share the same prefix without
 * colliding.
 *
 * Linux caps thread names at 15 characters plus NUL, so keep @p prefix
 * short (typically 10 characters or fewer).
 *
 * @param prefix Name prefix, e.g. `"kvikio"` or `"kvikio-dev"`.
 * @return An init task suitable for `BS::thread_pool`.
 */
[[nodiscard]] inline std::function<void()> make_thread_pool_init_task(std::string prefix)
{
  auto counter = std::make_shared<std::atomic<unsigned int>>(0);
  return [counter = std::move(counter), prefix = std::move(prefix)]() {
    unsigned int const idx = counter->fetch_add(1, std::memory_order_relaxed);
    // Linux comm limit is 15 chars + NUL.
    std::array<char, 16> name{};
    std::snprintf(name.data(), name.size(), "%s-%u", prefix.c_str(), idx);
    pthread_setname_np(pthread_self(), name.data());
  };
}

}  // namespace kvikio
