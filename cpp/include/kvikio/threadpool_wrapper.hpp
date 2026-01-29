/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <BS_thread_pool.hpp>

namespace kvikio {

/**
 * @brief Thread pool type used for parallel I/O operations.
 */
using ThreadPool = BS::thread_pool;

}  // namespace kvikio
