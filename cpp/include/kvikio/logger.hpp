/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <kvikio/logger_macros.hpp>
#include <kvikio/shim/utils.hpp>

#include <rapids_logger/logger.hpp>

namespace KVIKIO_EXPORT kvikio {
rapids_logger::logger& default_logger();
}  // namespace KVIKIO_EXPORT kvikio
