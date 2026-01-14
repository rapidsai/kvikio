/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>

namespace kvikio {

enum class RemoteBackendType : uint8_t {
  LIBCURL_EASY,
  LIBCURL_MULTI_POLL,
};

}  // namespace kvikio
