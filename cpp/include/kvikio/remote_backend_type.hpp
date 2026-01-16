/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>

namespace kvikio {

/**
 * @brief Enum representing the backend implementation for remote file I/O operations.
 *
 * KvikIO supports multiple libcurl-based backends for fetching data from remote endpoints (S3,
 * HTTP, etc.). Each backend has different performance characteristics.
 */
enum class RemoteBackendType : uint8_t {
  LIBCURL_EASY,  ///< Use libcurl's easy interface with a thread pool for parallelism. Each chunk is
                 ///< fetched by a separate thread using blocking curl_easy_perform() calls. This is
                 ///< the default backend.
  LIBCURL_MULTI_POLL,  ///< Use libcurl's multi interface with poll-based concurrent transfers. A
                       ///< single call manages multiple concurrent connections using
                       ///< curl_multi_poll(), with k-way buffering to overlap network I/O with
                       ///< host-to-device transfers. This can reduce thread overhead for
                       ///< high-connection-count scenarios.
};

}  // namespace kvikio
