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

#include <optional>
#include <string>
#include <utility>

#include <kvikio/shim/libcurl.hpp>

namespace kvikio::detail {
/**
 * @brief Search CA bundle file and directory paths
 *
 * This function searches for the Certificate Authority (CA) paths required for TLS/SSL verification
 * in libcurl. The search is performed in the following priority order, returning as soon as either
 * a bundle file or a directory is found:
 * - CA bundle file: Check env vars CURL_CA_BUNDLE, SSL_CERT_FILE
 * - CA directory: Check env var SSL_CERT_DIR
 * - CA bundle file: Search distribution-specific locations for accessible bundle
 * - CA directory: Search distribution-specific locations for accessible directory
 * - CA bundle file: Check if curl's compile-time default bundle path is accessible
 * - CA directory: Check if curl's compile-time default directory path is accessible
 *
 * @return Result containing CA bundle file and CA certificate directory
 *
 * @exception std::runtime_error if neither CA bundle nor directory is found
 *
 * @note Environment Variables:
 * - CURL_CA_BUNDLE: Override CA bundle file location (curl-specific)
 * - SSL_CERT_FILE: Override CA bundle file location (OpenSSL-compatible)
 * - SSL_CERT_DIR: Override CA directory location (OpenSSL-compatible)
 */
std::pair<std::optional<std::string>, std::optional<std::string>> get_ca_paths();

/**
 * @brief Configure curl handle with discovered CA certificate paths
 *
 * As a performance optimization, the discovered CA certificate paths are cached to avoid repeated
 * searching.
 *
 * @param curl Curl handle to configure with CA certificate paths
 */
void set_up_ca_paths(CurlHandle& curl);
}  // namespace kvikio::detail
