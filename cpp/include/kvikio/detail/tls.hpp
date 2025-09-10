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
 * @brief Search and cache CA bundle file and directory paths
 *
 * This function performs a one-time discovery of Certificate Authority (CA) paths required for
 * TLS/SSL verification in libcurl. It searches for both:
 * - CA bundle file (for CURLOPT_CAINFO in libcurl): A single file containing concatenated CA
 * certificates
 * - CA directory (for CURLOPT_CAPATH in libcurl): A directory containing individual CA
 * certificate files with hash-based naming (requires c_rehash)
 *
 * The search follows this priority order:
 * - Environment variables (CURL_CA_BUNDLE, SSL_CERT_FILE, SSL_CERT_DIR)
 * - Distribution-specific standard locations
 * - libcurl compile-time defaults
 *
 * The results are cached to avoid repeated searches.
 *
 * @return Cached result containing CA bundle file and CA certificate directory
 *
 * @exception std::runtime_error if neither CA bundle nor directory is found
 *
 * @par Environment Variables:
 * - CURL_CA_BUNDLE: Override CA bundle file location (curl-specific)
 * - SSL_CERT_FILE: Override CA bundle file location (OpenSSL-compatible)
 * - SSL_CERT_DIR: Override CA directory location (OpenSSL-compatible)
 */
std::pair<std::optional<std::string>, std::optional<std::string>> const& get_ca_paths();

void set_up_ca_paths(CurlHandle& curl);
}  // namespace kvikio::detail
