/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <initializer_list>

namespace kvikio::detail {
// Must run before curl_global_init() or any use of the private OpenSSL copy.
// The candidate list is injectable for tests; OPENSSL_CONF always takes precedence.
void initialize_openssl(std::initializer_list<char const*> config_paths = {
                          "/etc/pki/tls/openssl.cnf", "/etc/ssl/openssl.cnf"});
}  // namespace kvikio::detail
