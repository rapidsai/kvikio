/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <openssl/ssl.h>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include "../../../cpp/src/detail/openssl.hpp"

int main(int argc, char** argv)
{
  try {
    auto const* before     = std::getenv("OPENSSL_CONF");
    std::string env_before = before == nullptr ? "<unset>" : before;
    if (argc >= 3) {
      kvikio::detail::initialize_openssl({argv[1], argv[2]});
    } else {
      kvikio::detail::initialize_openssl();
    }
    SSL_CTX* ctx = SSL_CTX_new(TLS_client_method());
    if (ctx == nullptr) { return 2; }
    std::cout << SSL_CTX_get_security_level(ctx) << "\n";
    if (argc == 4) {
      // Only test cipher-policy enforcement here; certificate trust is covered
      // by the remote TLS tests. This connects to the local pytest TLS server.
      BIO* connection = BIO_new_ssl_connect(ctx);
      if (connection == nullptr) {
        SSL_CTX_free(ctx);
        return 2;
      }
      BIO_set_conn_hostname(connection, argv[3]);
      int connected = BIO_do_connect(connection);
      int handshake = connected > 0 ? BIO_do_handshake(connection) : 0;
      BIO_free_all(connection);
      if (handshake <= 0) {
        SSL_CTX_free(ctx);
        std::cerr << "TLS handshake rejected\n";
        return 1;
      }
    }
    SSL_CTX_free(ctx);
    auto const* after = std::getenv("OPENSSL_CONF");
    return env_before == (after == nullptr ? "<unset>" : after) ? 0 : 3;
  } catch (std::exception const& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
