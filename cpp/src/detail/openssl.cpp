/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "openssl.hpp"

#include <openssl/conf.h>
#include <openssl/err.h>
#include <openssl/ssl.h>
#include <sys/stat.h>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

namespace kvikio::detail {
namespace {
std::string openssl_errors()
{
  std::string errors;
  ERR_print_errors_cb(
    [](char const* text, std::size_t length, void* context) {
      static_cast<std::string*>(context)->append(text, length);
      return 1;
    },
    &errors);
  return errors;
}
}  // namespace

void initialize_openssl(std::initializer_list<char const*> config_paths)
{
  char const* filename = std::getenv("OPENSSL_CONF");
  if (filename == nullptr) {
    for (auto const* candidate : config_paths) {
      struct stat info{};
      if (stat(candidate, &info) == 0) {
        if (!S_ISREG(info.st_mode)) {
          throw std::runtime_error("OpenSSL configuration is not a regular file: " +
                                   std::string(candidate));
        }
        filename = candidate;
        break;
      }
      if (errno != ENOENT && errno != ENOTDIR) {
        throw std::runtime_error("Cannot inspect OpenSSL configuration " + std::string(candidate) +
                                 ": " + std::strerror(errno));
      }
    }
  }

  // Empty OPENSSL_CONF explicitly disables configuration. With no host file,
  // avoid falling back to the conda package's nonexistent build-time prefix.
  if (filename == nullptr || filename[0] == '\0') {
    if (OPENSSL_init_ssl(OPENSSL_INIT_NO_LOAD_CONFIG, nullptr) != 1) {
      throw std::runtime_error("Cannot initialize bundled OpenSSL: " + openssl_errors());
    }
    return;
  }

  std::unique_ptr<OPENSSL_INIT_SETTINGS, decltype(&OPENSSL_INIT_free)> settings{OPENSSL_INIT_new(),
                                                                                OPENSSL_INIT_free};
  if (!settings || OPENSSL_INIT_set_config_filename(settings.get(), filename) != 1) {
    throw std::runtime_error("Cannot configure bundled OpenSSL: " + openssl_errors());
  }
  // Unlike OpenSSL's default initialization, do not ignore a missing selected
  // file or a configuration module error: that could silently bypass policy.
  OPENSSL_INIT_set_config_file_flags(settings.get(), CONF_MFLAGS_DEFAULT_SECTION);
  if (OPENSSL_init_ssl(OPENSSL_INIT_LOAD_CONFIG, settings.get()) != 1) {
    throw std::runtime_error("Cannot load OpenSSL configuration " + std::string(filename) + ": " +
                             openssl_errors());
  }
}
}  // namespace kvikio::detail
