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

#include <unistd.h>
#include <cstdlib>

#include <curl/curl.h>
#include <kvikio/detail/tls.hpp>
#include <kvikio/error.hpp>
#include <kvikio/shim/libcurl.hpp>
#include <optional>
#include <stdexcept>

namespace kvikio::detail {

namespace {
/**
 * @brief Find the first accessible path from multiple sources
 *
 * This helper function searches for a readable file or directory path by checking,
 * in order of priority:
 * - Environment variables (CURL_CA_BUNDLE, SSL_CERT_FILE, SSL_CERT_DIR)
 * - Distribution-specific standard locations
 * - libcurl compile-time defaults
 *
 * @param common_env_vars List of environment variable names to check
 * @param common_locations List of filesystem paths to search
 * @param compile_time_path Default path from libcurl's compile-time configuration (may be nullptr)
 *
 * @return Path to the first accessible location found, or std::nullopt if no accessible path
 * exists
 */
std::optional<std::string> path_helper(std::vector<std::string> const& common_env_vars,
                                       std::vector<std::string> const& common_locations,
                                       char const* compile_time_path)
{
  // Search environment variables
  for (auto const& env_var : common_env_vars) {
    auto const* path = std::getenv(env_var.data());
    if (path != nullptr) { return path; }
  }

  // Search common locations
  for (auto const& location : common_locations) {
    // Check whether the file/directory exists, and whether it grants read permission to the calling
    // process's real UID and GID. If the path is a symbolic link, it is dereferenced.
    auto const result = access(location.data(), R_OK);

    if (result != -1) { return location; }
  }

  // Use compile-time path if it exists
  if (compile_time_path != nullptr && access(compile_time_path, R_OK) != -1) {
    return compile_time_path;
  }

  return std::nullopt;
}
}  // namespace

std::pair<std::optional<std::string>, std::optional<std::string>> const& get_ca_paths()
{
  static auto paths = []() -> std::pair<std::optional<std::string>, std::optional<std::string>> {
    auto* version_info = curl_version_info(::CURLVERSION_NOW);
    KVIKIO_EXPECT(version_info != nullptr, "Failed to get curl version info");

    auto const ca_bundle_file = path_helper(
      {
        "CURL_CA_BUNDLE",  // curl program
        "SSL_CERT_FILE"    // OpenSSL
      },
      {"/etc/ssl/certs/ca-certificates.crt",  // Debian/Ubuntu, Arch, Alpine, Gentoo
       "/etc/pki/tls/certs/ca-bundle.crt",    // RHEL/CentOS/Rocky/AlmaLinux, Fedora
       "/etc/ssl/ca-bundle.pem",              // OpenSUSE/SLES
       "/etc/pki/tls/cert.pem",               // RHEL-based (symlink to ca-bundle.crt)
       "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem",  // Fedora 28+, RHEL 8+

       // Additional locations mentioned by libcurl:
       // https://github.com/curl/curl/blob/master/CMakeLists.txt
       "/usr/share/ssl/certs/ca-bundle.crt",
       "/usr/local/share/certs/ca-root-nss.crt",
       "/etc/ssl/cert.pem"},
      version_info->cainfo);

    auto const ca_directory = path_helper(
      {
        "SSL_CERT_DIR"  // OpenSSL
      },
      {
        "/etc/ssl/certs",      // Debian/Ubuntu, Arch, Alpine, OpenSUSE, Gentoo
        "/etc/pki/tls/certs",  // RHEL/CentOS/Rocky/AlmaLinux, Fedora
      },
      version_info->capath);

    // At least one path must exist
    KVIKIO_EXPECT(ca_bundle_file.has_value() || ca_directory.has_value(),
                  "Failed to find accessible CA certificates.",
                  std::runtime_error);

    return {ca_bundle_file, ca_directory};
  }();

  return paths;
}

void set_up_ca_paths(CurlHandle& curl)
{
  auto const& [ca_bundle_file, ca_directory] = get_ca_paths();

  if (ca_bundle_file.has_value()) {
    curl.setopt(CURLOPT_CAINFO, ca_bundle_file->c_str());
  } else if (ca_directory.has_value()) {
    curl.setopt(CURLOPT_CAPATH, ca_directory->c_str());
  }
}
}  // namespace kvikio::detail
