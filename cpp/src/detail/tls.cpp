/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
 * @brief Search for a CA certificate path using environment variables
 *
 * @param env_vars Environment variable names to check in order
 * @return Path string if found in any environment variable, std::nullopt otherwise
 */
std::optional<std::string> find_ca_path_from_env_var(std::vector<std::string> const& env_vars)
{
  for (auto const& env_var : env_vars) {
    auto const* path = std::getenv(env_var.data());
    if (path != nullptr) { return path; }
  }

  return std::nullopt;
}

/**
 * @brief Search for a CA certificate path in standard system locations
 *
 * @param system_paths file system paths to check in order
 * @return First accessible path if found, std::nullopt otherwise
 */
std::optional<std::string> find_ca_path_in_system_locations(
  std::vector<std::string> const& system_paths)
{
  for (auto const& path : system_paths) {
    // Check whether the file/directory exists, and whether it grants read permission to the calling
    // process's real UID and GID. If the path is a symbolic link, it is dereferenced.
    auto const result = access(path.data(), R_OK);

    if (result != -1) { return path; }
  }

  return std::nullopt;
}

/**
 * @brief Get CA certificate path from curl's compile-time defaults
 *
 * @param default_path Path provided by curl_version_info (may be nullptr)
 * @return Path string if accessible, std::nullopt otherwise
 */
std::optional<std::string> get_ca_path_from_curl_defaults(char const* default_path)
{
  if (default_path != nullptr && access(default_path, R_OK) != -1) { return default_path; }

  return std::nullopt;
}
}  // namespace

std::pair<std::optional<std::string>, std::optional<std::string>> get_ca_paths()
{
  auto* version_info = curl_version_info(::CURLVERSION_NOW);
  KVIKIO_EXPECT(version_info != nullptr, "Failed to get curl version info", std::runtime_error);

  std::optional<std::string> ca_bundle_file;
  std::optional<std::string> ca_directory;

  // Priority 1: CA bundle file from environment variables
  ca_bundle_file = find_ca_path_from_env_var({
    "CURL_CA_BUNDLE",  // curl program
    "SSL_CERT_FILE"    // OpenSSL
  });
  if (ca_bundle_file.has_value()) { return {ca_bundle_file, ca_directory}; }

  // Priority 2: CA directory from environment variables
  ca_directory = find_ca_path_from_env_var({
    "SSL_CERT_DIR"  // OpenSSL
  });
  if (ca_directory.has_value()) { return {ca_bundle_file, ca_directory}; }

  // Priority 3: CA bundle file from system locations
  ca_bundle_file = find_ca_path_in_system_locations(
    {"/etc/ssl/certs/ca-certificates.crt",                 // Debian/Ubuntu, Arch, Alpine, Gentoo
     "/etc/pki/tls/certs/ca-bundle.crt",                   // RHEL/CentOS/Rocky/AlmaLinux, Fedora
     "/etc/ssl/ca-bundle.pem",                             // OpenSUSE/SLES
     "/etc/pki/tls/cert.pem",                              // RHEL-based (symlink to ca-bundle.crt)
     "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem",  // Fedora 28+, RHEL 8+

     // Additional locations mentioned by libcurl:
     // https://github.com/curl/curl/blob/master/CMakeLists.txt
     "/usr/share/ssl/certs/ca-bundle.crt",
     "/usr/local/share/certs/ca-root-nss.crt",
     "/etc/ssl/cert.pem"});
  if (ca_bundle_file.has_value()) { return {ca_bundle_file, ca_directory}; }

  // Priority 4: CA directory from system locations
  ca_directory = find_ca_path_in_system_locations({
    "/etc/ssl/certs",     // Debian/Ubuntu, Arch, Alpine, OpenSUSE, Gentoo
    "/etc/pki/tls/certs"  // RHEL/CentOS/Rocky/AlmaLinux, Fedora
  });
  if (ca_directory.has_value()) { return {ca_bundle_file, ca_directory}; }

  // Priority 5: CA bundle file from curl compile-time defaults
  ca_bundle_file = get_ca_path_from_curl_defaults(version_info->cainfo);
  if (ca_bundle_file.has_value()) { return {ca_bundle_file, ca_directory}; }

  // Priority 6: CA directory from curl compile-time defaults
  ca_directory = get_ca_path_from_curl_defaults(version_info->capath);
  if (ca_directory.has_value()) { return {ca_bundle_file, ca_directory}; }

  // At least one path must exist
  KVIKIO_EXPECT(ca_bundle_file.has_value() || ca_directory.has_value(),
                "Failed to find accessible CA certificates.",
                std::runtime_error);
  return {ca_bundle_file, ca_directory};
}

void set_up_ca_paths(CurlHandle& curl)
{
  static auto const [ca_bundle_file, ca_directory] = get_ca_paths();

  if (ca_bundle_file.has_value()) {
    curl.setopt(CURLOPT_CAINFO, ca_bundle_file->c_str());
    curl.setopt(CURLOPT_CAPATH, nullptr);
  } else if (ca_directory.has_value()) {
    curl.setopt(CURLOPT_CAINFO, nullptr);
    curl.setopt(CURLOPT_CAPATH, ca_directory->c_str());
  }
}
}  // namespace kvikio::detail
