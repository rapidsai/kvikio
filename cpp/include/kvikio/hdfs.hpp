/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <optional>

#include <kvikio/remote_handle.hpp>

namespace kvikio {

/**
 * @brief A remote endpoint for Apache Hadoop WebHDFS.
 *
 * This endpoint is for accessing HDFS files via the WebHDFS REST API over HTTP/HTTPS. If KvikIO is
 * run within Docker, pass `--network host` to the `docker run` command to ensure proper name node
 * connectivity.
 */
class WebHdfsEndpoint : public RemoteEndpoint {
 private:
  std::string _url;
  std::optional<std::string> _username;

 public:
  /**
   * @brief Create an WebHDFS endpoint from a url.
   *
   * @param url The WebHDFS HTTP/HTTPS url to the remote file.
   * @param username Optional user name.
   *
   * @note The optional username for authentication is determined in the following descending
   * priority order:
   * - Function parameter `username`
   * - Query string in URL (?user.name=xxx)
   * - Environment variable `KVIKIO_WEBHDFS_USERNAME`
   */
  explicit WebHdfsEndpoint(std::string url, std::optional<std::string> username = std::nullopt);

  /**
   * @brief Create an WebHDFS endpoint from the host, port, file path and optionally username.
   *
   * @param host Host
   * @param port Port
   * @param remote_file_path Remote file path
   * @param username Optional user name.
   *
   * @note The optional username for authentication is determined in the following descending
   * priority order:
   * - Function parameter `username`
   * - Environment variable `KVIKIO_WEBHDFS_USERNAME`
   */
  explicit WebHdfsEndpoint(std::string host,
                           std::string port,
                           std::string remote_file_path,
                           std::optional<std::string> username = std::nullopt);

  ~WebHdfsEndpoint() override = default;
  void setopt(CurlHandle& curl) override;
  std::string str() const override;
  std::size_t get_file_size() override;
  void setup_range_request(CurlHandle& curl, std::size_t file_offset, std::size_t size) override;

  /**
   * @brief Whether the given URL is valid for the WebHDFS endpoints.
   *
   * @param url A URL.
   * @return Boolean answer.
   */
  static bool is_url_valid(std::string const& url) noexcept;
};
}  // namespace kvikio
