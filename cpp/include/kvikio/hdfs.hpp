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

#include <kvikio/remote_handle.hpp>

namespace kvikio {

/**
 * @brief A remote endpoint for Apache Hadoop WebHDFS.
 *
 * If KvikIO is run within a Docker, the argument `--network host` needs to be passed to the `docker
 * run` command.
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
   */
  explicit WebHdfsEndpoint(std::string url);

  /**
   * @brief Create an WebHDFS endpoint from the host, port, file path and optionally username.
   *
   * @param host Host
   * @param port Port
   * @param remote_file_path Remote file path
   * @param username User name
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
   * @brief Whether the given URL is compatible with the WebHDFS endpoint.
   *
   * @param url A URL.
   * @return Boolean answer.
   */
  static bool is_url_compatible(std::string const& url) noexcept;
};
}  // namespace kvikio
