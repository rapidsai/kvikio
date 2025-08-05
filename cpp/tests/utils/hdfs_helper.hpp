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

#include <span>
#include <string>

namespace kvikio::test {

class WebHdfsTestHelper {
 private:
  std::string _host;
  std::string _port;
  std::string _username;
  std::string _url_before_path;

 public:
  WebHdfsTestHelper(std::string const& host, std::string const& port, std::string const& username);

  const std::string& host() const noexcept;

  const std::string& port() const noexcept;

  bool can_connect() noexcept;

  bool upload_data(std::span<std::byte> buffer, std::string const& remote_file_path) noexcept;

  bool delete_data(std::string const& remote_file_path) noexcept;
};

}  // namespace kvikio::test
