/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <span>
#include <string>

namespace kvikio::test {

/**
 * @brief Helper class to create and upload a file on WebHDFS so as to enable read testing.
 */
class WebHdfsTestHelper {
 private:
  std::string _host;
  std::string _port;
  std::string _username;
  std::string _url_before_path;

 public:
  WebHdfsTestHelper(std::string const& host, std::string const& port, std::string const& username);

  /**
   * @brief Whether KvikIO can connect to the WebHDFS server.
   *
   * @return A boolean answer.
   */
  bool can_connect() noexcept;

  /**
   * @brief Copy the data from a host buffer to a remote file on the WebHDFS server.
   *
   * @param buffer View to the host buffer whose data will be copied to the WebHDFS server
   * @param remote_file_path Remote file path
   * @return True if the file has been successfully uploaded; false otherwise.
   */
  bool upload_data(std::span<std::byte> buffer, std::string const& remote_file_path) noexcept;

  /**
   * @brief Delete a remote file on the WebHDFS server.
   *
   * @param remote_file_path Remote file path
   * @return True if the file has been successfully deleted; false otherwise.
   */
  bool delete_data(std::string const& remote_file_path) noexcept;
};

}  // namespace kvikio::test
