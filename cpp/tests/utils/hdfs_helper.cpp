/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hdfs_helper.hpp"

#include <algorithm>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>

#include <kvikio/detail/remote_handle.hpp>
#include <kvikio/shim/libcurl.hpp>

namespace kvikio::test {

namespace {

/**
 * @brief Helper struct that wraps a buffer view and tracks how many data have been processed via an
 * offset value.
 */
struct tracked_buffer_t {
  std::span<std::byte> buffer;
  std::size_t offset;
};

/**
 * @brief Callback for `CURLOPT_READFUNCTION` to upload data.
 *
 * @param data
 * @param size Curl internal implementation always sets this parameter to 1
 * @param num_bytes_max The maximum number of bytes that can be uploaded
 * @param userdata Must be cast from `tracked_buffer_t*`
 * @return The number of bytes that have been copied to the transfer buffer.
 */
std::size_t callback_upload(char* data, std::size_t size, std::size_t num_bytes_max, void* userdata)
{
  auto new_data_size_max = size * num_bytes_max;
  auto* tracked_buffer   = reinterpret_cast<tracked_buffer_t*>(userdata);

  // All data have been uploaded. Nothing more to do.
  if (tracked_buffer->offset >= tracked_buffer->buffer.size()) { return 0; }

  auto copy_size =
    std::min(new_data_size_max, tracked_buffer->buffer.size() - tracked_buffer->offset);
  std::memcpy(data, tracked_buffer->buffer.data() + tracked_buffer->offset, copy_size);
  tracked_buffer->offset += copy_size;

  return copy_size;
}
}  // namespace

WebHdfsTestHelper::WebHdfsTestHelper(std::string const& host,
                                     std::string const& port,
                                     std::string const& username)
  : _host{host}, _port{port}, _username{username}
{
  std::stringstream ss;
  ss << "http://" << host << ":" << port << "/webhdfs/v1";
  _url_before_path = ss.str();
}

bool WebHdfsTestHelper::can_connect() noexcept
{
  try {
    auto curl = create_curl_handle();

    std::stringstream ss;
    ss << _url_before_path << "/?user.name=" << _username << "&op=GETHOMEDIRECTORY";

    curl.setopt(CURLOPT_URL, ss.str().c_str());

    std::string response{};
    curl.setopt(CURLOPT_WRITEDATA, &response);
    curl.setopt(CURLOPT_WRITEFUNCTION, kvikio::detail::callback_get_string_response);
    curl.setopt(CURLOPT_FOLLOWLOCATION, 1L);
    curl.perform();
    return true;
  } catch (std::exception const& e) {
    std::cout << e.what() << "\n";
    return false;
  }
}

bool WebHdfsTestHelper::upload_data(std::span<std::byte> buffer,
                                    std::string const& remote_file_path) noexcept
{
  try {
    // Official reference on how to create and write to a file:
    // https://hadoop.apache.org/docs/current/hadoop-project-dist/hadoop-hdfs/WebHDFS.html#Create_and_Write_to_a_File
    std::string redirect_url;

    {
      // Step 1: Submit a HTTP PUT request without automatically following redirects and without
      // sending the file data.
      auto curl = create_curl_handle();

      std::stringstream ss;
      ss << _url_before_path << remote_file_path << "?user.name=" << _username << "&op=CREATE";
      std::string redirect_data_node_location{};

      curl.setopt(CURLOPT_URL, ss.str().c_str());
      curl.setopt(CURLOPT_FOLLOWLOCATION, 0L);
      curl.setopt(CURLOPT_CUSTOMREQUEST, "PUT");

      std::string response{};
      curl.setopt(CURLOPT_HEADERDATA, &response);
      curl.setopt(CURLOPT_HEADERFUNCTION, kvikio::detail::callback_get_string_response);

      curl.perform();

      long http_status_code{};
      curl.getinfo(CURLINFO_RESPONSE_CODE, &http_status_code);
      KVIKIO_EXPECT(http_status_code == 307, "Redirection from name node to data node failed.");

      std::regex const pattern{R"(Location:\s*(.*)\s*)"};
      std::smatch match_results;
      bool found = std::regex_search(response, match_results, pattern);
      KVIKIO_EXPECT(found,
                    "Regular expression search failed. Cannot extract redirect location from the "
                    "JSON response.");
      redirect_url = match_results[1].str();
    }

    {
      // Step 2: Submit another HTTP PUT request using the URL in the Location header with the file
      // data to be written.
      auto curl = create_curl_handle();
      curl.setopt(CURLOPT_URL, redirect_url.c_str());
      curl.setopt(CURLOPT_UPLOAD, 1L);

      tracked_buffer_t tracked_buffer{.buffer = buffer, .offset = 0};
      curl.setopt(CURLOPT_READDATA, &tracked_buffer);
      curl.setopt(CURLOPT_READFUNCTION, callback_upload);
      curl.setopt(CURLOPT_INFILESIZE_LARGE, static_cast<curl_off_t>(buffer.size()));

      curl.perform();

      long http_status_code{};
      curl.getinfo(CURLINFO_RESPONSE_CODE, &http_status_code);
      KVIKIO_EXPECT(http_status_code == 201, "File creation failed.");
    }

    return true;
  } catch (std::exception const& e) {
    std::cout << e.what() << "\n";
    return false;
  }
}

bool WebHdfsTestHelper::delete_data(std::string const& remote_file_path) noexcept
{
  try {
    // Official reference on how to delete a file:
    // https://hadoop.apache.org/docs/current/hadoop-project-dist/hadoop-hdfs/WebHDFS.html#Delete_a_File.2FDirectory
    auto curl = create_curl_handle();

    std::stringstream ss;
    ss << _url_before_path << remote_file_path << "?user.name=" << _username << "&op=DELETE";
    std::string const url = ss.str();
    std::string redirect_data_node_location{};

    curl.setopt(CURLOPT_URL, url.c_str());
    curl.setopt(CURLOPT_FOLLOWLOCATION, 1L);
    curl.setopt(CURLOPT_CUSTOMREQUEST, "DELETE");

    std::string response{};
    curl.setopt(CURLOPT_HEADERDATA, &response);
    curl.setopt(CURLOPT_HEADERFUNCTION, kvikio::detail::callback_get_string_response);

    curl.perform();

    long http_status_code{};
    curl.getinfo(CURLINFO_RESPONSE_CODE, &http_status_code);
    KVIKIO_EXPECT(http_status_code == 200, "File deletion failed.");

    return true;
  } catch (std::exception const& e) {
    std::cout << e.what() << "\n";
    return false;
  }
}
}  // namespace kvikio::test
