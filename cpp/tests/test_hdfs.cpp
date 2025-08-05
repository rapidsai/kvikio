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

#include <gtest/gtest.h>
#include <memory>

#include <kvikio/file_handle.hpp>
#include <kvikio/hdfs.hpp>
#include <kvikio/remote_handle.hpp>

#include "utils/hdfs_helper.hpp"
#include "utils/utils.hpp"

class WebHdfsTest : public testing::Test {
 protected:
  void SetUp() override
  {
    kvikio::test::TempDir tmp_dir{true};
    _filepath                = tmp_dir.path() / "test.bin";
    std::size_t num_elements = 1024ull * 1024ull;
    _host_buf                = CreateTempFile<value_type>(_filepath, num_elements);
    _dev_buf                 = kvikio::test::DevBuffer{_host_buf};

    _host             = "localhost";
    _port             = "9870";
    _username         = "rladmin";
    _remote_file_path = "/tmp/kvikio-test-webhdfs.bin";

    _webhdfs_helper = std::make_unique<kvikio::test::WebHdfsTestHelper>(_host, _port, _username);

    if (!_webhdfs_helper->can_connect()) {
      GTEST_SKIP() << "Cannot connect to WebHDFS. Skipping all tests for this fixture.";
    }

    std::span<std::byte> buffer{reinterpret_cast<std::byte*>(_host_buf.data()),
                                _host_buf.size() * sizeof(value_type)};
    if (!_webhdfs_helper->upload_data(buffer, _remote_file_path)) {
      GTEST_SKIP()
        << "Failed to upload test data using WebHDFS. Skipping all tests for this fixture.";
    };

    std::stringstream ss;
    ss << "http://" << _host << ":" << _port << "/webhdfs/v1" << _remote_file_path;
    _url_without_query = ss.str();

    ss << "?op=OPEN";
    _url_with_query = ss.str();
  }

  void TearDown() override { _webhdfs_helper->delete_data(_remote_file_path); }

  template <typename T>
  std::vector<T> CreateTempFile(std::string const& filepath, std::size_t num_elements)
  {
    std::vector<T> v(num_elements);
    std::iota(v.begin(), v.end(), 0);
    kvikio::FileHandle f(filepath, "w");
    auto fut = f.pwrite(v.data(), v.size() * sizeof(T));
    fut.get();
    _file_size = f.nbytes();
    return v;
  }

  std::filesystem::path _dir;
  std::filesystem::path _filepath;
  std::size_t _file_size;
  std::vector<double> _host_buf;
  using value_type = decltype(_host_buf)::value_type;
  kvikio::test::DevBuffer<value_type> _dev_buf;

  std::string _url_with_query;
  std::string _url_without_query;

  std::string _host;
  std::string _port;
  std::string _username;
  std::string _remote_file_path;

  std::unique_ptr<kvikio::test::WebHdfsTestHelper> _webhdfs_helper;
};

TEST_F(WebHdfsTest, webhdfs_remote_handle)
{
  //   auto do_test = [&](std::string const& url,
  //                      std::size_t num_elements_to_skip,
  //                      std::size_t num_elements_to_read,
  //                      std::size_t task_size) {
  //     kvikio::RemoteHandle remote_handle{std::make_unique<kvikio::WebHdfsEndpoint>(url)};
  //     auto const offset             = num_elements_to_skip * sizeof(value_type);
  //     auto const expected_read_size = num_elements_to_read * sizeof(value_type);

  //     // host
  //     {
  //       std::vector<value_type> out_host_buf(num_elements_to_read, {});
  //       auto fut = remote_handle.pread(out_host_buf.data(), expected_read_size, offset,
  //       task_size); auto const read_size = fut.get(); for (std::size_t i = num_elements_to_skip;
  //       i < num_elements_to_read; ++i) {
  //         EXPECT_EQ(_host_buf[i], out_host_buf[i - num_elements_to_skip]);
  //       }
  //       EXPECT_EQ(read_size, expected_read_size);
  //     }

  //     // device
  //     {
  //       kvikio::test::DevBuffer<value_type> out_device_buf(num_elements_to_read);
  //       auto fut = remote_handle.pread(out_device_buf.ptr, expected_read_size, offset,
  //       task_size); auto const read_size = fut.get(); auto out_host_buf    =
  //       out_device_buf.to_vector(); for (std::size_t i = num_elements_to_skip; i <
  //       num_elements_to_read; ++i) {
  //         EXPECT_EQ(_host_buf[i], out_host_buf[i - num_elements_to_skip]);
  //       }
  //       EXPECT_EQ(read_size, expected_read_size);
  //     }
  //   };

  //   std::array urls{_url_with_query, _url_without_query};
  //   std::vector<std::size_t> task_sizes{256, 1024, kvikio::defaults::task_size()};

  //   for (auto const& url : urls) {
  //     for (const auto& task_size : task_sizes) {
  //       for (const auto& num_elements_to_read : {10, 9999}) {
  //         for (const auto& num_elements_to_skip : {0, 10, 100, 1000, 9999}) {
  //           do_test(url, num_elements_to_skip, num_elements_to_read, task_size);
  //         }
  //       }
  //     }
  //   }

  kvikio::RemoteHandle remote_handle{std::make_unique<kvikio::WebHdfsEndpoint>(_url_without_query)};
  auto file_size = remote_handle.nbytes();
  std::cout << "file_size: " << file_size << "\n";
  std::vector<value_type> out_host_buf(file_size / sizeof(value_type), {});
  auto fut             = remote_handle.pread(out_host_buf.data(), file_size);
  auto const read_size = fut.get();
  EXPECT_EQ(read_size, file_size);
}
