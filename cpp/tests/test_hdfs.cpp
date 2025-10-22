/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>
#include <cstdlib>
#include <memory>

#include <kvikio/file_handle.hpp>
#include <kvikio/hdfs.hpp>
#include <kvikio/remote_handle.hpp>

#include "utils/hdfs_helper.hpp"
#include "utils/utils.hpp"

// This test makes the following assumptions:
// - This KvikIO unit test is run on the "name node" of a WebHDFS server.
// - Port 9870 (default for WebHDFS) is being used to listen to the requests.
// - The environment variable `KVIKIO_USER` is specified prior to the test. It contains a valid user
//   name that has been granted access to the HDFS.
// - The user has the proper permission to create a file under the `/tmp` directory on the HDFS.
// - If the unit test is run within a Docker. The following arguments are passed to the `docker run`
//   command:
//   - `--network host`
//   - `--env KVIKIO_USER=<hdfs_username>`
//
// If any of these assumptions is not satisfied, this unit test is expected to be skipped
// gracefully.

using value_type = double;

namespace kvikio::test {
struct Config {
  std::size_t num_elements{1024ull * 1024ull};
  std::vector<value_type> host_buf;
  kvikio::test::DevBuffer<value_type> dev_buf;
  std::string host;
  std::string port;
  std::string _username;
  std::string remote_file_path;
  bool file_created{false};
};
}  // namespace kvikio::test

class WebHdfsTest : public testing::Test {
 protected:
  static void SetUpTestSuite()
  {
    config.num_elements = 1024ull * 1024ull;
    config.host_buf.resize(config.num_elements);
    std::iota(config.host_buf.begin(), config.host_buf.end(), 0);

    config.dev_buf = kvikio::test::DevBuffer<value_type>{config.host_buf};

    config.host = "localhost";
    config.port = "9870";

    config.remote_file_path = "/tmp/kvikio-test-webhdfs.bin";

    auto res = std::getenv("KVIKIO_USER");
    if (res) {
      config._username = res;
    } else {
      GTEST_SKIP() << "Environment variable KVIKIO_USER is not set for this test.";
    }

    webhdfs_helper =
      std::make_unique<kvikio::test::WebHdfsTestHelper>(config.host, config.port, config._username);

    if (!webhdfs_helper->can_connect()) {
      GTEST_SKIP() << "Cannot connect to WebHDFS. Skipping all tests for this fixture.";
    }

    std::span<std::byte> buffer{reinterpret_cast<std::byte*>(config.host_buf.data()),
                                config.host_buf.size() * sizeof(value_type)};
    if (!webhdfs_helper->upload_data(buffer, config.remote_file_path)) {
      GTEST_SKIP()
        << "Failed to upload test data using WebHDFS. Skipping all tests for this fixture.";
    };

    config.file_created = true;
  }

  static void TearDownTestSuite()
  {
    if (config.file_created) { webhdfs_helper->delete_data(config.remote_file_path); }
  }

  static kvikio::test::Config config;
  static std::unique_ptr<kvikio::test::WebHdfsTestHelper> webhdfs_helper;
};

kvikio::test::Config WebHdfsTest::config{};
std::unique_ptr<kvikio::test::WebHdfsTestHelper> WebHdfsTest::webhdfs_helper{};

TEST_F(WebHdfsTest, constructor)
{
  auto do_test = [&](kvikio::RemoteHandle& remote_handle) {
    kvikio::test::DevBuffer<value_type> out_device_buf(config.num_elements);
    auto read_size    = remote_handle.read(out_device_buf.ptr, remote_handle.nbytes());
    auto out_host_buf = out_device_buf.to_vector();
    for (std::size_t i = 0; i < config.num_elements; ++i) {
      EXPECT_EQ(config.host_buf[i], out_host_buf[i]);
    }
    EXPECT_EQ(read_size, remote_handle.nbytes());
  };

  std::stringstream ss;
  ss << "http://" << config.host << ":" << config.port << "/webhdfs/v1" << config.remote_file_path
     << "?user.name=" << config._username;
  std::vector<kvikio::RemoteHandle> remote_handles;

  remote_handles.emplace_back(std::make_unique<kvikio::WebHdfsEndpoint>(ss.str()));
  remote_handles.emplace_back(std::make_unique<kvikio::WebHdfsEndpoint>(
    config.host, config.port, config.remote_file_path, config._username));

  for (auto& remote_handle : remote_handles) {
    do_test(remote_handle);
  }
}

TEST_F(WebHdfsTest, read_parallel)
{
  auto do_test = [&](std::string const& url,
                     std::size_t num_elements_to_skip,
                     std::size_t num_elements_to_read,
                     std::size_t task_size) {
    kvikio::RemoteHandle remote_handle{std::make_unique<kvikio::WebHdfsEndpoint>(url)};
    auto const offset             = num_elements_to_skip * sizeof(value_type);
    auto const expected_read_size = num_elements_to_read * sizeof(value_type);

    // host
    {
      std::vector<value_type> out_host_buf(num_elements_to_read, {});
      auto fut = remote_handle.pread(out_host_buf.data(), expected_read_size, offset, task_size);
      auto const read_size = fut.get();
      for (std::size_t i = num_elements_to_skip; i < num_elements_to_read; ++i) {
        EXPECT_EQ(config.host_buf[i], out_host_buf[i - num_elements_to_skip]);
      }
      EXPECT_EQ(read_size, expected_read_size);
    }

    // device
    {
      kvikio::test::DevBuffer<value_type> out_device_buf(num_elements_to_read);
      auto fut = remote_handle.pread(out_device_buf.ptr, expected_read_size, offset, task_size);
      auto const read_size = fut.get();
      auto out_host_buf    = out_device_buf.to_vector();
      for (std::size_t i = num_elements_to_skip; i < num_elements_to_read; ++i) {
        EXPECT_EQ(config.host_buf[i], out_host_buf[i - num_elements_to_skip]);
      }
      EXPECT_EQ(read_size, expected_read_size);
    }
  };

  std::stringstream ss;
  ss << "http://" << config.host << ":" << config.port << "/webhdfs/v1" << config.remote_file_path
     << "?user.name=" << config._username;
  std::vector<std::size_t> task_sizes{256, 1024, kvikio::defaults::task_size()};

  for (const auto& task_size : task_sizes) {
    for (const auto& num_elements_to_read : {10, 9999}) {
      for (const auto& num_elements_to_skip : {0, 10, 100, 1000, 9999}) {
        do_test(ss.str(), num_elements_to_skip, num_elements_to_read, task_size);
      }
    }
  }
}
