/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <iostream>
#include <numeric>

#include <kvikio/batch.hpp>
#include <kvikio/buffer.hpp>
#include <kvikio/cufile/driver.hpp>
#include <kvikio/defaults.hpp>
#include <kvikio/error.hpp>
#include <kvikio/file_handle.hpp>

using namespace std;

void check(bool condition)
{
  if (!condition) {
    std::cout << "Error" << std::endl;
    exit(-1);
  }
}

constexpr int NELEM      = 1024;                 // Number of elements used throughout the test
constexpr int SIZE       = NELEM * sizeof(int);  // Size of the memory allocations (in bytes)
constexpr int LARGE_SIZE = 8 * SIZE;             // LARGE SIZE to test partial submit (in bytes)

int main()
{
  cout << "KvikIO defaults: " << endl;
  if (kvikio::defaults::is_compat_mode_preferred()) {
    cout << "  Compatibility mode: enabled" << endl;
  } else {
    kvikio::DriverInitializer manual_init_driver;
    cout << "  Compatibility mode: disabled" << endl;
    kvikio::DriverProperties props;
    cout << "DriverProperties: " << endl;
    cout << "  nvfs version: " << props.get_nvfs_major_version() << "."
         << props.get_nvfs_minor_version() << endl;
    cout << "  Allow compatibility mode: " << std::boolalpha << props.get_nvfs_allow_compat_mode()
         << endl;
    cout << "  Pool mode - enabled: " << std::boolalpha << props.get_nvfs_poll_mode()
         << ", threshold: " << props.get_nvfs_poll_thresh_size() << " kb" << endl;
    cout << "  Max pinned memory: " << props.get_max_pinned_memory_size() << " kb" << endl;
    cout << "  Max batch IO size: " << props.get_max_batch_io_size() << endl;
  }

  std::vector<int> a(SIZE);
  std::iota(a.begin(), a.end(), 0);
  std::vector<int> b(SIZE);
  std::vector<int> c(SIZE);
  check(kvikio::is_host_memory(a.data()) == true);

  {
    kvikio::FileHandle file1("/tmp/test-file1", "w");
    kvikio::FileHandle file2("/tmp/test-file2", "w");
    std::future<std::size_t> fut1 = file1.pwrite(a.data(), SIZE);
    std::future<std::size_t> fut2 = file2.pwrite(a.data(), SIZE);
    size_t written                = fut1.get() + fut2.get();
    check(written == SIZE * 2);
    check(SIZE == file1.nbytes());
    check(SIZE == file2.nbytes());
    cout << "Write: " << written << endl;
  }
  {
    kvikio::FileHandle file1("/tmp/test-file1", "r");
    kvikio::FileHandle file2("/tmp/test-file2", "r");
    std::future<std::size_t> fut1 = file1.pread(b.data(), SIZE);
    std::future<std::size_t> fut2 = file2.pread(c.data(), SIZE);
    size_t read                   = fut1.get() + fut2.get();
    check(read == SIZE * 2);
    check(SIZE == file1.nbytes());
    check(SIZE == file2.nbytes());
    for (int i = 0; i < NELEM; ++i) {
      check(a[i] == b[i]);
      check(a[i] == c[i]);
    }
    cout << "Parallel POSIX read (" << kvikio::defaults::thread_pool_nthreads()
         << " threads): " << read << endl;
  }
}
