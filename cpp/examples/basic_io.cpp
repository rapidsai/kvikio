/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <iostream>

#include <cuda_runtime_api.h>

#include <kvikio/buffer.hpp>
#include <kvikio/defaults.hpp>
#include <kvikio/driver.hpp>
#include <kvikio/file_handle.hpp>

using namespace std;

void check(bool condition)
{
  if (!condition) {
    std::cout << "Error" << std::endl;
    exit(-1);
  }
}

constexpr int NELEM = 1024;
constexpr int SIZE  = NELEM * sizeof(int);

int main()
{
  check(cudaSetDevice(0) == cudaSuccess);

  cout << "KvikIO defaults: " << endl;
  if (kvikio::defaults::compat_mode()) {
    cout << "  Compatibility mode: enabled" << endl;
  } else {
    kvikio::DriverInitializer manual_init_driver;
    cout << "  Compatibility mode: disabled" << endl;
    kvikio::DriverProperties props;
    cout << "DriverProperties: " << endl;
    cout << "  Version: " << props.get_nvfs_major_version() << "." << props.get_nvfs_minor_version()
         << endl;
    cout << "  Allow compatibility mode: " << std::boolalpha << props.get_nvfs_allow_compat_mode()
         << endl;
    cout << "  Pool mode - enabled: " << std::boolalpha << props.get_nvfs_poll_mode()
         << ", threshold: " << props.get_nvfs_poll_thresh_size() << " kb" << endl;
    cout << "  Max pinned memory: " << props.get_max_pinned_memory_size() << " kb" << endl;
  }

  int* a{};
  check(cudaHostAlloc((void**)&a, SIZE, cudaHostAllocDefault) == cudaSuccess);
  for (int i = 0; i < NELEM; ++i) {
    a[i] = i;
  }
  int* b      = (int*)malloc(SIZE);
  void* a_dev = nullptr;
  void* b_dev = nullptr;
  void* c_dev = nullptr;
  check(cudaMalloc(&a_dev, SIZE) == cudaSuccess);
  check(cudaMalloc(&b_dev, SIZE) == cudaSuccess);
  check(cudaMalloc(&c_dev, SIZE) == cudaSuccess);

  check(kvikio::is_host_memory(a) == true);
  check(kvikio::is_host_memory(b) == true);
  check(kvikio::is_host_memory(a_dev) == false);
  check(kvikio::is_host_memory(b_dev) == false);
  check(kvikio::is_host_memory(c_dev) == false);

  {
    kvikio::FileHandle f("/tmp/test-file", "w");
    check(cudaMemcpy(a_dev, a, SIZE, cudaMemcpyHostToDevice) == cudaSuccess);
    size_t written = f.pwrite(a_dev, SIZE, 0, 1).get();
    check(written == SIZE);
    check(written == f.nbytes());
    cout << "Write: " << written << endl;
  }
  {
    kvikio::FileHandle f("/tmp/test-file", "r");
    size_t read = f.pread(b_dev, SIZE, 0, 1).get();
    check(read == SIZE);
    check(read == f.nbytes());
    cout << "Read:  " << read << endl;
    check(cudaMemcpy(b, b_dev, SIZE, cudaMemcpyDeviceToHost) == cudaSuccess);
    for (int i = 0; i < NELEM; ++i) {
      check(a[i] == b[i]);
    }
  }
  kvikio::defaults::thread_pool_nthreads_reset(16);
  {
    kvikio::FileHandle f("/tmp/test-file", "w");
    size_t written = f.pwrite(a_dev, SIZE).get();
    check(written == SIZE);
    check(written == f.nbytes());
    cout << "Parallel write (" << kvikio::defaults::thread_pool_nthreads()
         << " threads): " << written << endl;
  }
  {
    kvikio::FileHandle f("/tmp/test-file", "r");
    size_t read = f.pread(b_dev, SIZE, 0).get();
    cout << "Parallel write (" << kvikio::defaults::thread_pool_nthreads() << " threads): " << read
         << endl;
    check(cudaMemcpy(b, b_dev, SIZE, cudaMemcpyDeviceToHost) == cudaSuccess);
    for (int i = 0; i < NELEM; ++i) {
      check(a[i] == b[i]);
    }
  }
  {
    kvikio::FileHandle f("/tmp/test-file", "r+", kvikio::FileHandle::m644);
    kvikio::buffer_register(c_dev, SIZE);
    size_t read = f.pread(b_dev, SIZE).get();
    check(read == SIZE);
    check(read == f.nbytes());
    kvikio::buffer_deregister(c_dev);
  }

  {
    kvikio::FileHandle f("/tmp/test-file", "w");
    size_t written = f.pwrite(a, SIZE).get();
    check(written == SIZE);
    check(written == f.nbytes());
    cout << "Parallel POSIX write (" << kvikio::defaults::thread_pool_nthreads()
         << " threads): " << written << endl;
  }
  {
    kvikio::FileHandle f("/tmp/test-file", "r");
    size_t read = f.pread(b, SIZE).get();
    check(read == SIZE);
    check(read == f.nbytes());
    for (int i = 0; i < NELEM; ++i) {
      check(a[i] == b[i]);
    }
    cout << "Parallel POSIX read (" << kvikio::defaults::thread_pool_nthreads()
         << " threads): " << read << endl;
  }
}
