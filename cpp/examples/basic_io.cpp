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

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <cufile/buffer.hpp>
#include <cufile/driver.hpp>
#include <cufile/file_handle.hpp>
#include <cufile/nvml.hpp>

using namespace std;

void check(bool condition)
{
  if (!condition) {
    std::cout << "Error" << std::endl;
    exit(-1);
  }
}

int main()
{
  check(cudaSetDevice(0) == cudaSuccess);
  cufile::DriverInitializer manual_init_driver;
  cufile::DriverProperties props;
  cout << "DriverProperties: " << endl;
  cout << "  Version: " << props.get_nvfs_major_version() << "." << props.get_nvfs_minor_version()
       << endl;
  cout << "  Allow compatibility mode: " << std::boolalpha << props.get_nvfs_allow_compat_mode()
       << endl;
  cout << "  Pool mode - enabled: " << std::boolalpha << props.get_nvfs_poll_mode()
       << ", threshold: " << props.get_nvfs_poll_thresh_size() << " kb" << endl;
  cout << "  Max pinned memory: " << props.get_max_pinned_memory_size() << " kb" << endl;
  cufile::NVML nvml;
  auto [mem_total, mem_free]   = nvml.get_memory();
  auto [bar1_total, bar1_free] = nvml.get_bar1_memory();
  cout << "nvml: " << endl;
  cout << "  GPU Memory Total : " << mem_total << " bytes" << endl;
  cout << "  BAR1 Memory Total: " << bar1_total << " bytes" << endl;

  int a[1024], b[1024];
  for (int i = 0; i < 1024; ++i) {
    a[i] = i;
  }
  void* a_dev = nullptr;
  void* b_dev = nullptr;
  void* c_dev = nullptr;
  check(cudaMalloc(&a_dev, sizeof(a)) == cudaSuccess);
  check(cudaMalloc(&b_dev, sizeof(a)) == cudaSuccess);
  check(cudaMalloc(&c_dev, sizeof(a)) == cudaSuccess);

  {
    cufile::FileHandle f("test-file", "w");
    check(cudaMemcpy(a_dev, &a, sizeof(a), cudaMemcpyHostToDevice) == cudaSuccess);
    size_t written = f.pwrite(a_dev, sizeof(a));
    check(written == sizeof(a));
    cout << "Write: " << written << endl;
  }
  {
    cufile::FileHandle f("test-file", "r");
    size_t read = f.pread(b_dev, sizeof(a));
    check(read == sizeof(a));
    cout << "Read:  " << read << endl;
    check(cudaMemcpy(&b, b_dev, sizeof(a), cudaMemcpyDeviceToHost) == cudaSuccess);
    for (int i = 0; i < 1024; ++i) {
      check(a[i] == b[i]);
    }
  }
  cufile::default_thread_pool::reset(16);
  {
    cufile::FileHandle f("test-file", "w");
    size_t written = f.pwrite(a_dev, sizeof(a));
    check(written == sizeof(a));
    cout << "Parallel write (" << cufile::default_thread_pool::nthreads()
         << " threads): " << written << endl;
  }
  {
    cufile::FileHandle f("test-file", "r");
    size_t read = f.pread(b_dev, sizeof(a), 0);
    cout << "Parallel write (" << cufile::default_thread_pool::nthreads() << " threads): " << read
         << endl;
    check(cudaMemcpy(&b, b_dev, sizeof(a), cudaMemcpyDeviceToHost) == cudaSuccess);
    for (int i = 0; i < 1024; ++i) {
      check(a[i] == b[i]);
    }
  }
  {
    cufile::FileHandle f("test-file", "r+", cufile::FileHandle::m644);
    cufile::buffer_register(c_dev, size(a));
    f.pwrite(c_dev, size(a), 0);
    cufile::buffer_deregister(c_dev);
  }
}
