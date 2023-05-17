/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

#include <kvikio/batch.hpp>
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

constexpr int NELEM = 1000;                 // Number of elements used throughout the test
constexpr int SIZE  = NELEM * sizeof(int);  // Size of the memory allocations (in bytes)

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
    cout << "  nvfs version: " << props.get_nvfs_major_version() << "."
         << props.get_nvfs_minor_version() << endl;
    cout << "  Allow compatibility mode: " << std::boolalpha << props.get_nvfs_allow_compat_mode()
         << endl;
    cout << "  Pool mode - enabled: " << std::boolalpha << props.get_nvfs_poll_mode()
         << ", threshold: " << props.get_nvfs_poll_thresh_size() << " kb" << endl;
    cout << "  Max pinned memory: " << props.get_max_pinned_memory_size() << " kb" << endl;
    cout << "  Max batch IO size: " << props.get_max_batch_io_size() << endl;
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

  if (kvikio::is_batch_available()) {
    // Here we use the batch API to read "/tmp/test-file" into `b_dev` by
    // submitting 4 batch operations.
    constexpr int num_of_batches = 4;
    constexpr int batchsize      = SIZE / num_of_batches;
    kvikio::DriverProperties props;
    check(num_of_batches < props.get_max_batch_io_size());

    // We open the file as usual.
    kvikio::FileHandle f("/tmp/test-file", "r");

    // Then we create a batch
    auto batch = kvikio::BatchHandle(num_of_batches);

    // And submit 4 operations each with its own offset
    std::vector<kvikio::BatchOp> ops;
    for (int i = 0; i < num_of_batches; ++i) {
      ops.push_back(kvikio::BatchOp{.file_handle   = f,
                                    .devPtr_base   = b_dev,
                                    .file_offset   = i * batchsize,
                                    .devPtr_offset = i * batchsize,
                                    .size          = batchsize,
                                    .opcode        = CUFILE_READ});
    }
    batch.submit(ops);

    // Finally, we wait on all 4 operations to be finished and check the result
    auto statuses = batch.status(num_of_batches, num_of_batches);
    check(statuses.size() == num_of_batches);
    size_t total_read = 0;
    for (auto status : statuses) {
      check(status.status == CUFILE_COMPLETE);
      check(status.ret == batchsize);
      total_read += status.ret;
    }
    check(cudaMemcpy(b, b_dev, SIZE, cudaMemcpyDeviceToHost) == cudaSuccess);
    for (int i = 0; i < NELEM; ++i) {
      check(a[i] == b[i]);
    }
    cout << "Batch read using 4 operations: " << total_read << endl;
  }
}
