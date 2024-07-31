/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <kvikio/defaults.hpp>
#include <kvikio/driver.hpp>
#include <kvikio/remote_handle.hpp>
#include <kvikio/utils.hpp>

using namespace std;

void check(bool condition)
{
  if (!condition) {
    std::cout << "Error" << std::endl;
    exit(-1);
  }
}

constexpr int NELEM = 1024;                 // Number of elements used throughout the test
constexpr int SIZE  = NELEM * sizeof(int);  // Size of the memory allocations (in bytes)

int main()
{
  check(cudaSetDevice(0) == cudaSuccess);

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
    kvikio::RemoteHandle f("test-bucket", "a1");
    size_t read = f.read(a, SIZE);
    check(read == SIZE);
    cout << "Read:  " << read << endl;
    for (int i = 0; i < NELEM; ++i) {
      check(a[i] == i);
    }
    read = f.read(a, 10 * sizeof(int), 10 * sizeof(int));
    check(read == 10 * sizeof(int));
    cout << "Read[10:20]: ";
    for (int i = 0; i < 10; ++i) {
      cout << a[i] << ", ";
    }
    cout << endl;
  }
}
