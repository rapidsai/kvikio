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
#pragma once

#include <filesystem>
#include <iostream>
#include <string>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <kvikio/error.hpp>

namespace kvikio::test {

/**                                                                                  \
 * @brief Error checking macro for CUDA runtime API functions.                       \
 *                                                                                   \
 * Invokes a CUDA runtime API function call. If the call does not return             \
 * `cudaSuccess`, invokes cudaGetLastError() to clear the error and throws an        \
 * exception detailing the CUDA error that occurred                                  \
 *                                                                                   \
 * Defaults to throwing std::runtime_error, but a custom exception may also be       \
 * specified.                                                                        \
 *                                                                                   \
 * Example:                                                                          \
 * ```c++                                                                            \
 *                                                                                   \
 * // Throws rmm::cuda_error if `cudaMalloc` fails                                   \
 * KVIKIO_CHECK_CUDA(cudaMalloc(&p, 100));                                           \
 *                                                                                   \
 * // Throws std::runtime_error if `cudaMalloc` fails                                \
 * KVIKIO_CHECK_CUDA(cudaMalloc(&p, 100), std::runtime_error);                       \
 * ```                                                                               \
 *                                                                                   \
 */                                                                                  \
#define KVIKIO_CHECK_CUDA(...)                                                       \
  GET_KVIKIO_CHECK_CUDA_MACRO(__VA_ARGS__, KVIKIO_CHECK_CUDA_2, KVIKIO_CHECK_CUDA_1) \
  (__VA_ARGS__)
#define GET_KVIKIO_CHECK_CUDA_MACRO(_1, _2, NAME, ...) NAME
#define KVIKIO_CHECK_CUDA_2(_call, _exception_type)                                             \
  do {                                                                                          \
    cudaError_t const error = (_call);                                                          \
    if (cudaSuccess != error) {                                                                 \
      cudaGetLastError();                                                                       \
      /*NOLINTNEXTLINE(bugprone-macro-parentheses)*/                                            \
      throw _exception_type{std::string{"CUDA error at: "} + __FILE__ + ":" +                   \
                            KVIKIO_STRINGIFY(__LINE__) + ": " + cudaGetErrorName(error) + " " + \
                            cudaGetErrorString(error)};                                         \
    }                                                                                           \
  } while (0)
#define KVIKIO_CHECK_CUDA_1(_call) KVIKIO_CHECK_CUDA_2(_call, std::runtime_error)

class TempDir {
 public:
  TempDir(const bool cleanup = true) : _cleanup{cleanup}
  {
    std::string tpl{std::filesystem::temp_directory_path() / "legate-dataframe.XXXXXX"};
    if (mkdtemp(tpl.data()) == nullptr) {
      throw std::runtime_error("TempDir: cannot make tmpdir: " + tpl);
    }
    _dir_path = tpl;
  }
  ~TempDir() noexcept
  {
    if (_cleanup) {
      try {
        std::filesystem::remove_all(_dir_path);
      } catch (...) {
        std::cout << "error while trying to remove " << _dir_path.string() << std::endl;
      }
    }
  }

  TempDir(const TempDir&)              = delete;
  TempDir& operator=(TempDir const&)   = delete;
  TempDir(const TempDir&&)             = delete;
  TempDir&& operator=(TempDir const&&) = delete;

  const std::filesystem::path& path() { return _dir_path; }

  operator std::string() { return path(); }

 private:
  const bool _cleanup;
  std::filesystem::path _dir_path{};
};

class DevBuffer {
 public:
  DevBuffer(std::size_t nbytes = 1024) : nbytes{nbytes}
  {
    KVIKIO_CHECK_CUDA(cudaMalloc(&ptr, nbytes));
  }

  ~DevBuffer() noexcept { cudaFree(ptr); }

  const std::size_t nbytes;
  void* ptr{nullptr};
};

}  // namespace kvikio::test
