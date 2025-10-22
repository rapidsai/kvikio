/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

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
 * // Throws std::runtime_error if `cudaMalloc` fails                                \
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

/**
 * @brief Help class to create a temporary directory.
 */
class TempDir {
 public:
  TempDir(bool const cleanup = true) : _cleanup{cleanup}
  {
    std::string tpl{std::filesystem::temp_directory_path() / "kvikio.XXXXXX"};
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

  TempDir(TempDir const&)              = delete;
  TempDir& operator=(TempDir const&)   = delete;
  TempDir(TempDir const&&)             = delete;
  TempDir&& operator=(TempDir const&&) = delete;

  std::filesystem::path const& path() { return _dir_path; }

  operator std::string() { return path(); }

 private:
  bool const _cleanup;
  std::filesystem::path _dir_path{};
};

/**
 * @brief Help class for creating and comparing buffers.
 */
template <typename T>
class DevBuffer {
 public:
  std::size_t nelem;
  std::size_t nbytes;
  void* ptr{nullptr};

  DevBuffer() : nelem{0}, nbytes{0} {};

  DevBuffer(std::size_t nelem) : nelem{nelem}, nbytes{nelem * sizeof(T)}
  {
    KVIKIO_CHECK_CUDA(cudaMalloc(&ptr, nbytes));
    KVIKIO_CHECK_CUDA(cudaMemset(ptr, 0, nbytes));
  }
  DevBuffer(std::vector<T> const& host_buffer) : DevBuffer{host_buffer.size()}
  {
    KVIKIO_CHECK_CUDA(cudaMemcpy(ptr, host_buffer.data(), nbytes, cudaMemcpyHostToDevice));
  }

  DevBuffer(DevBuffer&& dev_buffer) noexcept
    : nelem{std::exchange(dev_buffer.nelem, 0)},
      nbytes{std::exchange(dev_buffer.nbytes, 0)},
      ptr{std::exchange(dev_buffer.ptr, nullptr)}
  {
  }

  DevBuffer& operator=(DevBuffer&& dev_buffer) noexcept
  {
    nelem  = std::exchange(dev_buffer.nelem, 0);
    nbytes = std::exchange(dev_buffer.nbytes, 0);
    ptr    = std::exchange(dev_buffer.ptr, nullptr);
    return *this;
  }

  ~DevBuffer() noexcept { cudaFree(ptr); }

  [[nodiscard]] static DevBuffer arange(std::size_t nelem, T start = 0)
  {
    std::vector<T> host_buffer(nelem);
    std::iota(host_buffer.begin(), host_buffer.end(), start);
    return DevBuffer{host_buffer};
  }

  [[nodiscard]] static DevBuffer zero_like(DevBuffer const& prototype)
  {
    DevBuffer ret{prototype.nelem};
    KVIKIO_CHECK_CUDA(cudaMemset(ret.ptr, 0, ret.nbytes));
    return ret;
  }

  [[nodiscard]] std::vector<T> to_vector() const
  {
    std::vector<T> ret(nelem);
    KVIKIO_CHECK_CUDA(cudaMemcpy(ret.data(), this->ptr, nbytes, cudaMemcpyDeviceToHost));
    return ret;
  }

  void pprint() const
  {
    std::cout << "DevBuffer(";
    for (auto item : to_vector()) {
      std::cout << static_cast<int64_t>(item) << ", ";
    }
    std::cout << ")" << std::endl;
  }
};

/**
 * @brief Check that two buffers are equal
 */
template <typename T>
inline void expect_equal(DevBuffer<T> const& a, DevBuffer<T> const& b)
{
  EXPECT_EQ(a.nbytes, b.nbytes);
  auto a_vec = a.to_vector();
  auto b_vec = b.to_vector();
  for (std::size_t i = 0; i < a.nelem; ++i) {
    EXPECT_EQ(a_vec[i], b_vec[i]) << "Mismatch at index " << i;
  }
}

}  // namespace kvikio::test
