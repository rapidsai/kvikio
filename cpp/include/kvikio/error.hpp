/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

#include <cstring>
#include <exception>
#include <string>
#include <system_error>

#include <kvikio/shim/cuda.hpp>
#include <kvikio/shim/cufile_h_wrapper.hpp>

namespace kvikio {

struct CUfileException : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

#ifndef CUDA_DRIVER_TRY
#define CUDA_DRIVER_TRY(...)                                                   \
  GET_CUDA_DRIVER_TRY_MACRO(__VA_ARGS__, CUDA_DRIVER_TRY_2, CUDA_DRIVER_TRY_1) \
  (__VA_ARGS__)
#define GET_CUDA_DRIVER_TRY_MACRO(_1, _2, NAME, ...) NAME
#define CUDA_DRIVER_TRY_2(_call, _exception_type)                                  \
  do {                                                                             \
    kvikio::detail::cuda_driver_try_2<_exception_type>(_call, __LINE__, __FILE__); \
  } while (0)
#define CUDA_DRIVER_TRY_1(_call) CUDA_DRIVER_TRY_2(_call, kvikio::CUfileException)
#endif

#ifndef CUFILE_TRY
#define CUFILE_TRY(...)                                         \
  GET_CUFILE_TRY_MACRO(__VA_ARGS__, CUFILE_TRY_2, CUFILE_TRY_1) \
  (__VA_ARGS__)
#define GET_CUFILE_TRY_MACRO(_1, _2, NAME, ...) NAME
#define CUFILE_TRY_2(_call, _exception_type)                                  \
  do {                                                                        \
    kvikio::detail::cufile_try_2<_exception_type>(_call, __LINE__, __FILE__); \
  } while (0)
#define CUFILE_TRY_1(_call) CUFILE_TRY_2(_call, kvikio::CUfileException)
#endif

#ifndef CUFILE_CHECK_BYTES_DONE
#define CUFILE_CHECK_BYTES_DONE(...)                                   \
  GET_CUFILE_CHECK_BYTES_DONE_MACRO(                                   \
    __VA_ARGS__, CUFILE_CHECK_BYTES_DONE_2, CUFILE_CHECK_BYTES_DONE_1) \
  (__VA_ARGS__)
#define GET_CUFILE_CHECK_BYTES_DONE_MACRO(_1, _2, NAME, ...) NAME
#define CUFILE_CHECK_BYTES_DONE_2(_nbytes_done, _exception_type)                                  \
  do {                                                                                            \
    kvikio::detail::cufile_check_bytes_done_2<_exception_type>(_nbytes_done, __LINE__, __FILE__); \
  } while (0)
#define CUFILE_CHECK_BYTES_DONE_1(_call) CUFILE_CHECK_BYTES_DONE_2(_call, kvikio::CUfileException)
#endif

namespace detail {
template <typename Exception>
void cuda_driver_try_2(CUresult error, int line_number, char const* filename)
{
  if (error == CUDA_ERROR_STUB_LIBRARY) {
    throw Exception{std::string{"CUDA error at: "} + std::string(filename) + ":" +
                    std::to_string(line_number) +
                    ": CUDA_ERROR_STUB_LIBRARY("
                    "The CUDA driver loaded is a stub library)"};
  }
  if (error != CUDA_SUCCESS) {
    char const* err_name     = nullptr;
    char const* err_str      = nullptr;
    CUresult err_name_status = cudaAPI::instance().GetErrorName(error, &err_name);
    CUresult err_str_status  = cudaAPI::instance().GetErrorString(error, &err_str);
    if (err_name_status == CUDA_ERROR_INVALID_VALUE) { err_name = "unknown"; }
    if (err_str_status == CUDA_ERROR_INVALID_VALUE) { err_str = "unknown"; }
    throw Exception{std::string{"CUDA error at: "} + filename + ":" + std::to_string(line_number) +
                    ": " + std::string(err_name) + "(" + std::string(err_str) + ")"};
  }
}

template <typename Exception>
void cufile_try_2(CUfileError_t error, int line_number, char const* filename)
{
  if (error.err != CU_FILE_SUCCESS) {
    if (error.err == CU_FILE_CUDA_DRIVER_ERROR) {
      CUresult const cuda_error = error.cu_err;
      CUDA_DRIVER_TRY(cuda_error);
    }
    throw Exception{std::string{"cuFile error at: "} + filename + ":" +
                    std::to_string(line_number) + ": " +
                    cufileop_status_error((CUfileOpError)std::abs(error.err))};
  }
}

template <typename Exception>
void cufile_check_bytes_done_2(ssize_t nbytes_done, int line_number, char const* filename)
{
  if (nbytes_done < 0) {
    auto const err = std::abs(nbytes_done);
    auto const msg = (err > CUFILEOP_BASE_ERR)
                       ? std::string(cufileop_status_error((CUfileOpError)err))
                       : std::string(std::strerror(err));
    throw Exception{std::string{"cuFile error at: "} + filename + ":" +
                    std::to_string(line_number) + ": " + msg};
  }
}

#define KVIKIO_LOG_ERROR(err_msg) kvikio::detail::log_error(err_msg, __LINE__, __FILE__)
void log_error(std::string_view err_msg, int line_number, char const* filename);

}  // namespace detail

}  // namespace kvikio
