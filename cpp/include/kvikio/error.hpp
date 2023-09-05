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
#pragma once

#include <exception>
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
#define CUDA_DRIVER_TRY_2(_call, _exception_type)                                              \
  do {                                                                                         \
    CUresult const error = (_call);                                                            \
    if (error == CUDA_ERROR_STUB_LIBRARY) {                                                    \
      throw(_exception_type){std::string{"CUDA error at: "} + __FILE__ + ":" +                 \
                             KVIKIO_STRINGIFY(__LINE__) +                                      \
                             ": CUDA_ERROR_STUB_LIBRARY("                                      \
                             "The CUDA driver loaded is a stub library)"};                     \
    }                                                                                          \
    if (error != CUDA_SUCCESS) {                                                               \
      const char* err_name     = nullptr;                                                      \
      const char* err_str      = nullptr;                                                      \
      CUresult err_name_status = cudaAPI::instance().GetErrorName(error, &err_name);           \
      CUresult err_str_status  = cudaAPI::instance().GetErrorString(error, &err_str);          \
      if (err_name_status == CUDA_ERROR_INVALID_VALUE) { err_name = "unknown"; }               \
      if (err_str_status == CUDA_ERROR_INVALID_VALUE) { err_str = "unknown"; }                 \
      throw(_exception_type){std::string{"CUDA error at: "} + __FILE__ + ":" +                 \
                             KVIKIO_STRINGIFY(__LINE__) + ": " + std::string(err_name) + "(" + \
                             std::string(err_str) + ")"};                                      \
    }                                                                                          \
  } while (0)
#define CUDA_DRIVER_TRY_1(_call) CUDA_DRIVER_TRY_2(_call, kvikio::CUfileException)
#endif

#ifdef KVIKIO_CUFILE_FOUND
#ifndef CUFILE_TRY
#define CUFILE_TRY(...)                                         \
  GET_CUFILE_TRY_MACRO(__VA_ARGS__, CUFILE_TRY_2, CUFILE_TRY_1) \
  (__VA_ARGS__)
#define GET_CUFILE_TRY_MACRO(_1, _2, NAME, ...) NAME
#define CUFILE_TRY_2(_call, _exception_type)                                     \
  do {                                                                           \
    CUfileError_t const error = (_call);                                         \
    if (error.err != CU_FILE_SUCCESS) {                                          \
      if (error.err == CU_FILE_CUDA_DRIVER_ERROR) {                              \
        CUresult const cuda_error = error.cu_err;                                \
        CUDA_DRIVER_TRY(cuda_error);                                             \
      }                                                                          \
      throw(_exception_type){std::string{"cuFile error at: "} + __FILE__ + ":" + \
                             KVIKIO_STRINGIFY(__LINE__) + ": " +                 \
                             cufileop_status_error(error.err)};                  \
    }                                                                            \
  } while (0)
#define CUFILE_TRY_1(_call) CUFILE_TRY_2(_call, kvikio::CUfileException)
#endif
#endif

#ifndef CUFILE_CHECK_STREAM_IO
#define CUFILE_CHECK_STREAM_IO(...)                                  \
  GET_CUFILE_CHECK_STREAM_IO_MACRO(                                  \
    __VA_ARGS__, CUFILE_CHECK_STREAM_IO_2, CUFILE_CHECK_STREAM_IO_1) \
  (__VA_ARGS__)
#define GET_CUFILE_CHECK_STREAM_IO_MACRO(_1, _2, NAME, ...) NAME
#ifdef KVIKIO_CUFILE_FOUND
#define CUFILE_CHECK_STREAM_IO_2(_nbytes_done, _exception_type)                            \
  do {                                                                                     \
    auto const _nbytes = *(_nbytes_done);                                                  \
    if (_nbytes < 0) {                                                                     \
      throw(_exception_type){std::string{"cuFile error at: "} + __FILE__ + ":" +           \
                             KVIKIO_STRINGIFY(__LINE__) + ": " + std::to_string(_nbytes)}; \
    }                                                                                      \
  } while (0)
#else
// if cufile isn't available, we don't do anything in the body
#define CUFILE_CHECK_STREAM_IO_2(_nbytes_done, _exception_type) \
  do {                                                          \
  } while (0)
#endif
#define CUFILE_CHECK_STREAM_IO_1(_call) CUFILE_CHECK_STREAM_IO_2(_call, kvikio::CUfileException)
#endif

}  // namespace kvikio
