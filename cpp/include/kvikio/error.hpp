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
#pragma once

#include <exception>
#include <system_error>

#include <cuda.h>

#include <cufile.h>

namespace kvikio {

struct CUfileException : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

#define KVIKIO_STRINGIFY_DETAIL(x) #x
#define KVIKIO_STRINGIFY(x)        KVIKIO_STRINGIFY_DETAIL(x)

#ifndef CUFILE_TRY
#define CUFILE_TRY(...)                                         \
  GET_CUFILE_TRY_MACRO(__VA_ARGS__, CUFILE_TRY_2, CUFILE_TRY_1) \
  (__VA_ARGS__)
#define GET_CUFILE_TRY_MACRO(_1, _2, NAME, ...) NAME
#define CUFILE_TRY_2(_call, _exception_type)                                                      \
  do {                                                                                            \
    CUfileError_t const error = (_call);                                                          \
    if (error.err != CU_FILE_SUCCESS) {                                                           \
      if (error.err == CU_FILE_CUDA_DRIVER_ERROR) {                                               \
        if (error.cu_err != CUDA_SUCCESS) {                                                       \
          const char* err_name;                                                                   \
          const char* err_str;                                                                    \
          CUresult err_name_status = cuGetErrorName(error.cu_err, &err_name);                     \
          CUresult err_str_status  = cuGetErrorString(error.cu_err, &err_str);                    \
          if (err_name_status == CUDA_ERROR_INVALID_VALUE) { err_name = "unknown"; }              \
          if (err_str_status == CUDA_ERROR_INVALID_VALUE) { err_str = "unknown"; }                \
          /*NOLINTNEXTLINE(bugprone-macro-parentheses)*/                                          \
          throw _exception_type{std::string{"CUDA error at: "} + __FILE__ + ":" +                 \
                                KVIKIO_STRINGIFY(__LINE__) + ": " + std::string(err_name) + "(" + \
                                std::string(err_str) + ")"};                                      \
        }                                                                                         \
      } else {                                                                                    \
        /*NOLINTNEXTLINE(bugprone-macro-parentheses)*/                                            \
        throw _exception_type{std::string{"cuFile error at: "} + __FILE__ + ":" +                 \
                              KVIKIO_STRINGIFY(__LINE__) + ": " +                                 \
                              cufileop_status_error(error.err)};                                  \
      }                                                                                           \
    }                                                                                             \
  } while (0)
#define CUFILE_TRY_1(_call) CUFILE_TRY_2(_call, CUfileException)
#endif

#ifndef CUDA_DRIVER_TRY
#define CUDA_DRIVER_TRY(...)                                                   \
  GET_CUDA_DRIVER_TRY_MACRO(__VA_ARGS__, CUDA_DRIVER_TRY_2, CUDA_DRIVER_TRY_1) \
  (__VA_ARGS__)
#define GET_CUDA_DRIVER_TRY_MACRO(_1, _2, NAME, ...) NAME
#define CUDA_DRIVER_TRY_2(_call, _exception_type)                                             \
  do {                                                                                        \
    CUresult const error = (_call);                                                           \
    if (error != CUDA_SUCCESS) {                                                              \
      const char* err_name;                                                                   \
      const char* err_str;                                                                    \
      CUresult err_name_status = cuGetErrorName(error, &err_name);                            \
      CUresult err_str_status  = cuGetErrorString(error, &err_str);                           \
      if (err_name_status == CUDA_ERROR_INVALID_VALUE) { err_name = "unknown"; }              \
      if (err_str_status == CUDA_ERROR_INVALID_VALUE) { err_str = "unknown"; }                \
      /*NOLINTNEXTLINE(bugprone-macro-parentheses)*/                                          \
      throw _exception_type{std::string{"CUDA error at: "} + __FILE__ + ":" +                 \
                            KVIKIO_STRINGIFY(__LINE__) + ": " + std::string(err_name) + "(" + \
                            std::string(err_str) + ")"};                                      \
    }                                                                                         \
  } while (0)
#define CUDA_DRIVER_TRY_1(_call) CUDA_DRIVER_TRY_2(_call, CUfileException)
#endif

}  // namespace kvikio
