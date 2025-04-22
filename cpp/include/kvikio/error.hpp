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
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>

#include <kvikio/shim/cuda.hpp>
#include <kvikio/shim/cufile_h_wrapper.hpp>

namespace kvikio {

struct CUfileException : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

class GenericSystemError : public std::system_error {
 public:
  GenericSystemError(const std::string& msg);
  GenericSystemError(const char* msg);
  GenericSystemError(const GenericSystemError& other)            = default;
  GenericSystemError& operator=(const GenericSystemError& other) = default;
  virtual ~GenericSystemError() noexcept                         = default;
};

#ifndef CUDA_DRIVER_TRY
/**
 * @addtogroup utility_error
 * @{
 */

/**
 * @brief Error checking macro for CUDA driver API functions.
 *
 * Invoke a CUDA driver API function call. If the call does not return CUDA_SUCCESS, throw an
 * exception detailing the CUDA error that occurred.
 *
 * Example:
 * ```
 * // Throws kvikio::CUfileException
 * CUDA_DRIVER_TRY(cudaAPI::instance().StreamSynchronize(_stream));
 *
 * // Throws std::runtime_error
 * CUDA_DRIVER_TRY(cudaAPI::instance().StreamSynchronize(_stream), std::runtime_error);
 * ```
 *
 * @param ... This macro accepts either one or two arguments:
 *   - The first argument must be a CUDA driver API error code.
 *   - When given, the second argument is the exception to be thrown. When not
 *     specified, defaults to kvikio::CUfileException.
 */
#define CUDA_DRIVER_TRY(...)                                                   \
  GET_CUDA_DRIVER_TRY_MACRO(__VA_ARGS__, CUDA_DRIVER_TRY_2, CUDA_DRIVER_TRY_1) \
  (__VA_ARGS__)
/** @} */

#define GET_CUDA_DRIVER_TRY_MACRO(_1, _2, NAME, ...) NAME
#define CUDA_DRIVER_TRY_2(_call, _exception_type)                                  \
  do {                                                                             \
    kvikio::detail::cuda_driver_try_2<_exception_type>(_call, __LINE__, __FILE__); \
  } while (0)
#define CUDA_DRIVER_TRY_1(_call) CUDA_DRIVER_TRY_2(_call, kvikio::CUfileException)
#endif

#ifndef CUFILE_TRY
/**
 * @addtogroup utility_error
 * @{
 */

/**
 * @brief Error checking macro for cuFile API functions.
 *
 * Invoke a cuFile API function call. If the call does not return CU_FILE_SUCCESS, throw an
 * exception detailing the cuFile error that occurred.
 *
 * Example:
 * ```
 * // Throws kvikio::CUfileException
 * CUFILE_TRY(cuFileAPI::instance().ReadAsync(...));
 *
 * // Throws std::runtime_error
 * CUFILE_TRY(cuFileAPI::instance().ReadAsync(...), std::runtime_error);
 * ```
 *
 * @param ... This macro accepts either one or two arguments:
 *   - The first argument must be a cuFile API error code.
 *   - When given, the second argument is the exception to be thrown. When not
 *     specified, defaults to kvikio::CUfileException.
 */
#define CUFILE_TRY(...)                                         \
  GET_CUFILE_TRY_MACRO(__VA_ARGS__, CUFILE_TRY_2, CUFILE_TRY_1) \
  (__VA_ARGS__)
/** @} */

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

/**
 * @addtogroup utility_error
 * @{
 */

/**
 * @brief Macro for checking pre-conditions or conditions that throws an exception when
 * a condition is violated.
 *
 * Defaults to throwing kvikio::CUfileException, but a custom exception may also be
 * specified.
 *
 * Example:
 * ```
 * // Throws kvikio::CUfileException
 * KVIKIO_EXPECT(p != nullptr, "Unexpected null pointer");
 *
 * // Throws std::runtime_error
 * KVIKIO_EXPECT(p != nullptr, "Unexpected nullptr", std::runtime_error);
 * ```
 *
 * @param ... This macro accepts either two or three arguments:
 *   - The first argument must be an expression that evaluates to true or
 *     false, and is the condition being checked.
 *   - The second argument is a string literal used to construct the `what` of
 *     the exception.
 *   - When given, the third argument is the exception to be thrown. When not
 *     specified, defaults to kvikio::CUfileException.
 */
#define KVIKIO_EXPECT(...) \
  GET_KVIKIO_EXPECT_MACRO(__VA_ARGS__, KVIKIO_EXPECT_3, KVIKIO_EXPECT_2)(__VA_ARGS__)
/** @} */

#define GET_KVIKIO_EXPECT_MACRO(_1, _2, _3, NAME, ...) NAME

#define KVIKIO_EXPECT_3(_condition, _msg, _exception_type)                                   \
  do {                                                                                       \
    kvikio::detail::kvikio_assertion<_exception_type>(_condition, _msg, __LINE__, __FILE__); \
  } while (0)

#define KVIKIO_EXPECT_2(_condition, _msg) KVIKIO_EXPECT_3(_condition, _msg, kvikio::CUfileException)

/**
 * @addtogroup utility_error
 * @{
 */

/**
 * @brief Indicates that an erroneous code path has been taken.
 *
 * Example usage:
 * ```
 * // Throws kvikio::CUfileException
 * KVIKIO_FAIL("Unsupported code path");
 *
 * // Throws std::runtime_error
 * KVIKIO_FAIL("Unsupported code path", std::runtime_error);
 * ```
 *
 * @param ... This macro accepts either one or two arguments:
 *   - The first argument is a string literal used to construct the `what` of
 *     the exception.
 *   - When given, the second argument is the exception to be thrown. When not
 *     specified, defaults to kvikio::CUfileException.
 */
#define KVIKIO_FAIL(...) \
  GET_KVIKIO_FAIL_MACRO(__VA_ARGS__, KVIKIO_FAIL_2, KVIKIO_FAIL_1)(__VA_ARGS__)
/** @} */

#define GET_KVIKIO_FAIL_MACRO(_1, _2, NAME, ...) NAME

#define KVIKIO_FAIL_2(_msg, _exception_type)                                            \
  do {                                                                                  \
    kvikio::detail::kvikio_assertion<_exception_type>(false, _msg, __LINE__, __FILE__); \
  } while (0)

#define KVIKIO_FAIL_1(_msg) KVIKIO_FAIL_2(_msg, kvikio::CUfileException)

namespace detail {
template <typename Exception>
void kvikio_assertion(bool condition, const char* msg, int line_number, char const* filename)
{
  if (!condition) {
    std::stringstream ss;
    ss << "KvikIO failure at: " << filename << ":" << line_number << ": ";
    if (msg == nullptr) {
      ss << "(no message)";
    } else {
      ss << msg;
    }
    throw Exception{ss.str()};
  };
}

template <typename Exception>
void kvikio_assertion(bool condition, const std::string& msg, int line_number, char const* filename)
{
  kvikio_assertion<Exception>(condition, msg.c_str(), line_number, filename);
}
}  // namespace detail

/**
 * @addtogroup utility_error
 * @{
 */

/**
 * @brief Error checking macro for Linux system call.
 *
 * Error checking for a Linux system call typically involves:
 * - Check the return value of the system call. A value of -1 indicates failure for the overwhelming
 *   majority of system calls.
 * - If failure, check the global error number <a
 *   href="https://man7.org/linux/man-pages/man3/errno.3.html" target="_blank">errno</a>. Use
 *   Linux utility functions to obtain detailed error information.
 *
 * This macro is used to perform the steps above. A simple SYSCALL_CHECK(ret) is
 * designed for the common cases where an integer of -1 indicates a failure, whereas a more complex
 * SYSCALL_CHECK(ret, "extra msg", error_value) for the remaining rare cases, such
 * as `(void*)-1` for mmap. At any rate, if a failure occurs, this macro throws an exception
 * (kvikio::GenericSystemError) with detailed error information.
 *
 * Example:
 * ```
 * // Common case: (int)-1 indicates an error.
 * SYSCALL_CHECK(open(file_name, flags, mode));
 *
 * // Rare case: (void*)-1 indicates an error.
 * SYSCALL_CHECK(mmap(addr, length,prot, flags, fd, offset), "mmap failed",
 * reinterpret_cast<void*>(-1));
 * ```
 *
 * @param ... This macro accepts the following arguments:
 *   - The first argument must be the return value of a Linux system call.
 *   - When given, the second argument is the extra message for the exception. When not specified,
 *     defaults to empty.
 *   - When given, the third argument is the error code value used to indicate an error. When not
 *     specified, defaults to -1.
 */
#define SYSCALL_CHECK(...)                                                                \
  GET_SYSCALL_CHECK_MACRO(__VA_ARGS__, SYSCALL_CHECK_3, SYSCALL_CHECK_2, SYSCALL_CHECK_1) \
  (__VA_ARGS__)
/** @} */

#define GET_SYSCALL_CHECK_MACRO(_1, _2, _3, NAME, ...) NAME
#define SYSCALL_CHECK_1(_return_value)                                   \
  do {                                                                   \
    kvikio::detail::check_linux_call(__LINE__, __FILE__, _return_value); \
  } while (0)
#define SYSCALL_CHECK_2(_return_value, _extra_msg)                                   \
  do {                                                                               \
    kvikio::detail::check_linux_call(__LINE__, __FILE__, _return_value, _extra_msg); \
  } while (0)
#define SYSCALL_CHECK_3(_return_value, _extra_msg, _error_value)                                   \
  do {                                                                                             \
    kvikio::detail::check_linux_call(__LINE__, __FILE__, _return_value, _extra_msg, _error_value); \
  } while (0)

namespace detail {
void handle_linux_call_error(int line_number, char const* filename, std::string_view extra_msg);

inline void check_linux_call(int line_number,
                             char const* filename,
                             int return_value,
                             std::string_view extra_msg = "",
                             int error_value            = -1)
{
  if (return_value == error_value) { handle_linux_call_error(line_number, filename, extra_msg); }
}

template <typename T>
void check_linux_call(
  int line_number, char const* filename, T return_value, std::string_view extra_msg, T error_value)
{
  if (return_value == error_value) { handle_linux_call_error(line_number, filename, extra_msg); }
}
}  // namespace detail

}  // namespace kvikio
