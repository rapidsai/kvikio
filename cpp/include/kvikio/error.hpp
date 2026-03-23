/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
  GenericSystemError(int err_code, std::string const& msg);
  GenericSystemError(int err_code, char const* msg);
  GenericSystemError(std::string const& msg);
  GenericSystemError(char const* msg);
  GenericSystemError(GenericSystemError const& other)            = default;
  GenericSystemError& operator=(GenericSystemError const& other) = default;
  virtual ~GenericSystemError() noexcept                         = default;
};

#define KVIKIO_VA_SELECT_3(_1, _2, _3, NAME, ...) NAME
#define KVIKIO_VA_SELECT_2(_1, _2, NAME, ...)     NAME

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
#define CUDA_DRIVER_TRY(...)                                            \
  KVIKIO_VA_SELECT_2(__VA_ARGS__, CUDA_DRIVER_TRY_2, CUDA_DRIVER_TRY_1) \
  (__VA_ARGS__)
/** @} */

#define CUDA_DRIVER_TRY_2(_call, _exception_type)                                \
  do {                                                                           \
    kvikio::detail::cuda_driver_try<_exception_type>(_call, __LINE__, __FILE__); \
  } while (0)
#define CUDA_DRIVER_TRY_1(_call) CUDA_DRIVER_TRY_2(_call, kvikio::CUfileException)

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
#define CUFILE_TRY(...)                                       \
  KVIKIO_VA_SELECT_2(__VA_ARGS__, CUFILE_TRY_2, CUFILE_TRY_1) \
  (__VA_ARGS__)
/** @} */

#define CUFILE_TRY_2(_call, _exception_type)                                \
  do {                                                                      \
    kvikio::detail::cufile_try<_exception_type>(_call, __LINE__, __FILE__); \
  } while (0)
#define CUFILE_TRY_1(_call) CUFILE_TRY_2(_call, kvikio::CUfileException)

#define CUFILE_CHECK_BYTES_DONE(...)                                                    \
  KVIKIO_VA_SELECT_2(__VA_ARGS__, CUFILE_CHECK_BYTES_DONE_2, CUFILE_CHECK_BYTES_DONE_1) \
  (__VA_ARGS__)
#define CUFILE_CHECK_BYTES_DONE_2(_nbytes_done, _exception_type)                                \
  do {                                                                                          \
    kvikio::detail::cufile_check_bytes_done<_exception_type>(_nbytes_done, __LINE__, __FILE__); \
  } while (0)
#define CUFILE_CHECK_BYTES_DONE_1(_call) CUFILE_CHECK_BYTES_DONE_2(_call, kvikio::CUfileException)

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
  KVIKIO_VA_SELECT_3(__VA_ARGS__, KVIKIO_EXPECT_3, KVIKIO_EXPECT_2)(__VA_ARGS__)
/** @} */

#define KVIKIO_EXPECT_3(_condition, _msg, _exception_type)          \
  do {                                                              \
    if (!(_condition)) {                                            \
      kvikio::detail::kvikio_fail<_exception_type>(                 \
        [&]() -> std::string { return _msg; }, __LINE__, __FILE__); \
    }                                                               \
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
#define KVIKIO_FAIL(...) KVIKIO_VA_SELECT_2(__VA_ARGS__, KVIKIO_FAIL_2, KVIKIO_FAIL_1)(__VA_ARGS__)
/** @} */

#define KVIKIO_FAIL_2(_msg, _exception_type)                      \
  do {                                                            \
    kvikio::detail::kvikio_fail<_exception_type>(                 \
      [&]() -> std::string { return _msg; }, __LINE__, __FILE__); \
  } while (0)

#define KVIKIO_FAIL_1(_msg) KVIKIO_FAIL_2(_msg, kvikio::CUfileException)

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
 * // Common case: -1 indicates an error.
 * SYSCALL_CHECK(open(file_name, flags, mode));
 *
 * // Rare case: (void*)-1 indicates an error.
 * SYSCALL_CHECK(mmap(addr, length, prot, flags, fd, offset), "mmap failed", MAP_FAILED);
 * ```
 *
 * @param ... This macro accepts the following arguments:
 *   - The first argument must be the return value of a Linux system call.
 *   - When given, the second argument is the extra message for the exception. When not specified,
 *     defaults to empty.
 *   - When given, the third argument is the error code value used to indicate an error. When not
 *     specified, defaults to -1.
 */
#define SYSCALL_CHECK(...)                                                           \
  KVIKIO_VA_SELECT_3(__VA_ARGS__, SYSCALL_CHECK_3, SYSCALL_CHECK_2, SYSCALL_CHECK_1) \
  (__VA_ARGS__)
/** @} */

#define SYSCALL_CHECK_1(_return_value)                                   \
  do {                                                                   \
    kvikio::detail::check_linux_call(_return_value, __LINE__, __FILE__); \
  } while (0)
#define SYSCALL_CHECK_2(_return_value, _extra_msg)                                   \
  do {                                                                               \
    kvikio::detail::check_linux_call(_return_value, __LINE__, __FILE__, _extra_msg); \
  } while (0)
#define SYSCALL_CHECK_3(_return_value, _extra_msg, _error_value)                                   \
  do {                                                                                             \
    kvikio::detail::check_linux_call(_return_value, __LINE__, __FILE__, _extra_msg, _error_value); \
  } while (0)

namespace detail {
/**
 * @brief Throw an exception with a formatted error message including source location.
 *
 * @tparam Exception The exception type to throw.
 * @tparam MsgFunc A callable type that returns a std::string error message.
 * @param msg_func Callable that produces the error message string.
 * @param line_number Source line number (typically from __LINE__).
 * @param filename Source file name (typically from __FILE__).
 * @exception Exception Always thrown with a message containing the source location and user
 * message.
 */
template <typename Exception, typename MsgFunc>
[[noreturn]] void kvikio_fail(MsgFunc&& msg_func, int line_number, char const* filename)
{
  std::string const msg = std::forward<MsgFunc>(msg_func)();
  throw Exception{std::string{"KvikIO failure at: "} + filename + ":" +
                  std::to_string(line_number) + ": " + (msg.empty() ? "(no message)" : msg)};
}

/**
 * @brief Check a CUDA driver API return code and throw on failure.
 *
 * @tparam Exception The exception type to throw.
 * @param error The CUresult return code from a CUDA driver API call.
 * @param line_number Source line number (typically from __LINE__).
 * @param filename Source file name (typically from __FILE__).
 * @exception Exception Thrown if @p error is not CUDA_SUCCESS.
 */
template <typename Exception>
void cuda_driver_try(CUresult error, int line_number, char const* filename)
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

/**
 * @brief Check a cuFile API return code and throw on failure.
 *
 * If the error indicates an underlying CUDA driver error, delegates to cuda_driver_try().
 *
 * @tparam Exception The exception type to throw.
 * @param error The CUfileError_t return code from a cuFile API call.
 * @param line_number Source line number (typically from __LINE__).
 * @param filename Source file name (typically from __FILE__).
 * @exception Exception Thrown if @p error does not indicate CU_FILE_SUCCESS.
 */
template <typename Exception>
void cufile_try(CUfileError_t error, int line_number, char const* filename)
{
  if (error.err != CU_FILE_SUCCESS) {
    if (error.err == CU_FILE_CUDA_DRIVER_ERROR) {
      cuda_driver_try<Exception>(error.cu_err, line_number, filename);
    }
    throw Exception{std::string{"cuFile error at: "} + filename + ":" +
                    std::to_string(line_number) + ": " + cufileop_status_error(error.err)};
  }
}

/**
 * @brief Check the byte count returned by a cuFile read/write and throw on failure.
 *
 * A negative value encodes an error: either a cuFile operation error (if above CUFILEOP_BASE_ERR)
 * or a standard errno value.
 *
 * @tparam Exception The exception type to throw.
 * @param nbytes_done The byte count returned by a cuFile read/write operation.
 * @param line_number Source line number (typically from __LINE__).
 * @param filename Source file name (typically from __FILE__).
 * @exception Exception Thrown if @p nbytes_done is negative.
 */
template <typename Exception>
void cufile_check_bytes_done(ssize_t nbytes_done, int line_number, char const* filename)
{
  if (nbytes_done < 0) {
    auto const err = std::abs(nbytes_done);
    auto const msg = (err > CUFILEOP_BASE_ERR)
                       ? std::string(cufileop_status_error(static_cast<CUfileOpError>(err)))
                       : std::string(std::strerror(err));
    throw Exception{std::string{"cuFile error at: "} + filename + ":" +
                    std::to_string(line_number) + ": " + msg};
  }
}

/**
 * @brief Throw a GenericSystemError with the current errno and a formatted message.
 *
 * This is the shared error-reporting path for check_linux_call() overloads.
 *
 * @param line_number Source line number (typically from __LINE__).
 * @param filename Source file name (typically from __FILE__).
 * @param extra_msg Optional extra context prepended to the error message.
 * @exception kvikio::GenericSystemError Always thrown, capturing the current errno.
 */
[[noreturn]] inline void handle_linux_call_error(int line_number,
                                                 char const* filename,
                                                 std::string_view extra_msg)
{
  std::stringstream ss;
  if (!extra_msg.empty()) { ss << extra_msg << " "; }
  ss << "Linux system/library function call error at: " << filename << ":" << line_number;

  // std::system_error::what() automatically contains the detailed error description
  // equivalent to calling strerror(errno)
  throw kvikio::GenericSystemError(ss.str());
}

/**
 * @brief Check the return value of a Linux system call and throw on failure.
 *
 * This non-template overload handles the common case where the return value is a long type (Linux
 * system call return type).
 *
 * @param return_value The return value of the system call.
 * @param line_number Source line number (typically from __LINE__).
 * @param filename Source file name (typically from __FILE__).
 * @param extra_msg Optional extra context for the error message (default: empty).
 * @param error_value The sentinel value indicating failure (default: -1).
 * @exception kvikio::GenericSystemError Thrown if @p return_value equals @p error_value.
 */
inline void check_linux_call(long return_value,
                             int line_number,
                             char const* filename,
                             std::string_view extra_msg = "",
                             long error_value           = -1)
{
  if (return_value == error_value) { handle_linux_call_error(line_number, filename, extra_msg); }
}

/**
 * @brief Check the return value of a Linux system call and throw on failure.
 *
 * This template overload handles non-integer return types such as `void*` from mmap().
 *
 * @tparam T The return type of the system call.
 * @param return_value The return value of the system call.
 * @param line_number Source line number (typically from __LINE__).
 * @param filename Source file name (typically from __FILE__).
 * @param extra_msg Extra context for the error message.
 * @param error_value The sentinel value indicating failure (e.g. MAP_FAILED for mmap).
 * @exception kvikio::GenericSystemError Thrown if @p return_value equals @p error_value.
 */
template <typename T>
void check_linux_call(
  T return_value, int line_number, char const* filename, std::string_view extra_msg, T error_value)
{
  if (return_value == error_value) { handle_linux_call_error(line_number, filename, extra_msg); }
}

}  // namespace detail

}  // namespace kvikio
