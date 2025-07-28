/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

#ifndef KVIKIO_LIBCURL_FOUND
#error \
  "cannot include the remote IO API, please build KvikIO with libcurl (-DKvikIO_REMOTE_SUPPORT=ON)"
#endif

#include <functional>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <curl/curl.h>

#include <kvikio/error.hpp>

namespace kvikio {

/**
 * @brief Singleton class to initialize and cleanup the global state of libcurl
 *
 * Notice, libcurl allows the use of a singleton class:
 *
 * In a C++ module, it is common to deal with the global constant situation by defining a special
 * class that represents the global constant environment of the module. A program always has exactly
 * one object of the class, in static storage. That way, the program automatically calls the
 * constructor of the object as the program starts up and the destructor as it terminates. As the
 * author of this libcurl-using module, you can make the constructor call curl_global_init and the
 * destructor call curl_global_cleanup and satisfy libcurl's requirements without your user having
 * to think about it. (Caveat: If you are initializing libcurl from a Windows DLL you should not
 * initialize it from DllMain or a static initializer because Windows holds the loader lock during
 * that time and it could cause a deadlock.)
 *
 * Source <https://curl.se/libcurl/c/libcurl.html>.
 */
class LibCurl {
 public:
  // We hold a unique pointer to the raw curl handle and set `curl_easy_cleanup` as its Deleter.
  using UniqueHandlePtr = std::unique_ptr<CURL, std::function<decltype(curl_easy_cleanup)>>;

 private:
  std::mutex _mutex{};
  // Curl handles free to be used.
  std::vector<UniqueHandlePtr> _free_curl_handles{};

  LibCurl();
  ~LibCurl() noexcept;

 public:
  static LibCurl& instance();

  /**
   * @brief Returns a free curl handle if available.
   */
  UniqueHandlePtr get_free_handle();

  /**
   * @brief Returns a curl handle, create a new handle if none is available.
   */
  UniqueHandlePtr get_handle();

  /**
   * @brief Retain a curl handle for later use.
   */
  void retain_handle(UniqueHandlePtr handle);
};

/**
 * @brief Representation of a curl easy handle pointer and its operations.
 *
 * An instance is given a `LibCurl::UniqueHandlePtr` on creation, which is
 * later retained on destruction.
 */
class CurlHandle {
 private:
  char _errbuf[CURL_ERROR_SIZE];
  LibCurl::UniqueHandlePtr _handle;

 public:
  /**
   * @brief Construct a new curl handle.
   *
   * Typically, do not call this directly instead use the `create_curl_handle()` macro.
   *
   * @param handle An unused curl easy handle pointer, which is retained on destruction.
   * @param source_file Path of source file of the caller (for error messages).
   * @param source_line Line of source file of the caller (for error messages).
   */
  CurlHandle(LibCurl::UniqueHandlePtr handle, std::string source_file, std::string source_line);
  ~CurlHandle() noexcept;

  /**
   * @brief CurlHandle support is not movable or copyable.
   */
  CurlHandle(CurlHandle const&)            = delete;
  CurlHandle& operator=(CurlHandle const&) = delete;
  CurlHandle(CurlHandle&& o)               = delete;
  CurlHandle& operator=(CurlHandle&& o)    = delete;

  /**
   * @brief Get the underlying curl easy handle pointer.
   */
  CURL* handle() noexcept;

  /**
   * @brief Set option for the curl handle.
   *
   * See <https://curl.se/libcurl/c/curl_easy_setopt.html> for available options.
   *
   * @tparam VAL The type of the value.
   * @param option The curl option to set.
   */
  template <typename VAL>
  void setopt(CURLoption option, VAL value)
  {
    CURLcode err = curl_easy_setopt(handle(), option, value);
    if (err != CURLE_OK) {
      std::stringstream ss;
      ss << "curl_easy_setopt() error "
         << "(" << curl_easy_strerror(err) << ")";
      KVIKIO_FAIL(ss.str(), std::runtime_error);
    }
  }

  /**
   * @brief Perform a blocking network transfer using previously set options.
   *
   * See <https://curl.se/libcurl/c/curl_easy_perform.html>.
   */
  void perform();

  /**
   * @brief Extract information from a curl handle.
   *
   * See <https://curl.se/libcurl/c/curl_easy_getinfo.html> for available options.
   *
   * @tparam OUTPUT The type of the output.
   * @param output The output, which is used as-is: `curl_easy_getinfo(..., output)`.
   */
  template <typename OUTPUT>
  void getinfo(CURLINFO info, OUTPUT* output)
  {
    CURLcode err = curl_easy_getinfo(handle(), info, output);
    if (err != CURLE_OK) {
      std::stringstream ss;
      ss << "curl_easy_getinfo() error "
         << "(" << curl_easy_strerror(err) << ")";
      KVIKIO_FAIL(ss.str(), std::runtime_error);
    }
  }
};

namespace detail {
/**
 * @brief Fix Conda's manipulation of __FILE__.
 *
 * Conda manipulates the path information in its shared libraries[1] with the results that the
 * C macro `__FILE__` might contain trailing `\0` chars. Normally, this isn't a problem because
 * `__FILE__` is a `const char*` that are terminated by the first encounter of `\0`. However, when
 * creating a `std::string` from a `char*`, the compiler might optimize the code such that the
 * `std::string` is created from the full size of `__FILE__` including the trailing `\0` chars.
 *
 * The extra `\0` is problematic if `CurlHandle` later throws an exception to Cython since, while
 * converting the exception to Python, Cython might truncate the error message.
 *
 * [1] <https://docs.conda.io/projects/conda-build/en/latest/resources/make-relocatable.html>
 */
__attribute__((noinline)) inline std::string fix_conda_file_path_hack(std::string filename)
{
  if (filename.data() != nullptr) { return std::string{filename.data()}; }
  return std::string{};
}
}  // namespace detail

/**
 * @brief Create a new curl handle.
 *
 * @returns A `kvikio::CurlHandle` instance ready to be used.
 */
#define create_curl_handle()                                             \
  kvikio::CurlHandle(kvikio::LibCurl::instance().get_handle(),           \
                     kvikio::detail::fix_conda_file_path_hack(__FILE__), \
                     KVIKIO_STRINGIFY(__LINE__))

}  // namespace kvikio
