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

#ifndef KVIKIO_LIBCURL_FOUND
#error \
  "cannot include the remote IO API, please build KvikIO with libcurl (-DKvikIO_REMOTE_SUPPORT=ON)"
#endif

#include <cstring>
#include <functional>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <curl/curl.h>

#include <kvikio/defaults.hpp>
#include <kvikio/error.hpp>
#include <kvikio/parallel_operation.hpp>
#include <kvikio/posix_io.hpp>
#include <kvikio/utils.hpp>

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

  LibCurl()
  {
    CURLcode err = curl_global_init(CURL_GLOBAL_DEFAULT);
    if (err != CURLE_OK) {
      throw std::runtime_error("cannot initialize libcurl - errorcode: " + std::to_string(err));
    }
    curl_version_info_data* ver = curl_version_info(::CURLVERSION_NOW);
    if ((ver->features & CURL_VERSION_THREADSAFE) == 0) {
      throw std::runtime_error("cannot initialize libcurl - built with thread safety disabled");
    }
  }
  ~LibCurl() noexcept
  {
    _free_curl_handles.clear();
    curl_global_cleanup();
  }

 public:
  static LibCurl& instance()
  {
    static LibCurl _instance;
    return _instance;
  }

  /**
   * @brief Returns a free curl handle if available.
   */
  UniqueHandlePtr get_free_handle()
  {
    UniqueHandlePtr ret;
    std::lock_guard const lock(_mutex);
    if (!_free_curl_handles.empty()) {
      ret = std::move(_free_curl_handles.back());
      _free_curl_handles.pop_back();
    }
    return ret;
  }

  /**
   * @brief Returns a curl handle, create a new handle if none is available.
   */
  UniqueHandlePtr get_handle()
  {
    // Check if we have a free handle available.
    UniqueHandlePtr ret = get_free_handle();
    if (ret) {
      curl_easy_reset(ret.get());
    } else {
      // If not, we create a new handle.
      CURL* raw_handle = curl_easy_init();
      if (raw_handle == nullptr) {
        throw std::runtime_error("libcurl: call to curl_easy_init() failed");
      }
      ret = UniqueHandlePtr(raw_handle, curl_easy_cleanup);
    }
    return ret;
  }

  /**
   * @brief Retain a curl handle for later use.
   */
  void retain_handle(UniqueHandlePtr handle)
  {
    std::lock_guard const lock(_mutex);
    _free_curl_handles.push_back(std::move(handle));
  }
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
  std::string _source_file;
  std::string _source_line;

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
  CurlHandle(LibCurl::UniqueHandlePtr handle, std::string source_file, std::string source_line)
    : _handle{std::move(handle)},
      _source_file(std::move(source_file)),
      _source_line(std::move(source_line))
  {
    // Need CURLOPT_NOSIGNAL to support threading, see
    // <https://curl.se/libcurl/c/CURLOPT_NOSIGNAL.html>
    setopt(CURLOPT_NOSIGNAL, 1L);

    // We always set CURLOPT_ERRORBUFFER to get better error messages.
    _errbuf[0] = 0;  // Set the error buffer as empty.
    setopt(CURLOPT_ERRORBUFFER, _errbuf);

    // Make curl_easy_perform() fail when receiving HTTP code errors.
    setopt(CURLOPT_FAILONERROR, 1L);
  }
  ~CurlHandle() noexcept { LibCurl::instance().retain_handle(std::move(_handle)); }

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
  CURL* handle() noexcept { return _handle.get(); }

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
      ss << "curl_easy_setopt() error near " << _source_file << ":" << _source_line;
      ss << "(" << curl_easy_strerror(err) << ")";
      throw std::runtime_error(ss.str());
    }
  }

  /**
   * @brief Perform a blocking network transfer using previously set options.
   *
   * See <https://curl.se/libcurl/c/curl_easy_perform.html>.
   */
  void perform()
  {
    // Perform the curl operation and check for errors.
    CURLcode err = curl_easy_perform(handle());
    if (err != CURLE_OK) {
      std::string msg(_errbuf);  // We can do this because we always initialize `_errbuf` as empty.
      std::stringstream ss;
      ss << "curl_easy_perform() error near " << _source_file << ":" << _source_line;
      if (msg.empty()) {
        ss << "(" << curl_easy_strerror(err) << ")";
      } else {
        ss << "(" << msg << ")";
      }
      throw std::runtime_error(ss.str());
    }
  }

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
      ss << "curl_easy_getinfo() error near " << _source_file << ":" << _source_line;
      ss << "(" << curl_easy_strerror(err) << ")";
      throw std::runtime_error(ss.str());
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
