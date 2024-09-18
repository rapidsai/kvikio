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
#error "cannot include <kvikio/remote_handle.hpp>, configuration did not find libcurl"
#endif

#include <cstring>
#include <sstream>
#include <stack>
#include <stdexcept>
#include <string>

#include <curl/curl.h>

#include <kvikio/defaults.hpp>
#include <kvikio/error.hpp>
#include <kvikio/parallel_operation.hpp>
#include <kvikio/posix_io.hpp>
#include <kvikio/utils.hpp>

namespace kvikio {
namespace detail {

/**
 * @brief Singleton class to initialize and cleanup the global state of libcurl
 *
 * https://curl.se/libcurl/c/libcurl.html
 * In a C++ module, it is common to deal with the global constant situation by defining a special
 * class that represents the global constant environment of the module. A program always has exactly
 * one object of the class, in static storage. That way, the program automatically calls the
 * constructor of the object as the program starts up and the destructor as it terminates. As the
 * author of this libcurl-using module, you can make the constructor call curl_global_init and the
 * destructor call curl_global_cleanup and satisfy libcurl's requirements without your user having
 * to think about it. (Caveat: If you are initializing libcurl from a Windows DLL you should not
 * initialize it from DllMain or a static initializer because Windows holds the loader lock during
 * that time and it could cause a deadlock.)
 */
class LibCurl {
 private:
  std::stack<CURL*> _free_curl_handles{};

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
    // clean up all retained easy handles
    while (!_free_curl_handles.empty()) {
      curl_easy_cleanup(_free_curl_handles.top());
      _free_curl_handles.pop();
    }
    curl_global_cleanup();
  }

  LibCurl(LibCurl const&)            = delete;
  LibCurl& operator=(LibCurl const&) = delete;
  LibCurl(LibCurl&& o)               = delete;
  LibCurl& operator=(LibCurl&& o)    = delete;

 public:
  void put(CURL* handle) { _free_curl_handles.push(handle); }

  static LibCurl& instance()
  {
    static LibCurl _instance;
    return _instance;
  }

  CURL* get()
  {
    // Check if we have a handle available.
    if (!_free_curl_handles.empty()) {
      CURL* ret = _free_curl_handles.top();
      _free_curl_handles.pop();
      curl_easy_reset(ret);
      return ret;
    }
    // If not, we create a new handle.
    CURL* ret = curl_easy_init();
    if (ret == nullptr) { throw std::runtime_error("libcurl: call to curl_easy_init() failed"); }
    return ret;
  }
};

/**
 * @brief A wrapper of a curl easy handle pointer `CURL*`.
 */
class CurlHandle {
 private:
  char _errbuf[CURL_ERROR_SIZE];
  CURL* _handle;
  std::string _source_file;
  std::string _source_line;

 public:
  CurlHandle(CURL* handle, std::string source_file, std::string source_line)
    : _handle{handle}, _source_file(std::move(source_file)), _source_line(std::move(source_line))
  {
  }
  ~CurlHandle() noexcept { detail::LibCurl::instance().put(_handle); }

  CurlHandle(CurlHandle const&)            = delete;
  CurlHandle& operator=(CurlHandle const&) = delete;
  CurlHandle(CurlHandle&& o)               = delete;
  CurlHandle& operator=(CurlHandle&& o)    = delete;

  CURL* handle() noexcept { return _handle; }

  template <typename OPT, typename VAL>
  void setopt(OPT option, VAL value)
  {
    CURLcode err = curl_easy_setopt(handle(), option, value);
    if (err != CURLE_OK) {
      std::stringstream ss;
      ss << "curl_easy_setopt() error near " << _source_file << ":" << _source_line;
      ss << "(" << curl_easy_strerror(err) << ")";
      throw std::runtime_error(ss.str());
    }
  }
  void perform()
  {
    setopt(CURLOPT_ERRORBUFFER, _errbuf);
    CURLcode err = curl_easy_perform(handle());
    if (err != CURLE_OK) {
      std::string msg(_errbuf);
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
  template <typename INFO, typename VALUE>
  void getinfo(INFO info, VALUE value)
  {
    CURLcode err = curl_easy_getinfo(handle(), info, value);
    if (err != CURLE_OK) {
      std::stringstream ss;
      ss << "curl_easy_getinfo() error near " << _source_file << ":" << _source_line;
      ss << "(" << curl_easy_strerror(err) << ")";
      throw std::runtime_error(ss.str());
    }
  }
};

#define create_curl_handle()  \
  kvikio::detail::CurlHandle( \
    kvikio::detail::LibCurl::instance().get(), __FILE__, KVIKIO_STRINGIFY(__LINE__))

inline std::size_t get_file_size(std::string url)
{
  auto curl = create_curl_handle();

  curl.setopt(CURLOPT_URL, url.c_str());
  curl.setopt(CURLOPT_NOBODY, 1L);
  curl.setopt(CURLOPT_FAILONERROR, 1L);
  curl.perform();

  curl_off_t cl;
  curl.getinfo(CURLINFO_CONTENT_LENGTH_DOWNLOAD_T, &cl);
  if (cl < 0) {
    throw std::runtime_error("cannot get size of " + url +
                             ", content-length not provided by the server");
  }
  return cl;
}

struct CallbackContext {
  char* buf;
  std::size_t size;
  std::size_t offset;
};

inline std::size_t callback_host_memory(char* data,
                                        std::size_t size,
                                        std::size_t nmemb,
                                        void* context)
{
  auto ctx           = reinterpret_cast<CallbackContext*>(context);
  std::size_t nbytes = size * nmemb;
  if (ctx->size < ctx->offset + nbytes) {
    std::cout << "callback_host_memory() - FAILED: " << ((void*)data)
              << ", ctx->buf: " << (void*)ctx->buf << ", offset: " << ctx->offset
              << ", nbytes: " << nbytes << std::endl;
    return CURL_WRITEFUNC_ERROR;
  }

  // std::cout << "callback_host_memory() - data: " << ((void*)data)
  //           << ", ctx->buf: " << (void*)ctx->buf << ", offset: " << ctx->offset
  //           << ", nbytes: " << nbytes << std::endl;

  std::memcpy(ctx->buf + ctx->offset, data, nbytes);
  ctx->offset += nbytes;
  return nbytes;
}

inline std::size_t callback_device_memory(char* data,
                                          std::size_t size,
                                          std::size_t nmemb,
                                          void* context)
{
  auto ctx           = reinterpret_cast<CallbackContext*>(context);
  std::size_t nbytes = size * nmemb;
  if (ctx->size < ctx->offset + nbytes) {
    std::cout << "callback_device_memory() - FAILED: " << ((void*)data)
              << ", ctx->buf: " << (void*)ctx->buf << ", offset: " << ctx->offset
              << ", nbytes: " << nbytes << std::endl;

    return CURL_WRITEFUNC_ERROR;
  }

  CUcontext cuda_ctx = get_context_from_pointer(ctx->buf);
  PushAndPopContext c(cuda_ctx);
  CUstream stream = detail::StreamsByThread::get();

  // std::cout << "callback_device_memory() - data: " << ((void*)data)
  //           << ", ctx->buf: " << (void*)ctx->buf << ", offset: " << ctx->offset
  //           << ", nbytes: " << nbytes << std::endl;

  CUDA_DRIVER_TRY(cudaAPI::instance().MemcpyHtoDAsync(
    convert_void2deviceptr(ctx->buf + ctx->offset), data, nbytes, stream));
  CUDA_DRIVER_TRY(cudaAPI::instance().StreamSynchronize(stream));

  ctx->offset += nbytes;
  return nbytes;
}

}  // namespace detail

/**
 * @brief Handle of remote file.
 */
class RemoteHandle {
 private:
  std::string _url;
  std::size_t _nbytes;

 public:
  RemoteHandle(std::string url, std::size_t nbytes) : _url(std::move(url)), _nbytes{nbytes}
  {
    std::cout << "RemoteHandle(" << _url << ") - nbytes: " << _nbytes << std::endl;
  }

  RemoteHandle(std::string const& url) : RemoteHandle(url, detail::get_file_size(url)) {}

  RemoteHandle(RemoteHandle const&)            = delete;
  RemoteHandle& operator=(RemoteHandle const&) = delete;
  RemoteHandle(RemoteHandle&& o)               = delete;
  RemoteHandle& operator=(RemoteHandle&& o)    = delete;

  /**
   * @brief Get the file size.
   *
   * Note, this is very fast, no communication needed.
   *
   * @return The number of bytes.
   */
  [[nodiscard]] std::size_t nbytes() const noexcept { return _nbytes; }

  /**
   * @brief Read from remote source into buffer (host or device memory).
   *
   * @param buf Pointer to host or device memory.
   * @param size Number of bytes to read.
   * @param file_offset File offset in bytes.
   * @return Number of bytes read, which is always `size`.
   */
  std::size_t read(void* buf, std::size_t size, std::size_t file_offset = 0)
  {
    KVIKIO_NVTX_FUNC_RANGE("RemoteHandle::read()", size);

    auto curl = create_curl_handle();

    curl.setopt(CURLOPT_URL, _url.c_str());
    curl.setopt(CURLOPT_FAILONERROR, 1L);

    std::string const byte_range =
      std::to_string(file_offset) + "-" + std::to_string(file_offset + size - 1);
    curl.setopt(CURLOPT_RANGE, byte_range.c_str());

    if (is_host_memory(buf)) {
      curl.setopt(CURLOPT_WRITEFUNCTION, detail::callback_host_memory);
    } else {
      curl.setopt(CURLOPT_WRITEFUNCTION, detail::callback_device_memory);
    }
    detail::CallbackContext ctx{.buf = reinterpret_cast<char*>(buf), .size = size, .offset = 0};
    curl.setopt(CURLOPT_WRITEDATA, &ctx);

    // std::cout << "read() - buf: " << buf << ", byte_range: " << byte_range << std::endl;
    curl.perform();
    return size;
  }

  /**
   * @brief Read from remote source into buffer (host or device memory) in parallel.
   *
   * This API is a parallel async version of `.read()` that partition the operation
   * into tasks of size `task_size` for execution in the default thread pool.
   *
   * @param buf Pointer to host or device memory.
   * @param size Number of bytes to read.
   * @param file_offset File offset in bytes.
   * @param task_size Size of each task in bytes.
   * @return Number of bytes read, which is `size` always.
   */
  std::future<std::size_t> pread(void* buf,
                                 std::size_t size,
                                 std::size_t file_offset = 0,
                                 std::size_t task_size   = defaults::task_size())
  {
    KVIKIO_NVTX_FUNC_RANGE("RemoteHandle::pread()", size);
    auto task = [this](void* devPtr_base,
                       std::size_t size,
                       std::size_t file_offset,
                       std::size_t devPtr_offset) -> std::size_t {
      return read(static_cast<char*>(devPtr_base) + devPtr_offset, size, file_offset);
    };
    return parallel_io(task, buf, size, file_offset, task_size, 0);
  }
};

}  // namespace kvikio
