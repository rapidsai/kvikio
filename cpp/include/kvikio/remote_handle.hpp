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

#include <cstddef>
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#include <kvikio/defaults.hpp>
#include <kvikio/error.hpp>
#include <kvikio/parallel_operation.hpp>
#include <kvikio/posix_io.hpp>
#include <kvikio/shim/libcurl.hpp>
#include <kvikio/utils.hpp>

namespace kvikio {
namespace detail {

/**
 * @brief Context used by the "CURLOPT_WRITEFUNCTION" callbacks.
 */
struct CallbackContext {
  char* buf;            // Output buffer to read into.
  std::size_t size;     // Total number of bytes to read.
  std::size_t offset;   // Offset into `buf` to start reading.
  bool overflow_error;  // Flag to indicate overflow.
};

/**
 * @brief A "CURLOPT_WRITEFUNCTION" to copy downloaded data to the output host buffer.
 *
 * See <https://curl.se/libcurl/c/CURLOPT_WRITEFUNCTION.html>.
 *
 * @param data Data downloaded by libcurl that is ready for consumption.
 * @param size Size of each element in `nmemb`; size is always 1.
 * @param nmemb Size of the data in `nmemb`.
 * @param context A pointer to an instance of `CallbackContext`.
 */
inline std::size_t callback_host_memory(char* data,
                                        std::size_t size,
                                        std::size_t nmemb,
                                        void* context)
{
  auto ctx                 = reinterpret_cast<CallbackContext*>(context);
  const std::size_t nbytes = size * nmemb;
  if (ctx->size < ctx->offset + nbytes) {
    ctx->overflow_error = true;
    return CURL_WRITEFUNC_ERROR;
  }
  KVIKIO_NVTX_FUNC_RANGE("RemoteHandle - callback_host_memory()", nbytes);
  std::memcpy(ctx->buf + ctx->offset, data, nbytes);
  ctx->offset += nbytes;
  return nbytes;
}

/**
 * @brief A "CURLOPT_WRITEFUNCTION" to copy downloaded data to the output device buffer.
 *
 * See <https://curl.se/libcurl/c/CURLOPT_WRITEFUNCTION.html>.
 *
 * @param data Data downloaded by libcurl that is ready for consumption.
 * @param size Size of each element in `nmemb`; size is always 1.
 * @param nmemb Size of the data in `nmemb`.
 * @param context A pointer to an instance of `CallbackContext`.
 */
inline std::size_t callback_device_memory(char* data,
                                          std::size_t size,
                                          std::size_t nmemb,
                                          void* context)
{
  auto ctx                 = reinterpret_cast<CallbackContext*>(context);
  const std::size_t nbytes = size * nmemb;
  if (ctx->size < ctx->offset + nbytes) {
    ctx->overflow_error = true;
    return CURL_WRITEFUNC_ERROR;
  }
  KVIKIO_NVTX_FUNC_RANGE("RemoteHandle - callback_device_memory()", nbytes);

  CUstream stream = detail::StreamsByThread::get();
  CUDA_DRIVER_TRY(cudaAPI::instance().MemcpyHtoDAsync(
    convert_void2deviceptr(ctx->buf + ctx->offset), data, nbytes, stream));
  // We have to sync since curl might overwrite or free `data`.
  CUDA_DRIVER_TRY(cudaAPI::instance().StreamSynchronize(stream));

  ctx->offset += nbytes;
  return nbytes;
}

}  // namespace detail

/**
 * @brief Abstract base class for remote endpoints.
 *
 * In this context, an endpoint refers to a remote file using a specify communication protocol.
 *
 * Each communication protocol, such as HTTP or S3, needs to implement this ABC and implement
 * their own ctor that takes communication protocol specific arguments.
 */
class RemoteEndpoint {
 public:
  /**
   * @brief Set needed connection options on a curl handle.
   *
   * Subsequently, a call to `curl.perform()` should connect to the endpoint.
   *
   * @param curl The curl handle.
   */
  virtual void setopt(CurlHandle& curl) = 0;

  /**
   * @brief Get a description of this remote point instance.
   *
   * @returns A string description.
   */
  virtual std::string str() = 0;

  virtual ~RemoteEndpoint() = default;
};

/**
 * @brief A remote endpoint using http.
 */
class HttpEndpoint : public RemoteEndpoint {
 private:
  std::string _url;

 public:
  HttpEndpoint(std::string url) : _url{std::move(url)} {}
  void setopt(CurlHandle& curl) override { curl.setopt(CURLOPT_URL, _url.c_str()); }
  std::string str() override { return _url; }
  ~HttpEndpoint() override = default;
};

/**
 * @brief Handle of remote file.
 */
class RemoteHandle {
 private:
  std::unique_ptr<RemoteEndpoint> _endpoint;
  std::size_t _nbytes;

 public:
  /**
   * @brief Create a new remote handle from an endpoint and a file size.
   *
   * @param endpoint Remote endpoint used for subsequently IO.
   * @param nbytes The size of the remote file (in bytes).
   */
  RemoteHandle(std::unique_ptr<RemoteEndpoint> endpoint, std::size_t nbytes)
    : _endpoint{std::move(endpoint)}, _nbytes{nbytes}
  {
  }

  /**
   * @brief Create a new remote handle from an endpoint (infers the file size).
   *
   * The file size is received from the remote server using `endpoint`.
   *
   * @param endpoint Remote endpoint used for subsequently IO.
   */
  RemoteHandle(std::unique_ptr<RemoteEndpoint> endpoint)
  {
    auto curl = create_curl_handle();

    endpoint->setopt(curl);
    curl.setopt(CURLOPT_NOBODY, 1L);
    curl.setopt(CURLOPT_FOLLOWLOCATION, 1L);
    curl.perform();
    curl_off_t cl;
    curl.getinfo(CURLINFO_CONTENT_LENGTH_DOWNLOAD_T, &cl);
    if (cl < 0) {
      throw std::runtime_error("cannot get size of " + endpoint->str() +
                               ", content-length not provided by the server");
    }
    _nbytes   = cl;
    _endpoint = std::move(endpoint);
  }

  // A remote handle is moveable but not copyable.
  RemoteHandle(RemoteHandle&& o)               = default;
  RemoteHandle& operator=(RemoteHandle&& o)    = default;
  RemoteHandle(RemoteHandle const&)            = delete;
  RemoteHandle& operator=(RemoteHandle const&) = delete;

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

    if (file_offset + size > _nbytes) {
      std::stringstream ss;
      ss << "cannot read " << file_offset << "+" << size << " bytes into a " << _nbytes
         << " bytes file (" << _endpoint->str() << ")";
      throw std::invalid_argument(ss.str());
    }
    const bool is_host_mem = is_host_memory(buf);
    auto curl              = create_curl_handle();
    _endpoint->setopt(curl);

    std::string const byte_range =
      std::to_string(file_offset) + "-" + std::to_string(file_offset + size - 1);
    curl.setopt(CURLOPT_RANGE, byte_range.c_str());

    if (is_host_mem) {
      curl.setopt(CURLOPT_WRITEFUNCTION, detail::callback_host_memory);
    } else {
      curl.setopt(CURLOPT_WRITEFUNCTION, detail::callback_device_memory);
    }
    detail::CallbackContext ctx{
      .buf = reinterpret_cast<char*>(buf), .size = size, .offset = 0, .overflow_error = false};
    curl.setopt(CURLOPT_WRITEDATA, &ctx);

    try {
      if (is_host_mem) {
        curl.perform();
      } else {
        PushAndPopContext c(get_context_from_pointer(buf));
        curl.perform();
      }
    } catch (std::runtime_error const& e) {
      if (ctx.overflow_error) {
        std::stringstream ss;
        ss << "maybe the server doesn't support file ranges? [" << e.what() << "]";
        throw std::overflow_error(ss.str());
      }
      throw;
    }
    return size;
  }

  /**
   * @brief Read from remote source into buffer (host or device memory) in parallel.
   *
   * This API is a parallel async version of `.read()` that partitions the operation
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
