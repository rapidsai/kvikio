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

#include <cassert>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>

#include <kvikio/defaults.hpp>
#include <kvikio/error.hpp>
#include <kvikio/parallel_operation.hpp>
#include <kvikio/posix_io.hpp>
#include <kvikio/remote_handle.hpp>
#include <kvikio/shim/libcurl.hpp>
#include <kvikio/utils.hpp>

namespace kvikio {

void HttpEndpoint::setopt(CurlHandle& curl) { curl.setopt(CURLOPT_URL, _url.c_str()); }

void S3Endpoint::setopt(CurlHandle& curl)
{
  curl.setopt(CURLOPT_URL, _url.c_str());
  curl.setopt(CURLOPT_AWS_SIGV4, _aws_sigv4.c_str());
  curl.setopt(CURLOPT_USERPWD, _aws_userpwd.c_str());
}

RemoteHandle::RemoteHandle(std::unique_ptr<RemoteEndpoint> endpoint)
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

namespace {

/**
 * @brief Context used by the "CURLOPT_WRITEFUNCTION" callbacks.
 */
struct CallbackContext {
  char* buf;              // Output buffer to read into.
  std::size_t size;       // Total number of bytes to read.
  std::ptrdiff_t offset;  // Offset into `buf` to start reading.
  bool overflow_error;    // Flag to indicate overflow.
  CallbackContext(void* buf, std::size_t size)
    : buf{static_cast<char*>(buf)}, size{size}, offset{0}, overflow_error{0}
  {
  }
  detail::BounceBufferH2D* bounce_buffer{nullptr};  // Only used by callback_device_memory
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
  std::size_t const nbytes = size * nmemb;
  if (ctx->size < ctx->offset + nbytes) {
    ctx->overflow_error = true;
    return CURL_WRITEFUNC_ERROR;
  }
  KVIKIO_NVTX_SCOPED_RANGE("RemoteHandle - callback_host_memory()", nbytes);
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
  std::size_t const nbytes = size * nmemb;
  if (ctx->size < ctx->offset + nbytes) {
    ctx->overflow_error = true;
    return CURL_WRITEFUNC_ERROR;
  }
  KVIKIO_NVTX_SCOPED_RANGE("RemoteHandle - callback_device_memory()", nbytes);

  ctx->bounce_buffer->write(data, nbytes);
  ctx->offset += nbytes;
  return nbytes;
}
}  // namespace

std::size_t RemoteHandle::read(void* buf, std::size_t size, std::size_t file_offset)
{
  KVIKIO_NVTX_SCOPED_RANGE("RemoteHandle::read()", size);

  if (file_offset + size > _nbytes) {
    std::stringstream ss;
    ss << "cannot read " << file_offset << "+" << size << " bytes into a " << _nbytes
       << " bytes file (" << _endpoint->str() << ")";
    throw std::invalid_argument(ss.str());
  }
  bool const is_host_mem = is_host_memory(buf);
  auto curl              = create_curl_handle();
  _endpoint->setopt(curl);

  std::string const byte_range =
    std::to_string(file_offset) + "-" + std::to_string(file_offset + size - 1);
  curl.setopt(CURLOPT_RANGE, byte_range.c_str());

  if (is_host_mem) {
    curl.setopt(CURLOPT_WRITEFUNCTION, callback_host_memory);
  } else {
    curl.setopt(CURLOPT_WRITEFUNCTION, callback_device_memory);
  }
  CallbackContext ctx{buf, size};
  curl.setopt(CURLOPT_WRITEDATA, &ctx);

  try {
    if (is_host_mem) {
      curl.perform();
    } else {
      PushAndPopContext c(get_context_from_pointer(buf));
      // We use a bounce buffer to avoid many small memory copies to device. Libcurl has a
      // maximum chunk size of 16kb (`CURL_MAX_WRITE_SIZE`) but chunks are often much smaller.
      detail::BounceBufferH2D bounce_buffer(detail::StreamsByThread::get(), buf);
      ctx.bounce_buffer = &bounce_buffer;
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

std::future<std::size_t> RemoteHandle::pread(void* buf,
                                             std::size_t size,
                                             std::size_t file_offset,
                                             std::size_t task_size)
{
  KVIKIO_NVTX_SCOPED_RANGE("RemoteHandle::pread()", size);
  auto task = [this](void* devPtr_base,
                     std::size_t size,
                     std::size_t file_offset,
                     std::size_t devPtr_offset) -> std::size_t {
    return read(static_cast<char*>(devPtr_base) + devPtr_offset, size, file_offset);
  };
  return parallel_io(task, buf, size, file_offset, task_size, 0);
}

}  // namespace kvikio
