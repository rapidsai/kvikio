/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>

namespace kvikio::detail {

// Defined in cpp/src/remote_handle.cpp. Only used by callback_device_memory(). The host path leaves
// CallbackContext::bounce_buffer as nullptr.
class BounceBufferH2D;

/**
 * @brief Context used by the libcurl CURLOPT_WRITEFUNCTION callbacks for remote reads.
 *
 * One instance per byte-range transfer. Captures the destination buffer, the total size requested,
 * and how many bytes have been written so far. The optional `bounce_buffer` is used only by
 * `callback_device_memory()`. The host-memory path leaves it as `nullptr`.
 */
struct CallbackContext {
  char* buf{nullptr};          ///< Output buffer (host memory directly, or staging for device).
  std::size_t size{0};         ///< Total number of bytes the caller asked us to read.
  std::ptrdiff_t offset{0};    ///< Bytes received so far.
  bool overflow_error{false};  ///< Set when the server returned more than `size`.
  BounceBufferH2D* bounce_buffer{nullptr};  ///< Only used by callback_device_memory().

  // Default-constructible so the multi-handle backend can build a `RemoteMultiTransfer`
  // and fill `buf`/`size` once the surrounding sub-range has been computed.
  CallbackContext() = default;

  // Convenience constructor used by the easy-path read() in remote_handle.cpp.
  CallbackContext(void* buf, std::size_t size) : buf{static_cast<char*>(buf)}, size{size} {}
};

/**
 * @brief Callback for `CURLOPT_WRITEFUNCTION` that copies received bytes directly into a host
 * buffer.
 *
 * The `context->buf` must be the destination host buffer. On overflow the function sets
 * `context->overflow_error = true` and returns `CURL_WRITEFUNC_ERROR`, which causes the surrounding
 * `curl_easy_perform()` / `curl_multi_perform()` to fail with `CURLE_WRITE_ERROR`.
 *
 * @param data Pointer to the libcurl-owned buffer of received bytes.
 * @param size Size of each element (always 1 per libcurl convention).
 * @param nmemb Number of bytes received
 * @param context Pointer to a `CallbackContext`.
 * @return Number of bytes consumed, or `CURL_WRITEFUNC_ERROR` on overflow.
 */
std::size_t callback_host_memory(char* data, std::size_t size, std::size_t nmemb, void* context);

/**
 * @brief Callback for `CURLOPT_WRITEFUNCTION` that stages received bytes through a pinned host
 * bounce buffer into device memory.
 *
 * The `context->bounce_buffer` must be a `BounceBufferH2D`.
 *
 * @param data Pointer to the libcurl-owned buffer of received bytes.
 * @param size Size of each element (always 1 per libcurl convention).
 * @param nmemb Number of bytes received
 * @param context Pointer to a `CallbackContext` with a non-null `bounce_buffer`.
 * @return Number of bytes consumed, or `CURL_WRITEFUNC_ERROR` on overflow.
 */
std::size_t callback_device_memory(char* data, std::size_t size, std::size_t nmemb, void* context);

/**
 * @brief Callback for `CURLOPT_WRITEFUNCTION` that copies received data into a `std::string`.
 *
 * @param data Pointer to the libcurl-owned buffer of received bytes.
 * @param size Size of each element (always 1 per libcurl convention).
 * @param num_bytes Number of bytes received
 * @param userdata Must be cast from `std::string*`
 * @return The number of bytes consumed by the callback
 */
std::size_t callback_get_string_response(char* data,
                                         std::size_t size,
                                         std::size_t num_bytes,
                                         void* userdata);
}  // namespace kvikio::detail
