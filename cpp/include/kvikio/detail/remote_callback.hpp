/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * and how many bytes have been written so far. Exactly one of `buf`, `bounce_buffer`, or
 * `pinned_buffer` is the live destination for any given transfer, depending on which callback is
 * wired up:
 * - `callback_host_memory`: writes to `buf`.
 * - `callback_device_memory` (easy-path): writes to device memory via `bounce_buffer`.
 * - `callback_pinned_buffer` (multi-poll device-path): writes to `pinned_buffer`. The surrounding
 *   reactor later copies the pinned buffer to device.
 */
struct CallbackContext {
  char* buf{nullptr};          ///< Host output buffer (used by `callback_host_memory`).
  std::size_t size{0};         ///< Total number of bytes the caller asked us to read.
  std::ptrdiff_t offset{0};    ///< Bytes received so far.
  bool overflow_error{false};  ///< Set when the server returned more than `size`.
  BounceBufferH2D* bounce_buffer{nullptr};  ///< Used by `callback_device_memory` (easy-path).
  void* pinned_buffer{nullptr};  ///< Used by `callback_pinned_buffer` (multi-poll device-path).

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
 * @brief Callback for `CURLOPT_WRITEFUNCTION` that copies received bytes into a pinned host
 * buffer.
 *
 * Used by the multi-poll backend's device-buffer path. `context->pinned_buffer` must point to a
 * pinned host allocation of at least `context->size` bytes. The reactor schedules the H2D from
 * `pinned_buffer` to the user's device buffer outside this callback (after `CURLMSG_DONE`).
 *
 * On overflow the function sets `context->overflow_error = true` and returns
 * `CURL_WRITEFUNC_ERROR`.
 *
 * @param data Pointer to the libcurl-owned buffer of received bytes.
 * @param size Size of each element (always 1 per libcurl convention).
 * @param nmemb Number of bytes received.
 * @param context Pointer to a `CallbackContext` with a non-null `pinned_buffer`.
 * @return Number of bytes consumed, or `CURL_WRITEFUNC_ERROR` on overflow.
 */
std::size_t callback_pinned_buffer(char* data, std::size_t size, std::size_t nmemb, void* context);

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
