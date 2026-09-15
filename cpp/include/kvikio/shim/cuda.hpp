/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <any>
#include <cstddef>
#include <functional>
#include <span>

#include <cuda.h>
#include <kvikio/shim/utils.hpp>
#include <stdexcept>

namespace kvikio {

namespace detail {
/**
 * @brief Non-templated class to hold any callable that returns CUresult
 */
class AnyCallable {
 private:
  std::any _callable;

 public:
  /**
   * @brief Assign a callable to the object
   *
   * @tparam Callable A callable that must return CUresult
   * @param c The callable object
   */
  template <typename Callable>
  void set(Callable&& c)
  {
    _callable = std::function(c);
  }

  /**
   * @brief Destroy the contained callable
   */
  void reset() { _callable.reset(); }

  /**
   * @brief Invoke the container callable
   *
   * @tparam Args Types of the argument. Must exactly match the parameter types of the contained
   * callable. For example, if the parameter is `std::size_t*`, an argument of `nullptr` must be
   * explicitly cast to `std::size_t*`.
   * @param args Arguments to be passed
   * @return CUDA driver API error code
   * @exception std::bad_any_cast if any argument type does not exactly match the parameter type of
   * the contained callable.
   */
  template <typename... Args>
  CUresult operator()(Args... args)
  {
    using T = std::function<CUresult(Args...)>;
    if (!_callable.has_value()) {
      throw std::runtime_error("No callable has been assigned to the wrapper yet.");
    }
    return std::any_cast<T&>(_callable)(args...);
  }

  /**
   * @brief Check if the object holds a callable
   */
  operator bool() const { return _callable.has_value(); }
};

}  // namespace detail

/**
 * @brief Shim layer of the cuda C-API
 *
 * This is a singleton class that use `dlopen` on construction to load the C-API of cuda.
 *
 * For example, `cudaAPI::instance().MemHostAlloc()` corresponds to calling `cuMemHostAlloc()`
 */
class cudaAPI {
 public:
  int driver_version{0};

  decltype(cuInit)* Init{nullptr};
  decltype(cuMemHostAlloc)* MemHostAlloc{nullptr};
  decltype(cuMemFreeHost)* MemFreeHost{nullptr};
  decltype(cuMemHostRegister)* MemHostRegister{nullptr};
  decltype(cuMemHostUnregister)* MemHostUnregister{nullptr};
  decltype(cuMemcpyHtoDAsync)* MemcpyHtoDAsync{nullptr};
  decltype(cuMemcpyDtoHAsync)* MemcpyDtoHAsync{nullptr};
  decltype(cuMemcpyAsync)* MemcpyAsync{nullptr};

  detail::AnyCallable MemcpyBatchAsync{};

  decltype(cuPointerGetAttribute)* PointerGetAttribute{nullptr};
  decltype(cuPointerGetAttributes)* PointerGetAttributes{nullptr};
  decltype(cuCtxCreate)* CtxCreate{nullptr};
  decltype(cuCtxDestroy)* CtxDestroy{nullptr};
  decltype(cuCtxPushCurrent)* CtxPushCurrent{nullptr};
  decltype(cuCtxPopCurrent)* CtxPopCurrent{nullptr};
  decltype(cuCtxGetCurrent)* CtxGetCurrent{nullptr};
  decltype(cuCtxGetDevice)* CtxGetDevice{nullptr};
  decltype(cuMemGetAddressRange)* MemGetAddressRange{nullptr};
  decltype(cuGetErrorName)* GetErrorName{nullptr};
  decltype(cuGetErrorString)* GetErrorString{nullptr};
  decltype(cuDeviceGet)* DeviceGet{nullptr};
  decltype(cuDeviceGetCount)* DeviceGetCount{nullptr};
  decltype(cuDeviceGetAttribute)* DeviceGetAttribute{nullptr};
  decltype(cuDevicePrimaryCtxRetain)* DevicePrimaryCtxRetain{nullptr};
  decltype(cuDevicePrimaryCtxRelease)* DevicePrimaryCtxRelease{nullptr};
  decltype(cuStreamSynchronize)* StreamSynchronize{nullptr};
  decltype(cuStreamCreate)* StreamCreate{nullptr};
  decltype(cuStreamDestroy)* StreamDestroy{nullptr};
  decltype(cuDriverGetVersion)* DriverGetVersion{nullptr};
  decltype(cuEventSynchronize)* EventSynchronize{nullptr};
  decltype(cuEventCreate)* EventCreate{nullptr};
  decltype(cuEventDestroy)* EventDestroy{nullptr};
  decltype(cuEventRecord)* EventRecord{nullptr};
  decltype(cuEventQuery)* EventQuery{nullptr};
  decltype(cuLaunchHostFunc)* LaunchHostFunc{nullptr};

 private:
  cudaAPI();

 public:
  cudaAPI(cudaAPI const&)        = delete;
  void operator=(cudaAPI const&) = delete;

  KVIKIO_EXPORT static cudaAPI& instance();

  /**
   * @brief Asynchronous memcpy that prefers `cuMemcpyBatchAsync` when supported.
   *
   * Equivalent to `cuda_memcpy_batch_async()` with a single entry.
   *
   * @param dst    Destination pointer (host or device under UVA).
   * @param src    Source pointer (host or device under UVA).
   * @param size   Number of bytes to copy.
   * @param stream CUDA stream for ordering.
   * @return CUresult from the underlying driver call.
   */
  static CUresult cuda_memcpy_async(CUdeviceptr dst,
                                    CUdeviceptr src,
                                    std::size_t size,
                                    CUstream stream);

  /**
   * @brief Asynchronous batched memcpy that prefers `cuMemcpyBatchAsync` when supported.
   *
   * Copies `sizes[i]` bytes from `srcs[i]` to `dsts[i]` for every entry. On CUDA 12.8 and newer,
   * when the batch symbol was loaded and `stream` is not the default stream, all entries are
   * submitted in a single `cuMemcpyBatchAsync` call using `CU_MEMCPY_SRC_ACCESS_ORDER_STREAM`.
   * Otherwise each entry is issued with `cuMemcpyAsync` and the first failure is returned without
   * attempting the remaining entries.
   *
   * @param dsts   Destination pointers (host or device under UVA).
   * @param srcs   Source pointers (host or device under UVA).
   * @param sizes  Number of bytes to copy for the corresponding entry.
   * @param stream CUDA stream for ordering.
   * @return CUresult from the underlying driver call, or `CUDA_SUCCESS` for an empty batch.
   * @exception std::invalid_argument if the three spans do not have the same length.
   */
  static CUresult cuda_memcpy_batch_async(std::span<CUdeviceptr const> dsts,
                                          std::span<CUdeviceptr const> srcs,
                                          std::span<std::size_t const> sizes,
                                          CUstream stream);
};

/**
 * @brief Check if the CUDA library is available
 *
 * Notice, this doesn't check if the runtime environment supports CUDA.
 *
 * @return The boolean answer
 */
bool is_cuda_available();

}  // namespace kvikio
