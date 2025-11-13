/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <any>
#include <functional>

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

  detail::AnyCallable MemcpyBatchAsync{};

  decltype(cuPointerGetAttribute)* PointerGetAttribute{nullptr};
  decltype(cuPointerGetAttributes)* PointerGetAttributes{nullptr};
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

 private:
  cudaAPI();

 public:
  cudaAPI(cudaAPI const&)        = delete;
  void operator=(cudaAPI const&) = delete;

  KVIKIO_EXPORT static cudaAPI& instance();
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
