/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <iostream>
#include <tuple>

#include <cuda.h>

#include <cufile.h>
#include <kvikio/error.hpp>

namespace kvikio {

inline off_t convert_size2off(std::size_t x)
{
  if (x >= std::numeric_limits<off_t>::max()) {
    throw CUfileException("size_t argument too large to fit off_t");
  }
  return static_cast<off_t>(x);
}

inline CUdeviceptr convert_void2deviceptr(const void* devPtr)
{
  /*NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)*/
  return reinterpret_cast<CUdeviceptr>(devPtr);
}

inline CUcontext get_context_from_device_pointer(const void* devPtr)
{
  CUcontext ctx{};
  auto dev = convert_void2deviceptr(devPtr);
  CUDA_TRY(cuPointerGetAttribute(&ctx, CU_POINTER_ATTRIBUTE_CONTEXT, dev));
  return ctx;
}

/**
 * @brief Push CUDA context on creation and pop it on destruction
 */
class PushAndPopContext {
 private:
  CUcontext _ctx;

 public:
  PushAndPopContext(CUcontext ctx) : _ctx{ctx} { CUDA_TRY(cuCtxPushCurrent(_ctx)); }
  PushAndPopContext(const void* devPtr) : _ctx{get_context_from_device_pointer(devPtr)}
  {
    CUDA_TRY(cuCtxPushCurrent(_ctx));
  }
  PushAndPopContext(const PushAndPopContext&) = delete;
  PushAndPopContext& operator=(PushAndPopContext const&) = delete;
  PushAndPopContext(PushAndPopContext&&)                 = delete;
  PushAndPopContext&& operator=(PushAndPopContext&&) = delete;
  ~PushAndPopContext()
  {
    try {
      CUDA_TRY(cuCtxPopCurrent(&_ctx), CUfileException);
    } catch (const CUfileException &e) {
      std::cerr << e.what() << std::endl;
    }
  }
};

// Find the base and offset of the memory allocation `devPtr` is in
inline std::tuple<void*, std::size_t, std::size_t> get_alloc_info(const void* devPtr,
                                                                  CUcontext* ctx = nullptr)
{
  auto dev = convert_void2deviceptr(devPtr);
  CUdeviceptr base_ptr{};
  std::size_t base_size{};
  CUcontext _ctx{};
  if (ctx != nullptr) {
    _ctx = *ctx;
  } else {
    _ctx = get_context_from_device_pointer(devPtr);
  }
  PushAndPopContext context(_ctx);
  CUDA_TRY(cuMemGetAddressRange(&base_ptr, &base_size, dev));
  std::size_t offset = dev - base_ptr;
  /*NOLINTNEXTLINE(performance-no-int-to-ptr, cppcoreguidelines-pro-type-reinterpret-cast)*/
  return std::make_tuple(reinterpret_cast<void*>(base_ptr), base_size, offset);
}

}  // namespace kvikio