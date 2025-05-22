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

/**
 * In order to support compilation when `cuda.h` isn't available, we
 * wrap all use of cuda in a `#ifdef KVIKIO_CUDA_FOUND` guard.
 *
 * The motivation here is to make KvikIO work in all circumstances so
 * that libraries doesn't have to implement there own fallback solutions.
 */
#ifdef KVIKIO_CUDA_FOUND
#include <cuda.h>
#else

// If CUDA isn't defined, we define some of the data types here.
// Notice, the functions and constant values don't need to match the CUDA
// definitions, but the types *do*, since downstream libraries dlsym()-ing
// the symbols at runtime rely on accurate type definitions. If we mismatch
// here, then those libraries will get "mismatched type alias redefinition"
// errors when they include our headers.

#if defined(_WIN64) || defined(__LP64__)
// Don't use uint64_t, we want to match the driver headers exactly
using CUdeviceptr = unsigned long long;
#else
using CUdeviceptr = unsigned int;
#endif
static_assert(sizeof(CUdeviceptr) == sizeof(void*));

using CUresult  = int;
using CUdevice  = int;
using CUcontext = struct CUctx_st*;
using CUstream  = struct CUstream_st*;

#define CUDA_ERROR_STUB_LIBRARY             0
#define CUDA_SUCCESS                        0
#define CUDA_ERROR_INVALID_VALUE            0
#define CU_POINTER_ATTRIBUTE_CONTEXT        0
#define CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL 0
#define CU_POINTER_ATTRIBUTE_DEVICE_POINTER 0
#define CU_MEMHOSTALLOC_PORTABLE            0
#define CU_STREAM_DEFAULT                   0

CUresult cuInit(...);
CUresult cuMemHostAlloc(...);
CUresult cuMemFreeHost(...);
CUresult cuMemcpyHtoDAsync(...);
CUresult cuMemcpyDtoHAsync(...);
CUresult cuPointerGetAttribute(...);
CUresult cuPointerGetAttributes(...);
CUresult cuCtxPushCurrent(...);
CUresult cuCtxPopCurrent(...);
CUresult cuCtxGetCurrent(...);
CUresult cuMemGetAddressRange(...);
CUresult cuGetErrorName(...);
CUresult cuGetErrorString(...);
CUresult cuDeviceGet(...);
CUresult cuDevicePrimaryCtxRetain(...);
CUresult cuDevicePrimaryCtxRelease(...);
CUresult cuStreamCreate(...);
CUresult cuStreamDestroy(...);
CUresult cuStreamSynchronize(...);

#endif
