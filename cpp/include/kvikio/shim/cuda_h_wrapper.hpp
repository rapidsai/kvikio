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
// Notice, this doesn't need to be ABI compatible with the CUDA definitions.

using CUresult    = int;
using CUdeviceptr = unsigned long long;
using CUdevice    = int;
using CUcontext   = void*;
using CUstream    = void*;

#define CUDA_ERROR_STUB_LIBRARY             0
#define CUDA_SUCCESS                        0
#define CUDA_ERROR_INVALID_VALUE            0
#define CU_POINTER_ATTRIBUTE_CONTEXT        0
#define CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL 0
#define CU_POINTER_ATTRIBUTE_DEVICE_POINTER 0
#define CU_MEMHOSTREGISTER_PORTABLE         0

CUresult cuInit(...);
CUresult cuMemHostAlloc(...);
CUresult cuMemFreeHost(...);
CUresult cuMemcpyHtoD(...);
CUresult cuMemcpyDtoH(...);
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
CUresult cuStreamSynchronize(...);

#endif
