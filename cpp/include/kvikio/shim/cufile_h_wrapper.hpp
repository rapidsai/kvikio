/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <sys/types.h>

#include <cuda.h>

/**
 * In order to support compilation when `cufile.h` isn't available, we
 * wrap all use of cufile in a `#ifdef KVIKIO_CUFILE_FOUND` guard.
 *
 * The motivation here is to make KvikIO work in all circumstances so
 * that libraries doesn't have to implement there own fallback solutions.
 */
#ifdef KVIKIO_CUFILE_FOUND
#include <cufile.h>
#else

// If cuFile isn't defined, we define some of the data types here.
// Notice, this doesn't need to be ABI compatible with the cufile definitions.

using CUfileHandle_t = void*;
using CUfileOpError  = int;
#define CUFILE_ERRSTR(x)          ("KvikIO not compiled with cuFile.h")
#define CUFILEOP_BASE_ERR         5000
#define CU_FILE_SUCCESS           0
#define CU_FILE_CUDA_DRIVER_ERROR 1

struct CUfileError_t {
  CUfileOpError err;  // cufile error
  CUresult cu_err;    // cuda driver error
};

using CUfileDriverControlFlags_t = enum CUfileDriverControlFlags {
  CU_FILE_USE_POLL_MODE     = 0,
  CU_FILE_ALLOW_COMPAT_MODE = 1
};

enum CUfileFileHandleType { CU_FILE_HANDLE_TYPE_OPAQUE_FD = 1 };

struct CUfileDescr_t {
  enum CUfileFileHandleType type;
  struct handle_t {
    int fd;
  } handle;
};

inline static char const* cufileop_status_error(CUfileOpError err) { return CUFILE_ERRSTR(err); };
CUfileError_t cuFileHandleRegister(...);
CUfileError_t cuFileHandleDeregister(...);
ssize_t cuFileRead(...);
ssize_t cuFileWrite(...);
CUfileError_t cuFileBufRegister(...);
CUfileError_t cuFileBufDeregister(...);
CUfileError_t cuFileDriverOpen(...);
CUfileError_t cuFileDriverClose(...);
CUfileError_t cuFileDriverGetProperties(...);
CUfileError_t cuFileDriverSetPollMode(...);
CUfileError_t cuFileDriverSetMaxCacheSize(...);
CUfileError_t cuFileDriverSetMaxPinnedMemSize(...);

#endif

// If some cufile APIs aren't defined, we define some of the data types here.
// Notice, this doesn't need to be ABI compatible with the cufile definitions and
// the lack of definitions is not a problem because the linker will never look for
// these symbols because the "real" function calls are made through the shim instance.

#ifndef KVIKIO_CUFILE_VERSION_API_FOUND
CUfileError_t cuFileGetVersion(...);
#endif
