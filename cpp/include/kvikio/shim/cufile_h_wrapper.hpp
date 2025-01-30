/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

#include <sys/types.h>

#include <kvikio/shim/cuda_h_wrapper.hpp>

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
#ifndef KVIKIO_CUFILE_BATCH_API_FOUND
typedef enum CUfileOpcode { CUFILE_READ = 0, CUFILE_WRITE } CUfileOpcode_t;

typedef enum CUFILEStatus_enum {
  CUFILE_WAITING  = 0x000001,  /* required value prior to submission */
  CUFILE_PENDING  = 0x000002,  /* once enqueued */
  CUFILE_INVALID  = 0x000004,  /* request was ill-formed or could not be enqueued */
  CUFILE_CANCELED = 0x000008,  /* request successfully canceled */
  CUFILE_COMPLETE = 0x0000010, /* request successfully completed */
  CUFILE_TIMEOUT  = 0x0000020, /* request timed out */
  CUFILE_FAILED   = 0x0000040  /* unable to complete */
} CUfileStatus_t;

typedef struct CUfileIOEvents {
  void* cookie;
  CUfileStatus_t status; /* status of the operation */
  size_t ret;            /* -ve error or amount of I/O done. */
} CUfileIOEvents_t;

CUfileError_t cuFileBatchIOSetUp(...);
CUfileError_t cuFileBatchIOSubmit(...);
CUfileError_t cuFileBatchIOGetStatus(...);
CUfileError_t cuFileBatchIOCancel(...);
CUfileError_t cuFileBatchIODestroy(...);
#endif

#ifndef KVIKIO_CUFILE_STREAM_API_FOUND
CUfileError_t cuFileReadAsync(...);
CUfileError_t cuFileWriteAsync(...);
CUfileError_t cuFileStreamRegister(...);
CUfileError_t cuFileStreamDeregister(...);
#endif

#ifndef KVIKIO_CUFILE_VERSION_API_FOUND
CUfileError_t cuFileGetVersion(...);
#endif
