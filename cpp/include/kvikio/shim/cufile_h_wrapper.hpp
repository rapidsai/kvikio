/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
 * In order to support compilation when `cufile.h` isn't available, we
 * wrap all use of cufile in a `#ifdef KVIKIO_CUFILE_FOUND` guard.
 *
 * The motivation here is to make KvikIO work in all circumstances so
 * that libraries doesn't have to implement there own fallback solutions.
 */
#ifdef KVIKIO_CUFILE_FOUND
#include <cufile.h>
#else
using CUfileDriverControlFlags_t = enum CUfileDriverControlFlags {
  CU_FILE_USE_POLL_MODE     = 0, /*!< use POLL mode. properties.use_poll_mode*/
  CU_FILE_ALLOW_COMPAT_MODE = 1  /*!< allow COMPATIBILITY mode. properties.allow_compat_mode*/
};
using CUfileHandle_t = void*;
#endif

// If the Batch API isn't defined, we define some of the data types here.
// Notice, this doesn't need to be ABI compatible with the cufile definitions.
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
#endif
