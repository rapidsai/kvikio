/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
 * wrap all use of cufile in a `#ifdef KVIKIO_CUFILE_EXIST` guard.
 *
 * The motivation here is to make KvikIO work in all circumstances so
 * that libraries doesn't have to implement there own fallback solutions.
 */
#ifndef KVIKIO_DISABLE_CUFILE
#if __has_include(<cufile.h>)
#include <cufile.h>
#define KVIKIO_CUFILE_EXIST
#endif
#endif

#ifndef KVIKIO_CUFILE_EXIST
using CUfileDriverControlFlags_t = enum CUfileDriverControlFlags {
  CU_FILE_USE_POLL_MODE     = 0, /*!< use POLL mode. properties.use_poll_mode*/
  CU_FILE_ALLOW_COMPAT_MODE = 1  /*!< allow COMPATIBILITY mode. properties.allow_compat_mode*/
};
using CUfileHandle_t = void*;
#endif
