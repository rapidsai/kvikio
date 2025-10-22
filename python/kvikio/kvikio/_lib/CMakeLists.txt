# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# Set the list of Cython files to build, one .so per file
set(cython_modules arr.pyx buffer.pyx defaults.pyx cufile_driver.pyx file_handle.pyx future.pyx
                   mmap.pyx
)

if(KvikIO_REMOTE_SUPPORT)
  message(STATUS "Building remote_handle.pyx (libcurl found)")
  list(APPEND cython_modules remote_handle.pyx)
else()
  message(
    STATUS
      "Skipping remote_handle.pyx (please set KvikIO_REMOTE_SUPPORT=ON for remote file support)"
  )
endif()

rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_modules}"
  LINKED_LIBRARIES kvikio::kvikio
)
