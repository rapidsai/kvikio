# =============================================================================
# Copyright (c) 2022-2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
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
  LINKED_LIBRARIES kvikio::kvikio nvcomp::nvcomp
)
if(USE_NVCOMP_RUNTIME_WHEEL)
  set(rpaths "$ORIGIN/../../nvidia/nvcomp")
  foreach(tgt IN LISTS RAPIDS_CYTHON_CREATED_TARGETS)
    set_property(
      TARGET ${tgt}
      PROPERTY INSTALL_RPATH ${rpaths}
      APPEND
    )
  endforeach()
endif()
