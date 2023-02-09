# =============================================================================
# Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

add_executable(basic_io basic_io.cpp)
target_include_directories(basic_io PRIVATE ../include ${cuFile_INCLUDE_DIRS})
target_link_libraries(basic_io PRIVATE kvikio CUDA::cudart)

if(CMAKE_COMPILER_IS_GNUCXX)
  set(KVIKIO_CXX_FLAGS "-Wall;-Werror;-Wno-unknown-pragmas")
  if(NOT cuFile_FOUND)
    set(KVIKIO_CXX_FLAGS "${KVIKIO_CXX_FLAGS};-DKVIKIO_DISABLE_CUFILE")
  endif()
  target_compile_options(basic_io PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${KVIKIO_CXX_FLAGS}>")
endif()
