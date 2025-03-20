# =============================================================================
# Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

enable_testing()

include(rapids-test)
rapids_test_init()

file(GLOB SOURCES "*.cpp")
add_executable(cpp_tests ${SOURCES})
set_target_properties(
  cpp_tests
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${KvikIO_BINARY_DIR}/gtests>"
             INSTALL_RPATH "\$ORIGIN/../../../lib"
             CXX_STANDARD 17
             CXX_STANDARD_REQUIRED ON
             # For std:: support of __int128_t. Can be removed once using cuda::std
             CXX_EXTENSIONS ON
             CUDA_STANDARD 17
             CUDA_STANDARD_REQUIRED ON
)
target_link_libraries(cpp_tests PRIVATE kvikio::kvikio GTest::gmock GTest::gtest CUDA::cudart)

rapids_test_add(
  NAME cpp_tests
  COMMAND cpp_tests
  GPUS 1
  INSTALL_COMPONENT_SET testing
)

rapids_test_install_relocatable(INSTALL_COMPONENT_SET testing DESTINATION bin/tests/libkvikio)
