# =============================================================================
# Copyright (c) 2024, NVIDIA CORPORATION.
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

# This function finds aws-sdk-cpp and sets any additional necessary environment variables.
function(find_and_configure_aws_sdk_cpp)
  include(${rapids-cmake-dir}/cpm/find.cmake)

  rapids_cpm_find(
    AWSSDK 1.11.393
    BUILD_EXPORT_SET kvikio-exports
    INSTALL_EXPORT_SET kvikio-exports
    COMPONENTS S3
    GLOBAL_TARGETS aws-cpp-sdk-s3
    CPM_ARGS
    GIT_REPOSITORY https://github.com/aws/aws-sdk-cpp.git
    GIT_TAG 1.11.393
    PATCH_COMMAND ${CMAKE_COMMAND} -E env GIT_COMMITTER_NAME=rapids-cmake GIT_COMMITTER_EMAIL=rapids.cmake@rapids.ai git am ${CMAKE_CURRENT_LIST_DIR}/patches/aws-sdk-cpp/0001-Don-t-set-CMP0077-to-OLD.patch
    OPTIONS "BUILD_ONLY s3" "BUILD_SHARED_LIBS OFF" "ENABLE_TESTING OFF" "ENABLE_UNITY_BUILD ON"
  )
endfunction()

find_and_configure_aws_sdk_cpp()
