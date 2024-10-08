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

# This function finds libcurl and sets any additional necessary environment variables.
function(find_and_configure_libcurl)
  include(${rapids-cmake-dir}/cpm/find.cmake)

  rapids_cpm_find(
    CURL 7.87.0
    GLOBAL_TARGETS libcurl
    BUILD_EXPORT_SET kvikio-exports
    INSTALL_EXPORT_SET kvikio-exports
    CPM_ARGS
    GIT_REPOSITORY https://github.com/curl/curl
    GIT_TAG curl-7_87_0
    OPTIONS "BUILD_CURL_EXE OFF" "BUILD_SHARED_LIBS OFF" "BUILD_TESTING OFF" "CURL_USE_LIBPSL OFF"
            "CURL_DISABLE_LDAP ON" "CMAKE_POSITION_INDEPENDENT_CODE ON"
  )
endfunction()

find_and_configure_libcurl()
