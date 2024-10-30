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

  # Work around https://github.com/curl/curl/issues/15351
  if(DEFINED CACHE{BUILD_TESTING})
    set(CACHE_HAS_BUILD_TESTING $CACHE{BUILD_TESTING})
  endif()

  # target_link_options(my_shared_library PRIVATE "-Wl,-Bsymbolic")
  rapids_cpm_find(
    CURL 7.87.0
    GLOBAL_TARGETS libcurl
    BUILD_EXPORT_SET kvikio-exports
    INSTALL_EXPORT_SET kvikio-exports
    CPM_ARGS
    GIT_REPOSITORY https://github.com/curl/curl
    GIT_TAG curl-7_87_0
    OPTIONS "BUILD_CURL_EXE OFF"
            "BUILD_SHARED_LIBS ON"
            "BUILD_TESTING OFF"
            "ENABLE_THREADED_RESOLVER ON"
            "CURL_USE_LIBPSL OFF"
            "CURL_DISABLE_LDAP ON"
            "CMAKE_POSITION_INDEPENDENT_CODE ON"
  )
  if(DEFINED CACHE_HAS_BUILD_TESTING)
    set(BUILD_TESTING
        ${CACHE_HAS_BUILD_TESTING}
        CACHE BOOL "" FORCE
    )
  else()
    unset(BUILD_TESTING CACHE)
  endif()
endfunction()

find_and_configure_libcurl()
