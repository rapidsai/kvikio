# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# This function finds libcurl and sets any additional necessary environment variables.
function(find_and_configure_libcurl)
  include(${rapids-cmake-dir}/cpm/find.cmake)

  # In CI builds, prefer the OpenSSL from the locked Conda build environment. The
  # host system OpenSSL can be older than the version required by the pinned curl
  # source, even when the build environment already contains a newer OpenSSL.
  if(DEFINED ENV{CONDA_PREFIX})
    set(_conda_prefix "$ENV{CONDA_PREFIX}")
    if(EXISTS "${_conda_prefix}/include/openssl/ssl.h"
       AND (EXISTS "${_conda_prefix}/lib/libcrypto.so"
            OR EXISTS "${_conda_prefix}/lib/libcrypto.a"
            OR EXISTS "${_conda_prefix}/lib/libcrypto.dylib"))
      set(OPENSSL_ROOT_DIR "${_conda_prefix}" CACHE PATH "Prefer locked Conda OpenSSL" FORCE)
    endif()
  endif()

  # Work around https://github.com/curl/curl/issues/15351
  if(DEFINED CACHE{BUILD_TESTING})
    set(CACHE_HAS_BUILD_TESTING $CACHE{BUILD_TESTING})
  endif()

  rapids_cpm_find(
    CURL 8.22.0
    GLOBAL_TARGETS libcurl
    CPM_ARGS
    GIT_REPOSITORY https://github.com/curl/curl
    GIT_TAG curl-8_22_0
    OPTIONS "BUILD_CURL_EXE OFF" "BUILD_SHARED_LIBS OFF" "BUILD_TESTING OFF" "CURL_USE_LIBPSL OFF"
            "CURL_DISABLE_LDAP ON" "CMAKE_POSITION_INDEPENDENT_CODE ON"
    EXCLUDE_FROM_ALL YES # Don't install libcurl.a (only needed when building libkvikio.so)
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
