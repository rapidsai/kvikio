# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# This function finds libcurl and sets any additional necessary environment variables.
function(find_and_configure_libcurl)
  include(${rapids-cmake-dir}/cpm/find.cmake)

  # Work around https://github.com/curl/curl/issues/15351
  if(DEFINED CACHE{BUILD_TESTING})
    set(CACHE_HAS_BUILD_TESTING $CACHE{BUILD_TESTING})
  endif()

  # KvikIO resolves the CA bundle at runtime (see detail/tls.cpp), so libcurl needs no built-in CA
  # directory. Since curl 8.21.0, a built-in CA directory is used whenever CURLOPT_CAPATH is NULL,
  # and any CA directory disables libcurl's CA store cache. Every new TLS connection then parses the
  # whole CA bundle again, which makes connection setup and reconnects expensive under load. This
  # must be a cache variable. curl removes the cache entry for "none", while an ordinary variable
  # passed through OPTIONS would survive and define the directory "none".
  set(CURL_CA_PATH
      "none"
      CACHE STRING "No built-in CA directory for the bundled libcurl" FORCE
  )

  rapids_cpm_find(
    CURL 8.13.0
    GLOBAL_TARGETS libcurl
    CPM_ARGS
    GIT_REPOSITORY https://github.com/curl/curl
    GIT_TAG curl-8_13_0
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
