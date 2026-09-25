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

  # Build libcurl without a default CA bundle or CA directory, relying instead on KvikIO's runtime
  # CA setup defined in tls.cpp. This is a workaround for a performance bug in curl >=8.21.0. Since
  # curl 8.21.0, the compile-time defaults are still used even when KvikIO sets CURLOPT_CAINFO or
  # CURLOPT_CAPATH to NULL. A CA directory disables libcurl's CA store cache, and when combined with
  # a CA bundle, every TLS connection re-parses the whole bundle. Set these as cache variables, not
  # in OPTIONS below. curl's CMake ignores an OPTIONS value on a fresh configure, and undesirably
  # uses "none" as the path on a reconfigure.
  set(CURL_CA_BUNDLE
      "none"
      CACHE STRING "No built-in CA bundle for the bundled libcurl" FORCE
  )
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
