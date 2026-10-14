# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# Wheel-only dependency. CPM caches pristine upstream source; all generated files stay in this build
# tree. Curl probes OpenSSL during configuration, so the libraries must already exist then.
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR CMAKE_CROSSCOMPILING)
  message(FATAL_ERROR "Bundled OpenSSL requires a native Linux build")
endif()
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
  set(openssl_platform linux-x86_64)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
  set(openssl_platform linux-aarch64)
else()
  message(FATAL_ERROR "Unsupported OpenSSL architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

set(openssl_version 3.6.5)
CPMAddPackage(
  NAME
  openssl
  VERSION
  ${openssl_version}
  URL
  https://github.com/openssl/openssl/releases/download/openssl-${openssl_version}/openssl-${openssl_version}.tar.gz
  URL_HASH
  SHA256=a2157c2830efdec3788939b00c9b0638306d3f0bbb76dc4832ee503bb397df98
  DOWNLOAD_ONLY
  YES
)
find_package(Perl REQUIRED)
find_program(KVIKIO_OPENSSL_MAKE NAMES gmake make REQUIRED)
include(ProcessorCount)
ProcessorCount(openssl_jobs)
if(openssl_jobs LESS 1 OR openssl_jobs GREATER 8)
  set(openssl_jobs 8)
endif()
if(DEFINED ENV{CMAKE_BUILD_PARALLEL_LEVEL} AND "$ENV{CMAKE_BUILD_PARALLEL_LEVEL}" MATCHES
                                               "^[1-9][0-9]*$"
)
  set(openssl_jobs "$ENV{CMAKE_BUILD_PARALLEL_LEVEL}")
endif()

set(openssl_build_dir "${CMAKE_BINARY_DIR}/_deps/openssl-build")
set(openssl_root "${CMAKE_BINARY_DIR}/_deps/openssl-install")
set(openssl_log "${CMAKE_BINARY_DIR}/_deps/openssl-build.log")
string(TOUPPER "${CMAKE_BUILD_TYPE}" openssl_build_type)
separate_arguments(
  openssl_cflags UNIX_COMMAND "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_${openssl_build_type}}"
)
set(openssl_configure
    "${CMAKE_COMMAND}"
    -E
    env
    "CC=${CMAKE_C_COMPILER}"
    "${PERL_EXECUTABLE}"
    "${openssl_SOURCE_DIR}/Configure"
    ${openssl_platform}
    no-shared
    no-module
    no-pinshared
    no-tests
    no-apps
    -fPIC
    "--prefix=${openssl_root}"
    --libdir=lib
    --openssldir=/etc/ssl
    ${openssl_cflags}
)
# Reconfigure with a clean private build if the source, compiler, flags, or configuration changes.
string(SHA256 openssl_signature
              "${openssl_version};${openssl_configure};${CMAKE_C_COMPILER_VERSION}"
)
set(openssl_previous_signature "")
if(EXISTS "${openssl_root}/build-signature")
  file(READ "${openssl_root}/build-signature" openssl_previous_signature)
endif()
if(NOT openssl_signature STREQUAL openssl_previous_signature
   OR NOT EXISTS "${openssl_root}/lib/libssl.a"
   OR NOT EXISTS "${openssl_root}/lib/libcrypto.a"
)
  file(REMOVE_RECURSE "${openssl_build_dir}" "${openssl_root}")
  file(MAKE_DIRECTORY "${openssl_build_dir}")
  file(WRITE "${openssl_log}" "")
  message(STATUS "Building OpenSSL ${openssl_version} from source; log: ${openssl_log}")
  foreach(step IN ITEMS configure install)
    if(step STREQUAL "configure")
      set(openssl_command ${openssl_configure})
    else()
      set(openssl_command "${KVIKIO_OPENSSL_MAKE}" "-j${openssl_jobs}" install_dev)
    endif()
    execute_process(
      COMMAND ${openssl_command}
      WORKING_DIRECTORY "${openssl_build_dir}"
      RESULT_VARIABLE openssl_result
      OUTPUT_VARIABLE openssl_output
      ERROR_VARIABLE openssl_error
    )
    file(APPEND "${openssl_log}" "${openssl_output}${openssl_error}")
    if(NOT openssl_result STREQUAL "0")
      message(FATAL_ERROR "OpenSSL ${step} failed: ${openssl_error}\nSee ${openssl_log}")
    endif()
  endforeach()
  file(WRITE "${openssl_root}/build-signature" "${openssl_signature}")
endif()

install(FILES "${openssl_SOURCE_DIR}/LICENSE.txt" DESTINATION "share/licenses/kvikio/openssl")
# Explicit paths avoid an older system OpenSSL or stale CMake cache entries.
set(OPENSSL_USE_STATIC_LIBS ON)
set(OPENSSL_INCLUDE_DIR "${openssl_root}/include")
set(OPENSSL_SSL_LIBRARY "${openssl_root}/lib/libssl.a")
set(OPENSSL_CRYPTO_LIBRARY "${openssl_root}/lib/libcrypto.a")
find_package(OpenSSL ${openssl_version} EXACT REQUIRED)
