# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# cmakelang does not yet recognize file(ARCHIVE_EXTRACT).

# cmake-lint: disable=E1126

# Wheel-only OpenSSL dependency. Extract archives and matching headers without conda, shared
# libraries, activation scripts, or build-prefix relocation.
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "Bundled OpenSSL is supported only on Linux")
endif()
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
  set(openssl_subdir linux-64)
  set(openssl_static_build h7cc23a3_0)
  set(openssl_headers_build h781a0a9_0)
  set(openssl_static_hash 33210cf94666ce96adba8c3568391d5670ce76d18412fc3c8ee35ceee8852878)
  set(openssl_headers_hash 9c1533f1cf784f0e250919c0c9762d0b0e6bc727a79290ffb7cea1ceca1d038c)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
  set(openssl_subdir linux-aarch64)
  set(openssl_static_build h5bc82ec_0)
  set(openssl_headers_build he6ad1d5_0)
  set(openssl_static_hash 88acb143c231a621a7ef18dc69294cdc7078acd42d38a339d78621dbd3bab281)
  set(openssl_headers_hash 238136f1df197ecd89bafd728a34d0642176f3170423d13d8276746d675a9485)
else()
  message(FATAL_ERROR "No bundled OpenSSL package for ${CMAKE_SYSTEM_PROCESSOR}")
endif()

set(openssl_version 3.6.5)
set(openssl_root "${CMAKE_BINARY_DIR}/_deps/openssl-${openssl_version}-${openssl_subdir}")
foreach(component IN ITEMS static headers)
  if(component STREQUAL "static")
    set(package libopenssl-static)
    set(package_files "lib/libcrypto.a" "lib/libssl.a")
  else()
    set(package openssl)
    set(package_files "include/*")
  endif()
  set(package_id "${package}-${openssl_version}-${openssl_${component}_build}")
  set(package_dir "${openssl_root}/${package_id}")
  file(MAKE_DIRECTORY "${package_dir}")
  file(
    DOWNLOAD "https://conda.anaconda.org/conda-forge/${openssl_subdir}/${package_id}.conda"
    "${package_dir}/package.conda"
    EXPECTED_HASH "SHA256=${openssl_${component}_hash}"
    TLS_VERIFY ON
  )
  file(ARCHIVE_EXTRACT INPUT "${package_dir}/package.conda" DESTINATION "${package_dir}" PATTERNS
       "pkg-*.tar.zst" "info-*.tar.zst"
  )
  file(ARCHIVE_EXTRACT INPUT "${package_dir}/pkg-${package_id}.tar.zst" DESTINATION
       "${openssl_root}" PATTERNS ${package_files}
  )
  if(component STREQUAL "static")
    file(ARCHIVE_EXTRACT INPUT "${package_dir}/info-${package_id}.tar.zst" DESTINATION
         "${package_dir}" PATTERNS "info/licenses/*"
    )
    install(FILES "${package_dir}/info/licenses/LICENSE.txt"
            DESTINATION "share/licenses/kvikio/openssl"
    )
  endif()
endforeach()

# Explicit paths avoid an older system OpenSSL or stale CMake cache entries.
set(OPENSSL_USE_STATIC_LIBS ON)
set(OPENSSL_INCLUDE_DIR "${openssl_root}/include")
set(OPENSSL_SSL_LIBRARY "${openssl_root}/lib/libssl.a")
set(OPENSSL_CRYPTO_LIBRARY "${openssl_root}/lib/libcrypto.a")
find_package(OpenSSL ${openssl_version} EXACT REQUIRED)
