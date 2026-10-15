#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This file is sourced by the libkvikio wheel build so OPENSSL_ROOT_DIR and
# CMAKE_ARGS remain available to scikit-build-core and auditwheel.

readonly KVIKIO_OPENSSL_VERSION="3.5.9"
readonly KVIKIO_OPENSSL_SHA256="603f5602e2eef00d77fbd429d34dcd5822bb301757a1bc9cdb24c670f1eb859a"
readonly KVIKIO_OPENSSL_SOURCE_URL="https://github.com/openssl/openssl/releases/download/openssl-${KVIKIO_OPENSSL_VERSION}/openssl-${KVIKIO_OPENSSL_VERSION}.tar.gz"

openssl_root_from_cmake_args=""
if [[ " ${CMAKE_ARGS:-} " =~ [[:space:]]-DOPENSSL_ROOT_DIR(:[[:alpha:]]+)?=([^[:space:]]+) ]]; then
  openssl_root_from_cmake_args="${BASH_REMATCH[2]}"
fi

if [[ -z "${OPENSSL_ROOT_DIR:-}" && -n "${openssl_root_from_cmake_args}" ]]; then
  OPENSSL_ROOT_DIR="${openssl_root_from_cmake_args}"
  export OPENSSL_ROOT_DIR
elif [[ -z "${OPENSSL_ROOT_DIR:-}" && -n "${CONDA_PREFIX:-}" ]]; then
  openssl_conda_header="${CONDA_PREFIX}/include/openssl/opensslv.h"
  if [[ -f "${openssl_conda_header}" ]]; then
    openssl_conda_major=$(awk '($1 == "#define" && $2 == "OPENSSL_VERSION_MAJOR") {print $3; exit} ($1 == "#" && $2 == "define" && $3 == "OPENSSL_VERSION_MAJOR") {print $4; exit}' "${openssl_conda_header}")
  else
    openssl_conda_major=""
  fi
  if [[ "${openssl_conda_major}" =~ ^[0-9]+$ ]] \
    && (( 10#${openssl_conda_major} >= 3 )) \
    && { [[ -e "${CONDA_PREFIX}/lib/libssl.so" ]] || [[ -e "${CONDA_PREFIX}/lib/libssl.a" ]]; } \
    && { [[ -e "${CONDA_PREFIX}/lib/libcrypto.so" ]] || [[ -e "${CONDA_PREFIX}/lib/libcrypto.a" ]]; }; then
    OPENSSL_ROOT_DIR="${CONDA_PREFIX}"
    export OPENSSL_ROOT_DIR
  fi
fi

if [[ -z "${OPENSSL_ROOT_DIR:-}" ]]; then
  # OpenSSL 3.5.9 Configure loads IPC::Cmd and Time::Piece even when tests are
  # disabled. The minimal Rocky wheel image may omit these separately packaged
  # modules; install only those prerequisites in the ephemeral wheel container
  # before building the private OpenSSL prefix.
  if ! perl -MIPC::Cmd -MTime::Piece -e 1 >/dev/null 2>&1; then
    if [[ "$(id -u)" != "0" ]]; then
      echo "OpenSSL source build requires Perl IPC::Cmd and Time::Piece, which are unavailable" >&2
      return 1
    fi

    if ! command -v dnf >/dev/null 2>&1; then
      echo "OpenSSL source build requires Perl IPC::Cmd and Time::Piece; install the distro modules before building" >&2
      return 1
    fi
    dnf install --nodocs --assumeyes perl-IPC-Cmd perl-Time-Piece || return $?

    if ! perl -MIPC::Cmd -MTime::Piece -e 1 >/dev/null 2>&1; then
      echo "Perl IPC::Cmd or Time::Piece remains unavailable after package installation" >&2
      return 1
    fi
  fi

  KVIKIO_OPENSSL_TEMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/kvikio-openssl-${KVIKIO_OPENSSL_VERSION}.XXXXXX")
  export KVIKIO_OPENSSL_TEMP_DIR

  openssl_archive="${KVIKIO_OPENSSL_TEMP_DIR}/openssl-${KVIKIO_OPENSSL_VERSION}.tar.gz"
  openssl_source="${KVIKIO_OPENSSL_TEMP_DIR}/openssl-${KVIKIO_OPENSSL_VERSION}"
  openssl_prefix="${KVIKIO_OPENSSL_TEMP_DIR}/prefix"

  curl --fail --location --retry 3 --silent --show-error \
    "${KVIKIO_OPENSSL_SOURCE_URL}" \
    --output "${openssl_archive}" || return $?
  printf '%s  %s\n' "${KVIKIO_OPENSSL_SHA256}" "${openssl_archive}" | sha256sum --check --status || return $?
  tar --extract --gzip --file "${openssl_archive}" --directory "${KVIKIO_OPENSSL_TEMP_DIR}" || return $?

  (
    cd "${openssl_source}" || exit
    ./Configure \
      --prefix="${openssl_prefix}" \
      --openssldir="${openssl_prefix}/ssl" \
      --libdir=lib \
      shared \
      no-tests
    make -j"$(nproc)"
    make install_sw
  ) || return $?

  OPENSSL_ROOT_DIR="${openssl_prefix}"
  export OPENSSL_ROOT_DIR
fi

if [[ -n "${OPENSSL_ROOT_DIR:-}" ]]; then
  if [[ -z "${openssl_root_from_cmake_args}" ]]; then
    CMAKE_ARGS="${CMAKE_ARGS:+${CMAKE_ARGS} }-DOPENSSL_ROOT_DIR:PATH=${OPENSSL_ROOT_DIR}"
    export CMAKE_ARGS
  fi

  for openssl_lib_dir in "${OPENSSL_ROOT_DIR}/lib" "${OPENSSL_ROOT_DIR}/lib64"; do
    if [[ -d "${openssl_lib_dir}" ]]; then
      LD_LIBRARY_PATH="${openssl_lib_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi
  done
  export LD_LIBRARY_PATH
fi
