#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# kvikio build script

# This script is used to build the component(s) in this repo from
# source, and can be called with various options to customize the
# build as needed (see the help output for details)

# Abort script on first error
set -e

NUMARGS=$#
ARGS=$*

# NOTE: ensure all dir changes are relative to the location of this
# script, and that this script resides in the repo dir!
REPODIR=$(cd "$(dirname "$0")"; pwd)

VALIDARGS="clean libkvikio kvikio -v -g -n --pydevelop -h"
HELP="$0 [clean] [libkvikio] [kvikio] [-v] [-g] [-n] [--cmake-args=\"<args>\"] [-h]
   clean                       - remove all existing build artifacts and configuration (start over)
   libkvikio                   - build and install the libkvikio C++ code
   kvikio                      - build and install the kvikio Python package (requires libkvikio)
   -v                          - verbose build mode
   -g                          - build for debug
   -n                          - no install step
   --pydevelop                 - Install Python packages in editable mode
   --cmake-args=\\\"<args>\\\" - pass arbitrary list of CMake configuration options (escape all quotes in argument)
   -h                          - print this text
   default action (no args) is to build and install 'libkvikio' and 'kvikio' targets
"
LIBKVIKIO_BUILD_DIR=${LIBKVIKIO_BUILD_DIR:=${REPODIR}/cpp/build}
KVIKIO_BUILD_DIR="${REPODIR}/python/kvikio/build/"
BUILD_DIRS="${LIBKVIKIO_BUILD_DIR} ${KVIKIO_BUILD_DIR}"

# Set defaults for vars modified by flags to this script
VERBOSE_FLAG=""
BUILD_TYPE=Release
INSTALL_TARGET=install
RAN_CMAKE=0
PYTHON_ARGS_FOR_INSTALL=("-v" "--no-build-isolation" "--no-deps" "--config-settings" "rapidsai.disable-cuda=true")


# Set defaults for vars that may not have been defined externally
# If INSTALL_PREFIX is not set, check PREFIX, then check
# CONDA_PREFIX, then fall back to install inside of $LIBKVIKIO_BUILD_DIR
INSTALL_PREFIX=${INSTALL_PREFIX:=${PREFIX:=${CONDA_PREFIX:=$LIBKVIKIO_BUILD_DIR/install}}}
export PARALLEL_LEVEL=${PARALLEL_LEVEL:-4}

function hasArg {
    (( NUMARGS != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

function cmakeArgs {
    # Check for multiple cmake args options
    if [[ $(echo "$ARGS" | { grep -Eo "\-\-cmake\-args" || true; } | wc -l ) -gt 1 ]]; then
        echo "Multiple --cmake-args options were provided, please provide only one: ${ARGS}"
        exit 1
    fi

    # Check for cmake args option
    if [[ -n $(echo "$ARGS" | { grep -E "\-\-cmake\-args" || true; } ) ]]; then
        # There are possible weird edge cases that may cause this regex filter to output nothing and fail silently
        # the true pipe will catch any weird edge cases that may happen and will cause the program to fall back
        # on the invalid option error
        EXTRA_CMAKE_ARGS=$(echo "$ARGS" | { grep -Eo "\-\-cmake\-args=\".+\"" || true; })
        if [[ -n ${EXTRA_CMAKE_ARGS} ]]; then
            # Remove the full  EXTRA_CMAKE_ARGS argument from list of args so that it passes validArgs function
            ARGS=${ARGS//$EXTRA_CMAKE_ARGS/}
            # Filter the full argument down to just the extra string that will be added to cmake call
            EXTRA_CMAKE_ARGS=$(echo "$EXTRA_CMAKE_ARGS" | grep -Eo "\".+\"" | sed -e 's/^"//' -e 's/"$//')
        fi
    fi
    read -ra EXTRA_CMAKE_ARGS <<< "$EXTRA_CMAKE_ARGS"
}


# Runs cmake if it has not been run already for build directory
# LIBKVIKIO_BUILD_DIR
function ensureCMakeRan {
    mkdir -p "${LIBKVIKIO_BUILD_DIR}"
    cd "${REPODIR}"/cpp
    if (( RAN_CMAKE == 0 )); then
        echo "Executing cmake for libkvikio..."
        cmake -B "${LIBKVIKIO_BUILD_DIR}" -S . \
              -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
              -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
              "${EXTRA_CMAKE_ARGS[@]}"
        RAN_CMAKE=1
    fi
}

if hasArg -h || hasArg --help; then
    echo "${HELP}"
    exit 0
fi

# Check for valid usage
if (( NUMARGS != 0 )); then
    # Check for cmake args
    cmakeArgs
    for a in ${ARGS}; do
    if ! (echo " ${VALIDARGS} " | grep -q " ${a} "); then
        echo "Invalid option or formatting, check --help: ${a}"
        exit 1
    fi
    done
fi

# Process flags
if hasArg -v; then
    VERBOSE_FLAG=-v
    export SKBUILD_BUILD_VERBOSE=true
    export SKBUILD_LOGGING_LEVEL=INFO
    set -x
fi
if hasArg -g; then
    BUILD_TYPE=Debug
    export SKBUILD_INSTALL_STRIP=false
    export SKBUILD_CMAKE_BUILD_TYPE=Debug
fi
if hasArg -n; then
    INSTALL_TARGET=""
fi
if hasArg --pydevelop; then
    PYTHON_ARGS_FOR_INSTALL+=("-e")
fi

# If clean given, run it prior to any other steps
if hasArg clean; then
    # If the dirs to clean are mounted dirs in a container, the
    # contents should be removed but the mounted dirs will remain.
    # The find removes all contents but leaves the dirs, the rmdir
    # attempts to remove the dirs but can fail safely.
    for bd in ${BUILD_DIRS}; do
        if [ -d "${bd}" ]; then
            find "${bd}" -mindepth 1 -delete
            rmdir "${bd}" || true
        fi
    done
fi

################################################################################
# Configure, build, and install libkvikio
if (( NUMARGS == 0 )) || hasArg libkvikio; then
    ensureCMakeRan
    echo "building libkvikio..."
    cmake --build "${LIBKVIKIO_BUILD_DIR}" -j"${PARALLEL_LEVEL}" ${VERBOSE_FLAG}
    if [[ ${INSTALL_TARGET} != "" ]]; then
        echo "installing libkvikio..."
        cmake --build "${LIBKVIKIO_BUILD_DIR}" --target install ${VERBOSE_FLAG}
    fi
fi

# Build and install the kvikio Python package
if (( NUMARGS == 0 )) || hasArg kvikio; then
    echo "building kvikio..."
    cd "${REPODIR}"/python/kvikio
    _EXTRA_CMAKE_ARGS=$(IFS=';'; echo "${EXTRA_CMAKE_ARGS[*]}")
    SKBUILD_CMAKE_ARGS="-DCMAKE_PREFIX_PATH=${INSTALL_PREFIX};-DCMAKE_LIBRARY_PATH=${LIBKVIKIO_BUILD_DIR};$_EXTRA_CMAKE_ARGS" \
        python -m pip install "${PYTHON_ARGS_FOR_INSTALL[@]}" .
fi
