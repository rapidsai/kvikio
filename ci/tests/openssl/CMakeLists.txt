# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

cmake_minimum_required(VERSION 4.0)
project(OpenSSLConfigProbe LANGUAGES C CXX)
include(../../../cmake/rapids_config.cmake)
include(rapids-cpm)
rapids_cpm_init()
include(../../../cpp/cmake/thirdparty/get_openssl.cmake)
add_executable(openssl_config_probe probe.cpp ../../../cpp/src/detail/openssl.cpp)
target_compile_features(openssl_config_probe PRIVATE cxx_std_17)
target_link_libraries(openssl_config_probe PRIVATE OpenSSL::SSL)
target_link_options(openssl_config_probe PRIVATE "LINKER:--exclude-libs,ALL")
