#
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
#

cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

project(CuFileJNI LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

find_package(JNI REQUIRED)

add_library(CuFileJNI SHARED src/CuFileJni.cpp)

set_source_files_properties(src/CuFileJni.cpp PROPERTIES LANGUAGE CUDA)

target_include_directories(
  CuFileJNI PRIVATE ${CUDA_INCLUDE_DIRS} ${JAVA_INCLUDE_PATH} ${JAVA_INCLUDE_PATH2}
)

target_link_libraries(CuFileJNI cufile cuda)
