cmake_minimum_required(VERSION 3.10)

project(CuFileJNI LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

find_package(JNI REQUIRED)

add_library(CuFileJNI SHARED
    src/CuFileJni.cpp
)

set_source_files_properties(src/CuFileJni.cpp PROPERTIES LANGUAGE CUDA)

target_include_directories(CuFileJNI PRIVATE
    ${CUDA_INCLUDE_DIRS}
    ${JAVA_INCLUDE_PATH}
    ${JAVA_INCLUDE_PATH2}
)

target_link_libraries(CuFileJNI cufile)