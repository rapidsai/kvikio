#!/usr/bin/env bash

mkdir -p biu_check_bad_tmp

g++ -isystem /usr/local/cuda/targets/sbsa-linux/include \
-g -MD -MT biu_check_bad_tmp/biu.cpp.o -MF biu_check_bad_tmp/biu.cpp.o.d -o biu_check_bad_tmp/biu.cpp.o -c /home/coder/kvikio/cpp/tests/biu.cpp

g++ -g biu_check_bad_tmp/biu.cpp.o -o biu_check_bad \
-Wl,-rpath,/usr/local/cuda-12.8/targets/sbsa-linux/lib /usr/local/cuda-12.8/targets/sbsa-linux/lib/libcudart.so \
-ldl  /usr/lib/aarch64-linux-gnu/librt.a

export CUFILE_ALLOW_COMPAT_MODE=false
export CUFILE_FORCE_COMPAT_MODE=false
export CUFILE_ENV_PATH_JSON=my_cufile.json

test_bin=./biu_check_bad

# gdb -ex start --args $test_bin
# valgrind $test_bin
$test_bin
