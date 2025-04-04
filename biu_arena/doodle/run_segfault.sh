#!/usr/bin/env bash

# build-kvikio-cpp -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON -j 16

# g++ -isystem /usr/local/cuda/targets/sbsa-linux/include -g -MD -MT biu.cpp.o -MF biu.cpp.o.d -o biu.cpp.o -c /home/coder/kvikio/cpp/tests/biu.cpp
# g++ -g biu.cpp.o -o biu_check  -Wl,-rpath,/usr/local/cuda-12.8/targets/sbsa-linux/lib  /usr/local/cuda-12.8/targets/sbsa-linux/lib/libcudart.so  -ldl  /usr/lib/aarch64-linux-gnu/librt.a

# g++ -Wall -g -I /usr/local/cuda/include/ -I /usr/local/cuda/targets/sbsa-linux/include/ -std=c++17 /home/coder/kvikio/cpp/tests/biu.cpp /usr/local/cuda/targets/sbsa-linux/include//cufile.h -o biu_check -L /usr/local/cuda/targets/sbsa-linux/lib/ -lcufile -L /usr/local/cuda/lib64/stubs -lcuda -Bstatic -L /usr/local/cuda/lib64/ -lcudart_static -lrt -lpthread -ldl -Bdynamic -lrt -ldl
export CUFILE_ALLOW_COMPAT_MODE=false
export CUFILE_FORCE_COMPAT_MODE=false
export CUFILE_ENV_PATH_JSON=my_cufile.json

# python segfault.py

# test_bin=/home/coder/kvikio/cpp/build/latest/gtests/cpp_tests
# test_bin=/home/coder/kvikio/cpp/build/latest/tests/biu_check
test_bin=./biu_check

gdb -ex start --args $test_bin
# valgrind $test_bin
# $test_bin
