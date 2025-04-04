#!/usr/bin/env bash

g++ -Wall -g -I /usr/local/cuda/include/ \
-I /usr/local/cuda/targets/sbsa-linux/include/ \
-std=c++17 /home/coder/kvikio/cpp/tests/biu.cpp /usr/local/cuda/targets/sbsa-linux/include/cufile.h \
-o biu_check_good -L /usr/local/cuda/targets/sbsa-linux/lib/ \
-lcufile -L /usr/local/cuda/lib64/stubs -lcuda \
-L /usr/local/cuda/lib64/ -lrt -lpthread -ldl

export CUFILE_ALLOW_COMPAT_MODE=false
export CUFILE_FORCE_COMPAT_MODE=false
export CUFILE_ENV_PATH_JSON=my_cufile.json

test_bin=./biu_check_good

# gdb -ex start --args $test_bin
# valgrind $test_bin
$test_bin
