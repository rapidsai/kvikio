#!/usr/bin/env bash

rm -rf biu_check_bad_tmp
mkdir biu_check_bad_tmp

g++ -isystem /usr/local/cuda/targets/sbsa-linux/include -g -o biu_check_bad_tmp/biu.cpp.o -c biu.cpp

g++ -g biu_check_bad_tmp/biu.cpp.o -o biu_check_bad \
-Wl,-rpath,/usr/local/cuda-12.8/targets/sbsa-linux/lib \
-ldl

export CUFILE_ALLOW_COMPAT_MODE=false
export CUFILE_FORCE_COMPAT_MODE=false
export CUFILE_ENV_PATH_JSON=my_cufile.json

test_bin=./biu_check_bad

$test_bin