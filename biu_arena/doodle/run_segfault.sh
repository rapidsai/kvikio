#!/usr/bin/env bash

# build-kvikio-cpp -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON -j 16

export CUFILE_ALLOW_COMPAT_MODE=true
export CUFILE_FORCE_COMPAT_MODE=true
export CUFILE_ENV_PATH_JSON=my_cufile.json

# python segfault.py

# test_bin=/home/coder/kvikio/cpp/build/latest/gtests/cpp_tests
test_bin=/home/coder/kvikio/cpp/build/latest/tests/biu_check

# gdb -ex start --args $test_bin
# valgrind $test_bin
$test_bin
