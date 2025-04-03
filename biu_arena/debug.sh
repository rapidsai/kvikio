#!/usr/bin/env bash

# export TMPDIR=/home/coder/kvikio/debug
export TMPDIR=/mnt/nvme_ubuntu_test

export KVIKIO_COMPAT_MODE=AUTO
test_bin=/home/coder/kvikio/cpp/build/latest/gtests/cpp_tests

gdb -ex start --args $test_bin
