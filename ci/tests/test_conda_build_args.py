# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the conda recipe's build.sh invocation without a CUDA toolchain."""

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("prefix_name", ["host_env", "host_env_" + "placehold_" * 20])
def test_conda_curl_arguments_reach_cmake(tmp_path, prefix_name):
    recipe = (ROOT / "conda/recipes/libkvikio/recipe.yaml").read_text()
    command = next(
        line.strip()
        for line in recipe.splitlines()
        if line.strip().startswith("./build.sh ")
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    cmake = bin_dir / "cmake"
    cmake.write_text('#!/bin/sh\nprintf "%s\\0" "$@" >> "$CMAKE_ARGUMENT_LOG"\n')
    cmake.chmod(0o755)
    argument_log = tmp_path / "cmake-arguments"
    prefix = tmp_path / prefix_name
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "PREFIX": str(prefix),
        "INSTALL_PREFIX": str(prefix),
        "LIBKVIKIO_BUILD_DIR": str(tmp_path / "build"),
        "RAPIDS_CUDA_VERSION": "12.9",
        "CMAKE_ARGUMENT_LOG": str(argument_log),
    }
    result = subprocess.run(
        ["bash", "-c", command], cwd=ROOT, env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    arguments = argument_log.read_bytes().decode().split("\0")
    assert "-DKvikIO_USE_SYSTEM_CURL=ON" in arguments
    assert f"-DCURL_LIBRARY={prefix}/lib/libcurl.so" in arguments
    assert f"-DCURL_INCLUDE_DIR={prefix}/include" in arguments
    assert "-DKvikIO_BUILD_BENCHMARKS=ON" in arguments
    assert "--build" in arguments
    assert "--target" not in arguments  # The recipe's -n must still disable install.
