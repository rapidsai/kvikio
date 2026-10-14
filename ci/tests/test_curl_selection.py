# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check curl selection without downloading dependencies or requiring CUDA."""

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "conda,override,installed,static_openssl,expected",
    [
        (True, None, True, False, "installed"),
        (True, None, True, True, "installed"),
        (True, None, False, False, "failure"),
        (True, None, False, True, "failure"),
        (False, None, False, False, "cpm"),
        (True, "OFF", False, False, "cpm"),
        (False, "ON", True, False, "installed"),
    ],
)
def test_curl_selection(tmp_path, conda, override, installed, static_openssl, expected):
    # Exercise the actual options after RAPIDS creates its conda environment target.
    source = (ROOT / "cpp/CMakeLists.txt").read_text()
    options = source.split(
        "rapids_cmake_support_conda_env(conda_env MODIFY_PREFIX_PATH)\n", 1
    )[1].split("rapids_cpm_init()", 1)[0]
    modules = tmp_path / "modules"
    modules.mkdir()
    (modules / "FindCURL.cmake").write_text(
        """if(NOT CURL_FIND_REQUIRED OR NOT CURL_FIND_VERSION STREQUAL "8.13.0")
  message(FATAL_ERROR "Expected required curl with the conda version floor")
endif()
if(CURL_USE_STATIC_LIBS)
  message(FATAL_ERROR "Conda curl must use shared libraries")
endif()
add_library(CURL::libcurl INTERFACE IMPORTED)
file(WRITE "${CMAKE_BINARY_DIR}/selection" "installed")
"""
        if installed
        else 'message(FATAL_ERROR "No installed curl")\n'
    )
    cpm = tmp_path / "cpm"
    cpm.mkdir()
    (cpm / "find.cmake").write_text(
        """function(rapids_cpm_find)
  file(WRITE "${CMAKE_BINARY_DIR}/selection" "cpm")
endfunction()
"""
    )
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 4.0)\nproject(CurlSelection NONE)\n"
        + ("add_library(conda_env INTERFACE)\n" if conda else "")
        + options
        + f'\nset(CMAKE_MODULE_PATH "{modules}")\n'
        + f'set(rapids-cmake-dir "{tmp_path}")\n'
        + f'include("{ROOT}/cpp/cmake/thirdparty/get_libcurl.cmake")\n'
    )
    build = tmp_path / "build"
    command = ["cmake", "-S", str(tmp_path), "-B", str(build)]
    if static_openssl:
        command.append("-DKvikIO_USE_STATIC_OPENSSL=ON")
    if override is not None:
        command.append(f"-DKvikIO_USE_SYSTEM_CURL={override}")
    result = subprocess.run(command, capture_output=True, text=True)
    if expected == "failure":
        assert result.returncode != 0
        assert "No installed curl" in result.stdout + result.stderr
        assert not (build / "selection").exists()
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert (build / "selection").read_text() == expected
