# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
PREPARE_SCRIPT = ROOT / "ci/prepare-openssl-prefix.sh"
OPENSSL_SHA256 = "603f5602e2eef00d77fbd429d34dcd5822bb301757a1bc9cdb24c670f1eb859a"


def source_prepare_script(tmp_path, *, openssl_root=None, cmake_args=None, conda_prefix=None):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    curl = fake_bin / "curl"
    curl.write_text("#!/bin/bash\nexit 99\n")
    curl.chmod(0o755)

    env = {**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}"}
    env.pop("OPENSSL_ROOT_DIR", None)
    if openssl_root is not None:
        env["OPENSSL_ROOT_DIR"] = str(openssl_root)
    if cmake_args is not None:
        env["CMAKE_ARGS"] = cmake_args
    if conda_prefix is not None:
        env["CONDA_PREFIX"] = str(conda_prefix)

    result = subprocess.run(
        [
            "bash",
            "-c",
            f'source "{PREPARE_SCRIPT}" || exit $?; printf "%s\\n" "${{OPENSSL_ROOT_DIR:-}}" "${{CMAKE_ARGS:-}}" "${{LD_LIBRARY_PATH:-}}"',
        ],
        check=False,
        capture_output=True,
        cwd=ROOT,
        env=env,
        text=True,
    )
    return result


def test_prepare_script_honors_explicit_openssl_root(tmp_path):
    openssl_root = tmp_path / "caller-openssl"
    (openssl_root / "lib").mkdir(parents=True)
    result = source_prepare_script(
        tmp_path, openssl_root=openssl_root, cmake_args="-DCMAKE_BUILD_TYPE=Release"
    )

    assert result.returncode == 0
    root, cmake_args, library_path = result.stdout.splitlines()
    assert root == str(openssl_root)
    assert cmake_args == f"-DCMAKE_BUILD_TYPE=Release -DOPENSSL_ROOT_DIR:PATH={openssl_root}"
    assert str(openssl_root / "lib") in library_path.split(":")


@pytest.mark.parametrize("root_argument", ["-DOPENSSL_ROOT_DIR=", "-DOPENSSL_ROOT_DIR:PATH="])
def test_prepare_script_preserves_explicit_cmake_openssl_root(tmp_path, root_argument):
    cmake_args = f"-DCMAKE_BUILD_TYPE=Release {root_argument}{tmp_path}/caller-openssl"
    result = source_prepare_script(tmp_path, cmake_args=cmake_args)

    assert result.returncode == 0
    assert result.stdout.splitlines()[1] == cmake_args


@pytest.mark.parametrize("version_define", ["#define OPENSSL_VERSION_MAJOR 3\n", "# define OPENSSL_VERSION_MAJOR  3\n"])
def test_prepare_script_reuses_valid_conda_openssl(tmp_path, version_define):
    conda_prefix = tmp_path / "conda"
    include = conda_prefix / "include/openssl"
    library = conda_prefix / "lib"
    include.mkdir(parents=True)
    library.mkdir(parents=True)
    (include / "opensslv.h").write_text(version_define)
    (library / "libssl.so").touch()
    (library / "libcrypto.so").touch()

    result = source_prepare_script(tmp_path, conda_prefix=conda_prefix)

    assert result.returncode == 0
    root, cmake_args, library_path = result.stdout.splitlines()
    assert root == str(conda_prefix)
    assert cmake_args == f"-DOPENSSL_ROOT_DIR:PATH={conda_prefix}"
    assert str(library) in library_path.split(":")


def test_prepare_script_does_not_reuse_conda_openssl_1(tmp_path):
    conda_prefix = tmp_path / "conda"
    include = conda_prefix / "include/openssl"
    library = conda_prefix / "lib"
    include.mkdir(parents=True)
    library.mkdir(parents=True)
    (include / "opensslv.h").write_text("#define OPENSSL_VERSION_MAJOR 1\n")
    (library / "libssl.so").touch()
    (library / "libcrypto.so").touch()

    result = source_prepare_script(tmp_path, conda_prefix=conda_prefix)

    assert result.returncode == 99
    assert "OPENSSL_ROOT_DIR" not in result.stdout


def test_prepare_script_pins_official_openssl_source_archive():
    script = PREPARE_SCRIPT.read_text()

    assert 'KVIKIO_OPENSSL_VERSION="3.5.9"' in script
    assert f'KVIKIO_OPENSSL_SHA256="{OPENSSL_SHA256}"' in script
    assert 'openssl-${KVIKIO_OPENSSL_VERSION}.tar.gz' in script
    assert "sha256sum --check --status" in script
