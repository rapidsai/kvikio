#!/usr/bin/env python3
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.


import argparse
import functools
import glob
import os.path
import pathlib
import subprocess
import sys
import tempfile
import urllib.request
from typing import Iterable


def root():
    return os.path.realpath(f"{os.path.dirname(os.path.realpath(__file__))}/..")


def run_cmd(cmd: Iterable[str], cwd=root(), check=False):
    res: subprocess.CompletedProcess = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=cwd
    )
    print(f"{cwd}$ " + " ".join(res.args), end=" ")
    if check:
        if res.returncode:
            print("FAILED")
            print(res.stdout.decode(), end="")
        else:
            print("PASSED")
    else:
        print("\n" + res.stdout.decode(), end="")
    return res.returncode


def cmake_format_cmd(version="23.02"):
    # Find files
    files = []
    for root in ["cpp", "python"]:
        for p in [
            "CMakeLists.txt",
            "cmake/*.cmake",
            "cmake/Modules/*.cmake",
            "cmake/thirdparty/*.cmake",
        ]:
            files.extend(glob.glob(f"{root}/{p}"))

    # Find config
    config_file = pathlib.Path(tempfile.gettempdir()) / f"cmake-format-{version}.json"
    if not config_file.is_file():
        # Download from rapids-cmake
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-"
            f"{version}/cmake-format-rapids-cmake.json",
            str(config_file),
        )
    return ["cmake-format", "--config-files", str(config_file)] + files


def main(args):
    check = ["--check"] if args.check else []
    ret = 0
    # Set the `check` argument for all runs
    cmd = functools.partial(run_cmd, check=check)

    # C++
    inplace = [] if args.check else ["-inplace"]
    ret += cmd(["python", "scripts/run-clang-format.py", "cpp"] + inplace)
    inplace = check if args.check else ["--in-place"]
    ret += cmd(cmake_format_cmd() + inplace)

    # Python
    python_root = f"{root()}/python"
    ret += cmd(["isort", "scripts"] + check)
    ret += cmd(["isort", "."] + check, cwd=python_root)
    ret += cmd(["black", "scripts"] + check)
    ret += cmd(["black", "."] + check, cwd=python_root)
    ret += cmd(["flake8", "--config=.flake8"], cwd=python_root)
    ret += cmd(["flake8", "--config=.flake8.cython"], cwd=python_root)
    ret += cmd(
        [
            "mypy",
            "--ignore-missing-imports",
            "kvikio",
            "tests",
            "examples",
            "benchmarks",
        ],
        cwd=python_root,
    )
    return 1 if ret else 0


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Inplace style formatting of the whole project "
            "using isort, black, flake8, mypy, and clang-format"
        )
    )
    parser.add_argument(
        "--check",
        action="store_true",
        default=False,
        help=(
            "Don't write the files back, just return the status. "
            "Return code 0 on success and 1 on failure."
        ),
    )
    retcode = main(parser.parse_args())
    print(f"format-all.py: " + ("FAILED" if retcode else "ALL-PASSED"))
    sys.exit(retcode)
