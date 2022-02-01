#!/usr/bin/env python3
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.


import argparse
import os.path
import subprocess
import sys
from typing import Iterable


def root():
    return os.path.realpath(f"{os.path.dirname(os.path.realpath(__file__))}/..")


def run_cmd(cmd: Iterable[str], cwd=root(), verbose=True):
    res: subprocess.CompletedProcess = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=cwd
    )
    if verbose:
        print(f"{cwd}$ " + " ".join(res.args))
        print(res.stdout.decode(), end="")
    return res.returncode


def main(args):
    # C++
    check = [] if args.check else ["-inplace"]
    run_cmd(["python", "scripts/run-clang-format.py", "cpp"] + check)

    # Python
    python_root = f"{root()}/python"
    check = ["--check"] if args.check else []
    ret = 0
    ret += run_cmd(["isort", "."] + check, cwd=python_root)
    ret += run_cmd(["black", "."] + check, cwd=python_root)
    ret += run_cmd(["flake8", "--config=.flake8"], cwd=python_root)
    ret += run_cmd(["flake8", "--config=.flake8.cython"], cwd=python_root)
    ret += run_cmd(
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
    sys.exit(retcode)
