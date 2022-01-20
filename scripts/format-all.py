#!/usr/bin/env python3
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.


import os.path
import subprocess
from typing import Iterable


def root():
    return os.path.realpath(f"{os.path.dirname(os.path.realpath(__file__))}/..")


def run_cmd(cmd: Iterable, cwd=root(), verbose=True):
    res: subprocess.CompletedProcess = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=cwd
    )
    if verbose:
        print(f"{cwd}$ " + " ".join(res.args))
        print(res.stdout.decode(), end="")


def main():
    # C++
    run_cmd(["python", "scripts/run-clang-format.py", "cpp", "-inplace"])

    # Python
    python_root = f"{root()}/python"
    run_cmd(["isort", "."], cwd=python_root)
    run_cmd(["black", "."], cwd=python_root)
    run_cmd(["flake8", "--config=.flake8"], cwd=python_root)
    run_cmd(["flake8", "--config=.flake8.cython"], cwd=python_root)
    run_cmd(
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


if __name__ == "__main__":
    main()
