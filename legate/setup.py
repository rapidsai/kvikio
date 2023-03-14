# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.


import os
from pathlib import Path

import versioneer
from setuptools import find_packages
from skbuild import setup

import legate.install_info as lg_install_info

legate_dir = Path(lg_install_info.libpath).parent.as_posix()

cmake_flags = [
    f"-Dlegate_core_ROOT:STRING={legate_dir}",
]

os.environ["SKBUILD_CONFIGURE_OPTIONS"] = " ".join(cmake_flags)


setup(
    name="kvikio",
    version=versioneer.get_version(),
    description="KvikIO - GPUDirect Storage",
    url="https://github.com/rapidsai/kvikio",
    author="NVIDIA Corporation",
    license="Apache 2.0",
    classifiers=[
        "Intended Audience :: Developers",
        "Topic :: Database",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    # packages=find_packages(
    #     where=".",
    #     include=["hello", "hello.*"],
    # ),
    packages=find_packages(exclude=["tests*"]),
    include_package_data=True,
    cmdclass=versioneer.get_cmdclass(),
    zip_safe=False,
)
