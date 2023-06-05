# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.


import os
from pathlib import Path

from setuptools import find_packages
from skbuild import setup

import legate.install_info as lg_install_info

legate_dir = Path(lg_install_info.libpath).parent.as_posix()

cmake_flags = [
    f"-Dlegate_core_ROOT:STRING={legate_dir}",
]

os.environ["SKBUILD_CONFIGURE_OPTIONS"] = " ".join(cmake_flags)


setup(
    packages=find_packages(exclude=["tests*"]),
    include_package_data=True,
    zip_safe=False,
)
