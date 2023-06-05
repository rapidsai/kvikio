# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

from setuptools import find_packages
from skbuild import setup

setup(
    packages=find_packages(exclude=["tests*"]),
    package_data={
        # Note: A dict comprehension with an explicit copy is necessary (rather
        # than something simpler like a dict.fromkeys) because otherwise every
        # package will refer to the same list and skbuild modifies it in place.
        key: ["*.pyi", "*.pxd"]
        for key in find_packages(include=["kvikio._lib"])
    },
    zip_safe=False,
)
