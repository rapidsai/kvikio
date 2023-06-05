# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

from setuptools import find_packages
from skbuild import setup

setup(
    name="kvikio",
    version="23.08.00",
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
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    # Include the separately-compiled shared library
    extras_require={"test": ["pytest", "pytest-cov"]},
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
