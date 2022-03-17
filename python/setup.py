# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

from setuptools import find_packages
from skbuild import setup

import versioneer

# Set `DEBUG_BUILD=true` to enable debug build
# DEBUG_BUILD = distutils.util.strtobool(os.environ.get("DEBUG_BUILD", "false"))

# Turn all warnings into errors.
# Cython.Compiler.Options.warning_errors = True

# Abort the compilation on the first error
# Cython.Compiler.Options.fast_fail = True

# install_requires = ["cython"]
# this_setup_scrip_dir = os.path.dirname(os.path.realpath(__file__))

# Throughout the script, we will populate `include_dirs`,
# `library_dirs` and `depends`.
# include_dirs = [os.path.dirname(sysconfig.get_path("include"))]
# library_dirs = [get_python_lib()]
# extra_objects = []
# depends = []  # Files to trigger rebuild when modified

# # Find and add CUDA include and binary paths
# CUDA_HOME = os.environ.get("CUDA_HOME", False)
# if not CUDA_HOME:
#     path_to_cuda_gdb = shutil.which("cuda-gdb")
#     if path_to_cuda_gdb is None:
#         raise OSError(
#             "Could not locate CUDA. "
#             "Please set the environment variable "
#             "CUDA_HOME to the path to the CUDA installation "
#             "and try again."
#         )
#     CUDA_HOME = os.path.dirname(os.path.dirname(path_to_cuda_gdb))
# if not os.path.isdir(CUDA_HOME):
#     raise OSError(f"Invalid CUDA_HOME: directory does not exist: {CUDA_HOME}")
# include_dirs.append(os.path.join(CUDA_HOME, "include"))
# library_dirs.append(os.path.join(CUDA_HOME, "lib64"))

# # Use CuFile location outside of CUDA_HOME
# if "CUFILE_HOME" in os.environ:
#     CUFILE_HOME = os.environ["CUFILE_HOME"]
#     if not os.path.isdir(CUDA_HOME):
#         raise OSError(f"Invalid CUFILE_HOME: directory does not exist: {CUFILE_HOME}")
#     include_dirs.append(os.path.join(CUFILE_HOME, "include"))
#     if os.path.isdir(os.path.join(CUFILE_HOME, "lib64")):
#         library_dirs.append(os.path.join(CUFILE_HOME, "lib64"))
#     else:
#         library_dirs.append(os.path.join(CUFILE_HOME, "lib"))
#
# # Add kvikio headers from the source tree (if available)
# kvikio_include_dir = os.path.abspath(f"{this_setup_scrip_dir}/../cpp/include")
# if os.path.isdir(kvikio_include_dir):
#     include_dirs = [kvikio_include_dir] + include_dirs
#     depends.extend(glob.glob(f"{kvikio_include_dir}/kvikio/*"))
#     depends.extend(f"{this_setup_scrip_dir}/../cpp/CMakeLists.txt")


setup(
    name="kvikio",
    version=versioneer.get_version(),
    description="KvikIO - GPUDirect Storage",
    url="https://github.com/rapidsai/cufile-bindings",
    author="NVIDIA Corporation",
    license="Apache 2.0",
    classifiers=[
        "Intended Audience :: Developers",
        "Topic :: Database",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    # Include the separately-compiled shared library
    extras_require={"test": ["pytest", "pytest-xdist"]},
    packages=find_packages(exclude=["tests*"]),
    package_data={
        # Note: A dict comprehension with an explicit copy is necessary (rather
        # than something simpler like a dict.fromkeys) because otherwise every
        # package will refer to the same list and skbuild modifies it in place.
        key: ["*.pyi", "*.pxd"]
        for key in find_packages(
            include=["kvikio._lib"]
        )
    },
    cmdclass=versioneer.get_cmdclass(),
    install_requires=["Cython>=0.29,<0.30"],
    zip_safe=False,
)
