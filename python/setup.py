# Copyright (c) 2021-2022, NVIDIA CORPORATION.


import glob
import os
import shutil
import sysconfig

# Must import in this order:
#   setuptools -> Cython.Distutils.build_ext -> setuptools.command.build_ext
# Otherwise, setuptools.command.build_ext ends up inheriting from
# Cython.Distutils.old_build_ext which we do not want
import setuptools

try:
    from Cython.Distutils.build_ext import new_build_ext as _build_ext
except ImportError:
    from setuptools.command.build_ext import build_ext as _build_ext

import distutils.util
from distutils.sysconfig import get_python_lib

import Cython.Compiler.Options
import setuptools.command.build_ext
from setuptools import setup
from setuptools.extension import Extension

import versioneer

# Set `DEBUG_BUILD=true` to enable debug build
DEBUG_BUILD = distutils.util.strtobool(os.environ.get("DEBUG_BUILD", "false"))

# Turn all warnings into errors.
Cython.Compiler.Options.warning_errors = True

# Abort the compilation on the first error
Cython.Compiler.Options.fast_fail = True

install_requires = ["cython"]
this_setup_scrip_dir = os.path.dirname(os.path.realpath(__file__))

# Throughout the script, we will populate `include_dirs`,
# `library_dirs` and `depends`.
include_dirs = [os.path.dirname(sysconfig.get_path("include"))]
library_dirs = [get_python_lib()]
depends = []  # Files to trigger rebuild when modified

# Find and add CUDA include and binary paths
CUDA_HOME = os.environ.get("CUDA_HOME", False)
if not CUDA_HOME:
    path_to_cuda_gdb = shutil.which("cuda-gdb")
    if path_to_cuda_gdb is None:
        raise OSError(
            "Could not locate CUDA. "
            "Please set the environment variable "
            "CUDA_HOME to the path to the CUDA installation "
            "and try again."
        )
    CUDA_HOME = os.path.dirname(os.path.dirname(path_to_cuda_gdb))
if not os.path.isdir(CUDA_HOME):
    raise OSError(f"Invalid CUDA_HOME: directory does not exist: {CUDA_HOME}")
include_dirs.append(os.path.join(CUDA_HOME, "include"))
library_dirs.append(os.path.join(CUDA_HOME, "lib64"))

# Add cuFile++ headers from the source tree (if available)
cufilexx_include_dir = os.path.abspath(f"{this_setup_scrip_dir}/../cpp/include")
if os.path.isdir(cufilexx_include_dir):
    include_dirs = [cufilexx_include_dir] + include_dirs
    depends.extend(glob.glob(f"{cufilexx_include_dir}/cufile/*"))
    depends.extend(f"{this_setup_scrip_dir}/../cpp/CMakeLists.txt")

extensions = [
    Extension(
        "cufile",
        sources=["src/cufile.pyx"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=["cuda", "cudart", "cufile", "nvidia-ml"],
        language="c++",
        extra_compile_args=["-std=c++17"],
        depends=depends,
    ),
    Extension(
        "arr",
        sources=["src/arr.pyx"],
        language="c++",
        extra_compile_args=["-std=c++17"],
        depends=depends,
    ),
]


def remove_flags(compiler, *flags):
    for flag in flags:
        try:
            compiler.compiler_so = list(filter((flag).__ne__, compiler.compiler_so))
        except Exception:
            pass


class build_ext(_build_ext):
    def build_extensions(self):
        # No '-Wstrict-prototypes' warning
        remove_flags(self.compiler, "-Wstrict-prototypes")
        if not DEBUG_BUILD:
            # Full optimization
            self.compiler.compiler_so.append("-O3")
        if not DEBUG_BUILD:
            # No debug symbols
            remove_flags(self.compiler, "-g", "-G", "-O1", "-O2")
        super().build_extensions()

    def finalize_options(self):
        if self.distribution.ext_modules:
            # Delay import this to allow for Cython-less installs
            from Cython.Build.Dependencies import cythonize

            nthreads = getattr(self, "parallel", None)  # -j option in Py3.5+
            nthreads = int(nthreads) if nthreads else None
            self.distribution.ext_modules = cythonize(
                self.distribution.ext_modules,
                nthreads=nthreads,
                force=self.force,
                gdb_debug=False,
                compiler_directives=dict(
                    profile=False,
                    language_level=3,
                    embedsignature=True,
                    binding=(not DEBUG_BUILD),
                ),
            )
        # Skip calling super() and jump straight to setuptools
        setuptools.command.build_ext.build_ext.finalize_options(self)


cmdclass = dict()
cmdclass.update(versioneer.get_cmdclass())
cmdclass["build_ext"] = build_ext

setup(
    name="cufile",
    version=versioneer.get_version(),
    description="cuFile - GPUDirect Storage",
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
    setup_requires=["Cython>=0.29,<0.30"],
    extras_require={"test": ["pytest", "pytest-xdist"]},
    ext_modules=extensions,
    cmdclass=cmdclass,
    install_requires=install_requires,
    zip_safe=False,
)
