Installation
============

KvikIO can be installed using Conda/Mamba or from source.


Conda/Mamba
-----------

We strongly recommend using `mamba <https://github.com/mamba-org/mamba>`_ inplace of conda, which we will do throughout the documentation.

Install the **stable release** from the ``rapidsai`` channel like:

.. code-block::

  # Install in existing environment
  mamba install -c rapidsai -c conda-forge kvikio

  # Create new environment (CUDA 13)
  mamba create -n kvikio-env -c rapidsai -c conda-forge python=3.14 cuda-version=13.1 kvikio

  # Create new environment (CUDA 12)
  mamba create -n kvikio-env -c rapidsai -c conda-forge python=3.14 cuda-version=12.9 kvikio

Install the **nightly release** from the ``rapidsai-nightly`` channel like:

.. code-block::

  # Install in existing environment
  mamba install -c rapidsai-nightly -c conda-forge kvikio

  # Create new environment (CUDA 13)
  mamba create -n kvikio-env -c rapidsai-nightly -c conda-forge python=3.14 cuda-version=13.1 kvikio

  # Create new environment (CUDA 12)
  mamba create -n kvikio-env -c rapidsai-nightly -c conda-forge python=3.14 cuda-version=12.9 kvikio

.. note::

  If the nightly install doesn't work, set ``channel_priority: flexible`` in your ``.condarc``.


PyPI
----

KvikIO is also available on PyPI. Install the latest release like:

.. code-block::

  pip install kvikio-cu13 # for CUDA 13

.. code-block::

  pip install kvikio-cu12 # for CUDA 12

Build from source
-----------------

In order to setup a development environment, we recommend Conda:

.. code-block::

  # CUDA 13
  mamba env create --name kvikio-dev --file conda/environments/all_cuda-131_arch-$(arch).yaml

The Python library depends on the C++ library, thus we build and install both:

.. code-block::

  ./build.sh libkvikio kvikio


One might have to define ``CUDA_HOME`` to the path to the CUDA installation.

In order to test the installation, run the following:

.. code-block::

  pytest python/kvikio/tests/


And to test performance, run the following:

.. code-block::

  python python/kvikio/kvikio/benchmarks/single_node_io.py
