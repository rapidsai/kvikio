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
  # Create new environment (CUDA 11.8)
  mamba create -n kvikio-env -c rapidsai -c conda-forge python=3.11 cuda-version=11.8 kvikio
  # Create new environment (CUDA 12.5)
  mamba create -n kvikio-env -c rapidsai -c conda-forge python=3.11 cuda-version=12.5 kvikio

Install the **nightly release** from the ``rapidsai-nightly`` channel like:

.. code-block::

  # Install in existing environment
  mamba install -c rapidsai-nightly -c conda-forge kvikio
  # Create new environment (CUDA 11.8)
  mamba create -n kvikio-env -c rapidsai-nightly -c conda-forge python=3.11 cuda-version=11.8 kvikio
  # Create new environment (CUDA 12.5)
  mamba create -n kvikio-env -c rapidsai-nightly -c conda-forge python=3.11 cuda-version=12.5 kvikio


.. note::

  If the nightly install doesn't work, set ``channel_priority: flexible`` in your ``.condarc``.

Build from source
-----------------

In order to setup a development environment run:

.. code-block::

  # CUDA 11.8
  mamba env create --name kvikio-dev --file conda/environments/all_cuda-118_arch-x86_64.yaml
  # CUDA 12.5
  mamba env create --name kvikio-dev --file conda/environments/all_cuda-125_arch-x86_64.yaml

To build and install the extension run:

.. code-block::

  ./build.sh kvikio


One might have to define ``CUDA_HOME`` to the path to the CUDA installation.

In order to test the installation, run the following:

.. code-block::

  pytest python/kvikio/tests/


And to test performance, run the following:

.. code-block::

  python python/kvikio/kvikio/benchmarks/single-node-io.py
