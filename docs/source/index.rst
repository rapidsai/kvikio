Welcome to KvikIO's Python documentation!
=========================================

KvikIO is a Python and C++ library for high performance file IO. It provides C++ and Python
bindings to `cuFile <https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html>`_,
which enables `GPUDirect Storage <https://developer.nvidia.com/blog/gpudirect-storage/>`_ (GDS).
KvikIO also works efficiently when GDS isn't available and can read/write both host and device data seamlessly.

KvikIO is a part of the `RAPIDS <https://rapids.ai/>`_ suite of open-source software libraries for GPU-accelerated data science.


.. note::
   This is the documentation for the Python library. For the C++ documentation, see under `libkvikio <https://docs.rapids.ai/api/libkvikio/nightly/>`_.


Contents
--------

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   install
   quickstart
   zarr
   runtime_settings
   api
   genindex
