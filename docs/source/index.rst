NVIDIA KvikIO Documentation
===========================

NVIDIA KvikIO is a Python and C++ library for high performance file IO. It provides C++ and Python
bindings to `cuFile <https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html>`_,
which enables `GPUDirect Storage <https://developer.nvidia.com/blog/gpudirect-storage/>`_ (GDS).
KvikIO also works efficiently when GDS isn't available and can read/write both host and device data seamlessly.

KvikIO is a part of the `NVIDIA CUDA-X libraries for data science <https://developer.nvidia.com/topics/ai/data-science/cuda-x-for-data-science>`_, an open-source suite of GPU-accelerated libraries.


Contents
========

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   install
   quickstart
   zarr
   remote_file
   statistics
   runtime_settings
   profiling
   python/index
   cpp/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
