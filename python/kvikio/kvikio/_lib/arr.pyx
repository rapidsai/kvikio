# Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

# cython: language_level=3


from cpython.array cimport array, newarrayobject
from cpython.buffer cimport PyBuffer_IsContiguous
from cpython.memoryview cimport PyMemoryView_FromObject, PyMemoryView_GET_BUFFER
from cpython.object cimport PyObject
from cpython.ref cimport Py_INCREF
from cpython.tuple cimport PyTuple_New, PyTuple_SET_ITEM
from cython cimport auto_pickle, boundscheck, initializedcheck, nonecheck, wraparound
from libc.stdint cimport uintptr_t
from libc.string cimport memcpy

try:
    from numpy import dtype as numpy_dtype
except ImportError:
    numpy_dtype = None


cdef dict itemsize_mapping = {
    intern("|b1"): 1,
    intern("|i1"): 1,
    intern("|u1"): 1,
    intern("<i2"): 2,
    intern(">i2"): 2,
    intern("<u2"): 2,
    intern(">u2"): 2,
    intern("<i4"): 4,
    intern(">i4"): 4,
    intern("<u4"): 4,
    intern(">u4"): 4,
    intern("<i8"): 8,
    intern(">i8"): 8,
    intern("<u8"): 8,
    intern(">u8"): 8,
    intern("<f2"): 2,
    intern(">f2"): 2,
    intern("<f4"): 4,
    intern(">f4"): 4,
    intern("<f8"): 8,
    intern(">f8"): 8,
    intern("<f16"): 16,
    intern(">f16"): 16,
    intern("<c8"): 8,
    intern(">c8"): 8,
    intern("<c16"): 16,
    intern(">c16"): 16,
    intern("<c32"): 32,
    intern(">c32"): 32,
}


cdef array array_Py_ssize_t = array("q")


cdef inline Py_ssize_t[::1] new_Py_ssize_t_array(Py_ssize_t n):
    return newarrayobject(
        (<PyObject*>array_Py_ssize_t).ob_type, n, array_Py_ssize_t.ob_descr
    )


@auto_pickle(False)
cdef class Array:
    """ An efficient wrapper for host and device array-like objects

    Parameters
    ----------
    obj: Object exposing the buffer protocol or __cuda_array_interface__
        A host and device array-like object
    """
    def __cinit__(self, obj):
        cdef dict iface = getattr(obj, "__cuda_array_interface__", None)
        self.cuda = (iface is not None)
        cdef const Py_buffer* pybuf
        cdef str typestr
        cdef tuple data, shape, strides
        cdef Py_ssize_t i
        if self.cuda:
            if iface.get("mask") is not None:
                raise NotImplementedError("mask attribute not supported")

            self.obj = obj
            data = iface["data"]
            self.ptr, self.readonly = data

            typestr = iface["typestr"]
            if typestr is None:
                raise ValueError("Expected `str`, but got `None`")
            elif typestr == "":
                raise ValueError("Got unexpected empty `str`")
            else:
                try:
                    self.itemsize = itemsize_mapping[typestr]
                except KeyError:
                    if numpy_dtype is not None:
                        self.itemsize = numpy_dtype(typestr).itemsize
                    else:
                        raise ValueError(
                            f"Unexpected data type, '{typestr}'."
                            " Please install NumPy to handle this format."
                        )

            shape = iface["shape"]
            strides = iface.get("strides")
            self.ndim = len(shape)
            if self.ndim > 0:
                self.shape_mv = new_Py_ssize_t_array(self.ndim)
                for i in range(self.ndim):
                    self.shape_mv[i] = shape[i]
                if strides is not None:
                    if len(strides) != self.ndim:
                        raise ValueError(
                            "The length of shape and strides must be equal"
                        )
                    self.strides_mv = new_Py_ssize_t_array(self.ndim)
                    for i in range(self.ndim):
                        self.strides_mv[i] = strides[i]
                else:
                    self.strides_mv = None
            else:
                self.shape_mv = None
                self.strides_mv = None
        else:
            mv = PyMemoryView_FromObject(obj)
            pybuf = PyMemoryView_GET_BUFFER(mv)

            if pybuf.suboffsets != NULL:
                raise NotImplementedError("Suboffsets are not supported")

            self.ptr = <uintptr_t>pybuf.buf
            self.obj = pybuf.obj
            self.readonly = <bint>pybuf.readonly
            self.ndim = <Py_ssize_t>pybuf.ndim
            self.itemsize = <Py_ssize_t>pybuf.itemsize

            if self.ndim > 0:
                self.shape_mv = new_Py_ssize_t_array(self.ndim)
                memcpy(
                    &self.shape_mv[0],
                    pybuf.shape,
                    self.ndim * sizeof(Py_ssize_t)
                )
                if not PyBuffer_IsContiguous(pybuf, b"C"):
                    self.strides_mv = new_Py_ssize_t_array(self.ndim)
                    memcpy(
                        &self.strides_mv[0],
                        pybuf.strides,
                        self.ndim * sizeof(Py_ssize_t)
                    )
                else:
                    self.strides_mv = None
            else:
                self.shape_mv = None
                self.strides_mv = None

    cpdef bint _c_contiguous(self):
        return _c_contiguous(
            self.itemsize, self.ndim, self.shape_mv, self.strides_mv
        )

    @property
    def c_contiguous(self):
        return self._c_contiguous()

    cpdef bint _f_contiguous(self):
        return _f_contiguous(
            self.itemsize, self.ndim, self.shape_mv, self.strides_mv
        )

    @property
    def f_contiguous(self):
        return self._f_contiguous()

    cpdef bint _contiguous(self):
        return _contiguous(
            self.itemsize, self.ndim, self.shape_mv, self.strides_mv
        )

    @property
    def contiguous(self):
        return self._contiguous()

    cpdef Py_ssize_t _nbytes(self):
        return _nbytes(self.itemsize, self.ndim, self.shape_mv)

    @property
    def nbytes(self):
        return self._nbytes()

    @property
    @boundscheck(False)
    @initializedcheck(False)
    @nonecheck(False)
    @wraparound(False)
    def shape(self):
        cdef tuple shape = PyTuple_New(self.ndim)
        cdef Py_ssize_t i
        cdef object o
        for i in range(self.ndim):
            o = self.shape_mv[i]
            Py_INCREF(o)
            PyTuple_SET_ITEM(shape, i, o)
        return shape

    @property
    @boundscheck(False)
    @initializedcheck(False)
    @nonecheck(False)
    @wraparound(False)
    def strides(self):
        cdef tuple strides = PyTuple_New(self.ndim)
        cdef Py_ssize_t i, s
        cdef object o
        if self.strides_mv is not None:
            for i from self.ndim > i >= 0 by 1:
                o = self.strides_mv[i]
                Py_INCREF(o)
                PyTuple_SET_ITEM(strides, i, o)
        else:
            s = self.itemsize
            for i from self.ndim > i >= 0 by 1:
                o = s
                Py_INCREF(o)
                PyTuple_SET_ITEM(strides, i, o)
                s *= self.shape_mv[i]
        return strides


@boundscheck(False)
@initializedcheck(False)
@nonecheck(False)
@wraparound(False)
cdef inline bint _c_contiguous(Py_ssize_t itemsize,
                               Py_ssize_t ndim,
                               Py_ssize_t[::1] shape_mv,
                               Py_ssize_t[::1] strides_mv) noexcept nogil:
    cdef Py_ssize_t i, s
    if strides_mv is not None:
        s = itemsize
        for i from ndim > i >= 0 by 1:
            if s != strides_mv[i]:
                return False
            s *= shape_mv[i]
    return True


@boundscheck(False)
@initializedcheck(False)
@nonecheck(False)
@wraparound(False)
cdef inline bint _f_contiguous(Py_ssize_t itemsize,
                               Py_ssize_t ndim,
                               Py_ssize_t[::1] shape_mv,
                               Py_ssize_t[::1] strides_mv) noexcept nogil:
    cdef Py_ssize_t i, s
    if strides_mv is not None:
        s = itemsize
        for i from 0 <= i < ndim by 1:
            if s != strides_mv[i]:
                return False
            s *= shape_mv[i]
    elif ndim > 1:
        return False
    return True


cdef inline bint _contiguous(Py_ssize_t itemsize,
                             Py_ssize_t ndim,
                             Py_ssize_t[::1] shape_mv,
                             Py_ssize_t[::1] strides_mv) noexcept nogil:
    cdef bint r = _c_contiguous(itemsize, ndim, shape_mv, strides_mv)
    if not r:
        r = _f_contiguous(itemsize, ndim, shape_mv, strides_mv)
    return r


@boundscheck(False)
@initializedcheck(False)
@nonecheck(False)
@wraparound(False)
cdef inline Py_ssize_t _nbytes(Py_ssize_t itemsize,
                               Py_ssize_t ndim,
                               Py_ssize_t[::1] shape_mv) noexcept nogil:
    cdef Py_ssize_t i, nbytes = itemsize
    for i in range(ndim):
        nbytes *= shape_mv[i]
    return nbytes


cpdef Array asarray(obj):
    """Coerce other objects to ``Array``. No-op for existing ``Array``s.

    Args:
        obj: Object exposing the Python buffer protocol or ``__cuda_array_interface__``

    Returns:
        Array: An instance of the ``Array`` class
    """
    if isinstance(obj, Array):
        return <Array>obj
    else:
        return Array(obj)


cdef pair[uintptr_t, size_t] parse_buffer_argument(
    buf, size, bint accept_host_buffer
) except *:
    """Parse `buf` and `size` argument and return a pointer and nbytes"""
    if not isinstance(buf, Array):
        buf = Array(buf)
    cdef Array arr = buf
    if not arr._contiguous():
        raise ValueError("Array must be contiguous")
    if not accept_host_buffer and not arr.cuda:
        raise ValueError("Non-CUDA buffers not supported")
    cdef size_t nbytes
    if size is None:
        nbytes = arr.nbytes
    elif size > arr.nbytes:
        raise ValueError("Size is greater than the size of the buffer")
    else:
        nbytes = size
    return pair[uintptr_t, size_t](arr.ptr, nbytes)
