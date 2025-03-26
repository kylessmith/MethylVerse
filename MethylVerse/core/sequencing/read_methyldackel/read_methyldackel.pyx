#cython: embedsignature=True
#cython: profile=False
#cython: language_level=3

import os
cimport cython
from libc.stdint cimport uint32_t, uint8_t, uint64_t, int64_t
from ailist.LabeledIntervalArray_core cimport LabeledIntervalArray, labeled_aiarray_t, labeled_aiarray_init, labeled_aiarray_add
from intervalframe import IntervalFrame
import numpy as np
cimport numpy as np
np.import_array()

import pandas as pd


DTYPE = np.double
ctypedef np.double_t DTYPE_t


cdef np.ndarray pointer_to_double_numpy_array(void *ptr, np.npy_intp size):
    """
    Convert c pointer to numpy array.
    The memory will be freed as soon as the ndarray is deallocated.

    Parameters
    ----------
        ptr : void
            Pointer to be given to numpy
        size : np.npy_intp
            Size of the array

    Returns
    -------
        arr : numpy.ndarray
            Numpy array from given pointer

    """

    # Import functions for numpy C header
    cdef extern from "numpy/arrayobject.h":
        void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)

    # Create shape of ndarray
    cdef np.npy_intp dims[1]
    dims[0] = size

    # Create ndarray from C pointer
    cdef np.ndarray arr = np.PyArray_SimpleNewFromData(1, &dims[0], np.NPY_DOUBLE, ptr)

    # Hand control of data freeing to numpy
    PyArray_ENABLEFLAGS(arr, np.NPY_ARRAY_OWNDATA)
    #np.PyArray_UpdateFlags(arr, arr.flags.num | np.NPY_OWNDATA)

    return arr


cdef labeled_aiarray_t *_read_methyldackel(const char *file_names, int array_length, int str_len):
    cdef labeled_aiarray_t *laia = find_all_intervals(&file_names[0], array_length, str_len)

    return laia


cpdef _extract_sparse_betas(const char *file_name, LabeledIntervalArray laia):
    #cdef double[::1] betas = np.repeat(np.nan, laia.size)
    #cdef double[::1] betas = np.zeros(laia.size)
    cdef sparse_record_t *sp_betas = insert_sparse_betas(file_name, laia.laia)

    # Create numpy array from C pointer
    cdef np.ndarray indices = pointer_to_numpy_array(sp_betas.indices, sp_betas.size)
    cdef np.ndarray values = pointer_to_double_numpy_array(sp_betas.values, sp_betas.size)

    return indices, values


cpdef _extract_sparse_covs(const char *file_name, LabeledIntervalArray laia):
    #cdef double[::1] betas = np.repeat(np.nan, laia.size)
    #cdef double[::1] betas = np.zeros(laia.size)
    cdef sparse_record_t *sp_covs = insert_sparse_coverage(file_name, laia.laia)

    # Create numpy array from C pointer
    cdef np.ndarray indices = pointer_to_numpy_array(sp_covs.indices, sp_covs.size)
    cdef np.ndarray values = pointer_to_double_numpy_array(sp_covs.values, sp_covs.size)

    return indices, values


def read_methyldackel(file_names: np.ndarray,
                             use_sparse: bool = True,
                             read_coverage: bool = False) -> IntervalFrame:
    """
    Read methyldackel files and return a sparse IntervalFrame of beta values.

    Parameters
    ----------
        file_names : np.ndarray
            Array of file names
    
    Returns
    -------
        iframe : IntervalFrame
            IntervalFrame with sparse data
    """

    # Find sample name
    meth_names = np.array([os.path.split(file)[-1].split(".bedGraph")[0] for file in file_names])
    # Remove CpG string
    meth_names = np.array([name.split("_CpG")[0] for name in meth_names])

    # Find all unique intervals
    cdef int array_length = len(file_names)
    cdef np.ndarray byte_labels = file_names.astype(bytes)
    cdef labeled_aiarray_t *laia = _read_methyldackel(np.PyArray_BYTES(byte_labels), array_length, byte_labels.itemsize)
    intervals = LabeledIntervalArray()
    intervals.set_list(laia)

    # Sort intervals
    intervals.construct()
    intervals.sort()
    
    # Initialize dataframe
    df = pd.DataFrame([], index = range(laia.total_nr))

    # Static type arrays
    cdef np.ndarray[np.int_t, ndim = 1] sorted_indices
    cdef np.ndarray[np.int_t, ndim = 1] sp_indices
    cdef np.ndarray[DTYPE_t, ndim = 1] sp_values

    # Iterate over files
    cdef bytes file_name
    cdef int i
    for i, file in enumerate(file_names):

        # Extract sparse betas
        file_name = file.encode('utf-8')
        if read_coverage:
            sp_indices, sp_values = _extract_sparse_covs(file_name, intervals)
        else:
            sp_indices, sp_values = _extract_sparse_betas(file_name, intervals)

        if use_sparse:
            # Sort sparse betas
            sorted_indices = np.argsort(sp_indices)
            sp_indices = sp_indices[sorted_indices]
            sp_values = sp_values[sorted_indices]

            # Construct sparse array (Requires increasing indices)
            sp_index = pd._libs.sparse.IntIndex(intervals.size, sp_indices)
            df[meth_names[i]] = pd.arrays.SparseArray(data=sp_values, sparse_index=sp_index, fill_value=np.nan, dtype=np.float64)
        else:
            df[meth_names[i]] = np.zeros(intervals.size)
            df[meth_names[i]].values[:] = np.nan
            df[meth_names[i]].values[sp_indices] = sp_values

    # Construct interval frame
    iframe = IntervalFrame(intervals, df)

    return iframe

