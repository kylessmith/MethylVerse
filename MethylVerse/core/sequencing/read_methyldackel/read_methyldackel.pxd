# cython: language_level=3

cimport cython
import numpy as np
cimport numpy as np

from ailist.array_query_core cimport pointer_to_numpy_array
from ailist.LabeledIntervalArray_core cimport LabeledIntervalArray, labeled_aiarray_t, labeled_aiarray_init, labeled_aiarray_add

from libc.stdint cimport uint32_t, uint8_t, uint64_t, int64_t, uint16_t, int32_t


cdef extern from "process_methyldackel.c":
    # C is include here so that it doesn't need to be compiled externally
    pass

cdef extern from "process_methyldackel.h":
    # C is include here so that it doesn't need to be compiled externally

    ctypedef struct sparse_record_t:
        int size							# Current size
        int max_size						# Maximum size
        double *values				        # Store values
        long *indices						# Store indices

    labeled_aiarray_t *find_all_intervals(const char file_names[], int length, int str_len) nogil

    void insert_betas(const char* fn, double *betas, labeled_aiarray_t *laia) nogil

    sparse_record_t *insert_sparse_betas(const char* fn, labeled_aiarray_t *laia) nogil

    sparse_record_t *insert_sparse_coverage(const char* fn, labeled_aiarray_t *laia) nogil


cdef np.ndarray pointer_to_double_numpy_array(void *ptr, np.npy_intp size)
cdef labeled_aiarray_t *_read_methyldackel(const char *file_names, int array_length, int str_len)
cpdef _extract_sparse_betas(const char *file_name, LabeledIntervalArray laia)
cpdef _extract_sparse_covs(const char *file_name, LabeledIntervalArray laia)