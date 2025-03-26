# cython: language_level=3

cimport cython
import numpy as np
cimport numpy as np

from ailist.array_query_core cimport pointer_to_numpy_array
from ailist.LabeledIntervalArray_core cimport LabeledIntervalArray, labeled_aiarray_t, labeled_aiarray_init, labeled_aiarray_add
from ailist.AIList_core cimport AIList, ailist_t, ailist_init, ailist_add

from libc.stdint cimport uint32_t, uint8_t, uint64_t, int64_t, uint16_t, int32_t


cdef extern from "process_merge.c":
    # C is include here so that it doesn't need to be compiled externally
    pass

cdef extern from "process_merge.h":
    # C is include here so that it doesn't need to be compiled externally

    labeled_aiarray_t *adjacent_merge(labeled_aiarray_t *laia, uint32_t n) nogil

    labeled_aiarray_t *adjacent_merge_less_than(labeled_aiarray_t *laia, double *values, double cutoff) nogil

    labeled_aiarray_t *adjacent_merge_greater_than(labeled_aiarray_t *laia, double *values, double cutoff) nogil


cdef labeled_aiarray_t *_merge_adjacent(labeled_aiarray_t *laia, uint32_t n)
cdef labeled_aiarray_t *_merge_less_than(labeled_aiarray_t *laia, double *values, double cutoff)
cdef labeled_aiarray_t *_merge_greater_than(labeled_aiarray_t *laia, double *values, double cutoff)