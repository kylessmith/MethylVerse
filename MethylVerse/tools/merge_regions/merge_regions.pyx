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


cdef labeled_aiarray_t *_merge_adjacent(labeled_aiarray_t *laia, uint32_t n):
    """
    """

    cdef labeled_aiarray_t *merged_laia = adjacent_merge(laia, n)

    return merged_laia


cdef labeled_aiarray_t *_merge_less_than(labeled_aiarray_t *laia, double *values, double cutoff):
    """
    """

    cdef labeled_aiarray_t *merged_laia = adjacent_merge_less_than(laia, values, cutoff)

    return merged_laia


cdef labeled_aiarray_t *_merge_greater_than(labeled_aiarray_t *laia, double *values, double cutoff):
    """
    """

    cdef labeled_aiarray_t *merged_laia = adjacent_merge_greater_than(laia, values, cutoff)

    return merged_laia


def merge_adjacent(LabeledIntervalArray laia, int n):
    """
    """

    # Make sure is constructed
    laia.construct()

    # Merge
    cdef uint32_t ncounts = n
    cdef labeled_aiarray_t *merged_laia =_merge_adjacent(laia.laia, ncounts)

    # Wrap c intervals
    intervals = LabeledIntervalArray()
    intervals.set_list(merged_laia)

    # Sort intervals
    intervals.construct()
    intervals.sort()

    return intervals


def merge_less_than(LabeledIntervalArray laia, np.ndarray values, float cutoff):
    """
    """

    # Make sure is constructed
    laia.construct()

    # Merge
    cdef double[::1] c_values = values
    cdef labeled_aiarray_t *merged_laia =_merge_less_than(laia.laia, &c_values[0], cutoff)

    # Wrap c intervals
    intervals = LabeledIntervalArray()
    intervals.set_list(merged_laia)

    # Sort intervals
    intervals.construct()
    intervals.sort()

    return intervals


def merge_greater_than(LabeledIntervalArray laia, np.ndarray values, float cutoff):
    """
    """

    # Make sure is constructed
    laia.construct()

    # Merge
    cdef double[::1] c_values = values
    cdef labeled_aiarray_t *merged_laia =_merge_greater_than(laia.laia, &c_values[0], cutoff)

    # Wrap c intervals
    intervals = LabeledIntervalArray()
    intervals.set_list(merged_laia)

    # Sort intervals
    intervals.construct()
    intervals.sort()

    return intervals