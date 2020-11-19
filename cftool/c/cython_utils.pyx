cimport cython
cimport numpy as np
import numpy as np

from libc.math cimport isnan
from cython.parallel import prange


# api


@cython.wraparound(False)
@cython.boundscheck(False)
def rolling_sum(np.ndarray[np.float64_t, ndim=1] flat_data, int window, int mean):
    cdef int i, section_idx, cursor
    cdef unsigned int num_data = len(flat_data)
    cdef unsigned int num_sections = num_data // window

    cdef double nan = np.nan
    cdef double sum_cache, valid_sum_cache
    cdef np.ndarray[np.float64_t, ndim=1] results = np.empty(num_data, dtype=np.float64)

    for section_idx in prange(num_sections, nogil=True):
        cursor = window * section_idx

        sum_cache = valid_sum_cache = 0.0
        for i in range(window):
            if not isnan(flat_data[cursor + i]):
                sum_cache = sum_cache + flat_data[cursor + i]
                valid_sum_cache = valid_sum_cache + 1.0

        if valid_sum_cache == 0.0:
            results[cursor + window - 1] = nan
        else:
            if mean == 0:
                results[cursor + window - 1] = sum_cache
            else:
                results[cursor + window - 1] = sum_cache / valid_sum_cache
        for i in range(window - 1):
            if cursor + window + i < num_data:
                if not isnan(flat_data[cursor + window + i]):
                    sum_cache = sum_cache + flat_data[cursor + window + i]
                    valid_sum_cache = valid_sum_cache + 1.0
                if not isnan(flat_data[cursor + i]):
                    sum_cache = sum_cache - flat_data[cursor + i]
                    valid_sum_cache = valid_sum_cache - 1.0
                if valid_sum_cache == 0.0:
                    results[cursor + window + i] = nan
                else:
                    if mean == 0:
                        results[cursor + window + i] = sum_cache
                    else:
                        results[cursor + window + i] = sum_cache / valid_sum_cache
            else:
                break

    return results[window - 1:]


@cython.wraparound(False)
@cython.boundscheck(False)
def rolling_min(np.ndarray[np.float64_t, ndim=1] flat_data, int window):
    cdef int i, section_idx, cursor
    cdef unsigned int num_data = len(flat_data)
    cdef unsigned int num_sections = num_data // window

    cdef double cache_min, final_min
    cdef np.ndarray[np.float64_t, ndim=1] final_results = np.empty(num_data, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] caches = np.empty(num_data, dtype=np.float64)

    for section_idx in prange(num_sections, nogil=True):
        cursor = window * section_idx

        cache_min = flat_data[cursor + window - 1]
        for i in range(window - 1, -1, -1):
            if flat_data[cursor + i] < cache_min:
                cache_min = flat_data[cursor + i]
            caches[cursor + i] = cache_min

        final_min = flat_data[cursor + window]
        final_results[cursor + window - 1] = caches[cursor]
        for i in range(window - 1):
            if cursor + window + i < num_data:
                if flat_data[cursor + window + i] < final_min:
                    final_min = flat_data[cursor + window + i]
                final_results[cursor + window + i] = min(final_min, caches[cursor + i + 1])
            else:
                break

    return final_results[window - 1:]


@cython.wraparound(False)
@cython.boundscheck(False)
def rolling_max(np.ndarray[np.float64_t, ndim=1] flat_data, int window):
    cdef int i, section_idx, cursor
    cdef unsigned int num_data = len(flat_data)
    cdef unsigned int num_sections = num_data // window

    cdef double cache_max, final_max
    cdef np.ndarray[np.float64_t, ndim=1] final_results = np.empty(num_data, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] caches = np.empty(num_data, dtype=np.float64)

    for section_idx in prange(num_sections, nogil=True):
        cursor = window * section_idx

        cache_max = flat_data[cursor + window - 1]
        for i in range(window - 1, -1, -1):
            if flat_data[cursor + i] > cache_max:
                cache_max = flat_data[cursor + i]
            caches[cursor + i] = cache_max

        final_max = flat_data[cursor + window]
        final_results[cursor + window - 1] = caches[cursor]
        for i in range(window - 1):
            if cursor + window + i < num_data:
                if flat_data[cursor + window + i] > final_max:
                    final_max = flat_data[cursor + window + i]
                final_results[cursor + window + i] = max(final_max, caches[cursor + i + 1])
            else:
                break

    return final_results[window - 1:]


@cython.wraparound(False)
@cython.boundscheck(False)
def ema(np.ndarray[np.float64_t, ndim=1] flat_data, float ratio):
    cdef int i
    cdef unsigned int num_data = len(flat_data)
    cdef np.ndarray[np.float64_t, ndim=1] results = np.empty(num_data, dtype=np.float64)
    cdef double rev_ratio = 1.0 - ratio
    cdef double current, running

    for i in range(num_data):
        current = flat_data[i]
        if isnan(current):
            results[i] = current
        else:
            if i == 0:
                running = ratio * current
            else:
                running = ratio * current + rev_ratio * running
            results[i] = running

    return results
