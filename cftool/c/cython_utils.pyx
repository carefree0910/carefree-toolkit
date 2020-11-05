cimport cython
cimport numpy as np
import numpy as np

from cython.parallel import prange


# api

@cython.wraparound(False)
@cython.boundscheck(False)
def rolling_min(np.ndarray[np.float32_t, ndim=1] flat_data, int window):
    cdef int i, section_idx, cursor
    cdef unsigned int num_data = len(flat_data)
    cdef unsigned int num_sections = num_data // window

    cdef float cache_min, final_min
    cdef np.ndarray[np.float32_t, ndim=1] final_results = np.empty(num_data, dtype=flat_data.dtype)
    cdef np.ndarray[np.float32_t, ndim=1] caches = np.empty(num_data, dtype=flat_data.dtype)

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
def rolling_max(np.ndarray[np.float32_t, ndim=1] flat_data, int window):
    cdef int i, section_idx, cursor
    cdef unsigned int num_data = len(flat_data)
    cdef unsigned int num_sections = num_data // window

    cdef float cache_max, final_max
    cdef np.ndarray[np.float32_t, ndim=1] final_results = np.empty(num_data, dtype=flat_data.dtype)
    cdef np.ndarray[np.float32_t, ndim=1] caches = np.empty(num_data, dtype=flat_data.dtype)

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
