import numpy as np
cimport numpy as cnp
from libcpp.vector cimport vector
from libc.math cimport fabs
cimport cython

#@cython.boundscheck(False)
#@cython.wraparound(False)
#@cython.nonecheck(False)
def decarry_batch(
    double[:, :] spectra_peaks,
    unsigned int[:] peak_counts,
    double[:] initial_carries,
    int[:] initial_carries_count,
    unsigned int threshold,
    double rtol
):
    # Declare all variables at the top level
    cdef unsigned int n_spectra = spectra_peaks.shape[0]
    cdef unsigned int max_peaks = spectra_peaks.shape[1]
    cdef unsigned int i, j, k, i_r, j_r
    cdef double peak_val, rel_diff

    # Declare vectors here
    cdef vector[double] current_carries
    cdef vector[int] current_carries_count
    cdef vector[int] current_back_cleaned
    # Indicating if a current carry was found in the current spectrum
    cdef vector[int] local_match_flags
    # Used for construction of next current_* vects
    cdef vector[double] next_carries
    cdef vector[int] next_carries_count
    cdef vector[int] next_back_cleaned
    # Array to check if a peak should be added to current_carries
    cdef int[:] peak_is_new = np.ones(max_peaks, dtype=np.int32)

    # Initialize results
    cdef int[:, :] uncarry_mask = np.ones((n_spectra, max_peaks), dtype=np.int32)
    cdef double[:, :] cleaned_spectra = np.zeros((n_spectra, max_peaks), dtype=np.float64)

    # Initialize from input
    for i in range(initial_carries.shape[0]):
        current_carries.push_back(initial_carries[i])
        current_carries_count.push_back(initial_carries_count[i])
        # Initialize back-cleaned status. Some might actually be already back-cleaned but that's okay.
        if current_carries_count[i] >= threshold:
            current_back_cleaned.push_back(1)
        else:
            current_back_cleaned.push_back(0)

    # Define back-clean indices
    cdef int j_r_lower, j_r_upper
    for i in range(threshold, n_spectra):
        local_match_flags.assign(current_carries.size(), 0)

        for j in range(peak_counts[i]):
            peak_val = spectra_peaks[i, j]
            if peak_val == 0: continue
            # Check Local
            for k in range(current_carries.size()):
                rel_diff = fabs(peak_val - current_carries[k]) / peak_val
                if rel_diff <= rtol:
                    peak_is_new[j] = 0
                    current_carries_count[k] += 1
                    local_match_flags[k] = 1
                    if current_carries_count[k] >= threshold:
                        uncarry_mask[i, j] = 0
                        if not current_back_cleaned[k]:
                            # Uncarry backwards but leave the first occurance
                            for i_r in range(i-1, i-threshold, -1):
                                j_r = np.searchsorted(spectra_peaks[i_r], peak_val)
                                j_r_lower = max(j_r-1, 0)
                                j_r_upper = min(j_r, peak_counts[i]-1)
                                lower_peak = spectra_peaks[i_r, j_r_lower]
                                upper_peak = spectra_peaks[i_r, j_r_upper]
                                # Check lower peak
                                rel_diff = fabs(peak_val - lower_peak) / peak_val
                                if rel_diff <= rtol:
                                    uncarry_mask[i_r, j_r_lower] = 0
                                # Check upper peak
                                rel_diff = fabs(peak_val - upper_peak) / peak_val
                                if rel_diff <= rtol:
                                    uncarry_mask[i_r, j_r_upper] = 0
                            current_back_cleaned[k] = 1
                    break # Stop checking carries once matched

        # Decrease unmatched carries
        for k in range(current_carries.size()):
            if local_match_flags[k] == 0:
                current_carries_count[k] -= 1

        # Refresh state for next spectrum
        next_carries.clear()
        next_carries_count.clear()
        next_back_cleaned.clear()

        for k in range(current_carries.size()):
            if current_carries_count[k] > 0:
                next_carries.push_back(current_carries[k])
                next_carries_count.push_back(current_carries_count[k])
                next_back_cleaned.push_back(current_back_cleaned[k])

        for j in range(peak_counts[i]):
            if peak_is_new[j]:
                next_carries.push_back(spectra_peaks[i, j])
                next_carries_count.push_back(0)
                next_back_cleaned.push_back(0)

        current_carries = next_carries
        current_carries_count = next_carries_count
        current_back_cleaned = next_back_cleaned
        peak_is_new = np.ones(max_peaks, dtype=np.int32)

    return (
        np.asarray(uncarry_mask),
        np.array(current_carries),
        np.array(current_carries_count),
    )