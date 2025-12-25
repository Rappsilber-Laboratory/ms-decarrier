import numpy as np
from numpy import nan
from ms_decarrier import decarry_batch

def test_backwards_cleanup():
    mz_batch = np.array([
        [1.2, 2.2, 3,   4.2, 5.2, 6.2],
        [1.3, 2.3, 3,   4.3, 5.3, 6.3],
        [1.4, 2.4, 3,   4.4, 5.4, 6.4],
        [1.5, 2.5, 3,   4.5, 5,   6.5],
        # end of overlap
        [1,   2.6, 3,   4.6, 5,   6.6],
        [1,   2.7, 3.7, 4.7, 5,   6.7],
        [1,   2.8, 3.8, 4.8, 5,   6.8],
        [1,   2.9, 3.9, 4.9, 5.9, 6.9],
        [1,   3.1, 4.1, 5.1, 6.1, 7.1],
        [2.2, 3.2, 4.2, 5.2, 6.2, 7.2],
        [2.3, 3.3, 4.3, 5.3, 6.3, 7.3],
    ])
    exp_decarry = np.array([
        [1, 1, 1, 1, 1, 1],
        [1, 1, 0, 1, 1, 1],
        [1, 1, 0, 1, 1, 1],
        [1, 1, 0, 1, 1, 1],
        # end of overlap
        [1, 1, 0, 1, 1, 1],
        [0, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
    ])
    decarry_mask = decarry_batch(
        spectra_peaks=mz_batch,
        peak_counts=np.array([len(s) for s in mz_batch], dtype=np.uint32),
        initial_carries=np.array([3, 5], dtype=np.float64),
        initial_carries_count=np.array([3, 0], dtype=np.int32),
        threshold=4,
        rtol=1*10e-6
    )
    pass
