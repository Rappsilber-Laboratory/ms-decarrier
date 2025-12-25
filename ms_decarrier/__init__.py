import numpy as np
import threading
from queue import Queue
from tqdm import tqdm
from pyteomics import mgf
from itertools import zip_longest
from .decarry_batch import decarry_batch

def decarry_file(input_file, output_file, threshold=4, rtol=100e-6):
    # Queues for parallel I/O
    read_queue = Queue(maxsize=10)
    write_queue = Queue(maxsize=10)

    # State carry-over: results from previous batch are passed to the next
    state = {
        'cur_c': np.array([], dtype=np.float64),
        'cur_cnt': np.array([], dtype=np.int32),
    }
    carry_mask = None

    # Reader Thread: Constantly loads batches of 1000
    def reader():
        batch = []
        # We use a try/except or simple loop for mgf stream
        for spectrum in mgf.read(input_file):
            batch.append(spectrum)
            if len(batch) >= 1_000:
                read_queue.put(batch)
                batch = []
        if batch:
            read_queue.put(batch)
        read_queue.put([])  # Sentinel to stop

    # Writer Thread: Writes processed spectra to disk
    def writer():
        with open(output_file, 'wt') as f:
            while True:
                batch = write_queue.get()
                if batch is None:
                    break
                mgf.write(batch, output=f)
                write_queue.task_done()

    # Start the I/O threads
    t_read = threading.Thread(target=reader, daemon=True)
    t_write = threading.Thread(target=writer, daemon=True)
    t_read.start()
    t_write.start()

    # Processing Loop (Main Thread)
    pbar = tqdm(desc="Cleaning Spectra", unit="spec")

    batch_overlap = [
        {
            'm/z array': np.array([]),
            'intensity array': np.array([]),
            'params': dict(),
        } for _ in range(threshold)
    ]
    skip_offset = threshold
    while True:
        raw_batch = read_queue.get()
        if len(raw_batch) > 0:
            raw_batch_overlapping = batch_overlap+raw_batch
        else:
            raw_batch_overlapping = batch_overlap
            # Set batch_overlap=[] so we don't skip it on clean-up
            batch_overlap = []
        mzs = [s['m/z array'] for s in raw_batch_overlapping]
        ints = [s['intensity array'] for s in raw_batch_overlapping]

        # Prepare the input matrix
        peak_counts = np.array([len(x) for x in mzs], dtype=np.int32)

        # Using zip_longest for padding
        spectra_mzs = np.array(
            list(zip_longest(*mzs, fillvalue=0))
        ).T
        spectra_intensities = np.array(
            list(zip_longest(*ints, fillvalue=0))
        ).T

        # Call Cython function
        # (Assuming return: cleaned_spectra, current_carries, current_counts, global_carries)
        # Force the types right at the point of the call
        carry_mask_batch, state['cur_c'], state['cur_cnt'] = decarry_batch(
            spectra_mzs,
            peak_counts.astype(np.uint32),
            initial_carries=state['cur_c'].astype(np.float64),
            initial_carries_count=state['cur_cnt'].astype(np.int32),
            threshold=threshold,
            rtol=rtol,
        )

        if carry_mask is None:
            carry_mask = carry_mask_batch[:-threshold]
        else:
            carry_mask = _fast_concat_and_pad(
                carry_mask[skip_offset:],
                carry_mask_batch[:-len(batch_overlap)],
            )

        # Reconstruct the batch with original parameters and intensities
        output_spectra = []
        keep_mask = carry_mask_batch[skip_offset:].astype(bool)
        for i, original_spec in enumerate(raw_batch_overlapping[:-len(batch_overlap)]):
            new_spec = {
                'm/z array': spectra_mzs[skip_offset:][i][keep_mask[i]],
                'intensity array': spectra_intensities[skip_offset:][i][keep_mask[i]],
                'params': original_spec.get('params', {})
            }
            output_spectra.append(new_spec)

        write_queue.put(output_spectra)
        pbar.update(len(raw_batch))

        overlap_size = min(threshold, len(raw_batch))
        batch_overlap = raw_batch[-overlap_size:]
        # The initial batch was processed and there
        # won't be dummy data prepended anymore
        skip_offset = 0
        if len(raw_batch) == 0:
            break

    # Wait for writer to finish flushing to disk
    write_queue.put(None)
    t_read.join()
    t_write.join()
    pbar.close()
    print(f"Done! Final Carry Count: {np.count_nonzero(~carry_mask)}")
    return carry_mask


def _fast_concat_and_pad(arr1, arr2):
    h1, w1 = arr1.shape
    h2, w2 = arr2.shape
    max_w = max(w1, w2)

    # Pre-allocate full result array
    res = np.zeros(
        shape=(h1 + h2, max_w),
        dtype=arr1.dtype,
    )

    # Place arrays in the result
    res[:h1, :w1] = arr1
    res[h1:, :w2] = arr2

    return res
