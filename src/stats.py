import numpy as np

def longest_run(mask):
    """Length of the longest run of True in a 1D boolean array."""
    if not mask.any():
        return 0
    padded = np.concatenate(([False], mask, [False]))
    d = np.diff(padded.astype(np.int8))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    return (ends - starts).max()