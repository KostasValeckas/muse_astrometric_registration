import numpy as np
import math

def longest_run(mask):
    """Length of the longest run of True in a 1D boolean array."""
    if not mask.any():
        return 0
    padded = np.concatenate(([False], mask, [False]))
    d = np.diff(padded.astype(np.int8))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    return (ends - starts).max()


def masked_ncc(a, b, mask):
    a = a[mask].astype(np.float64)
    b = b[mask].astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    denom = np.sqrt((a**2).sum() * (b**2).sum())
    return 0.0 if denom == 0 else (a * b).sum() / denom


def ternary_search(f, left, right, eps=1e-2, nan_default=1.0):
    tried = []

    def evaluate(x):
        y = f(x)
        if math.isnan(y):
            y = nan_default
        tried.append((x, y))
        return y

    while right - left > eps:
        m1 = left + (right - left) / 3
        m2 = right - (right - left) / 3

        # Check if the two search points cross
        if m1 >= m2:
            return None, None, None

        f1 = evaluate(m1)
        f2 = evaluate(m2)

        if f1 < f2:
            right = m2
        else:
            left = m1

    x_min = (left + right) / 2
    y_min = evaluate(x_min)

    return x_min, y_min, tried