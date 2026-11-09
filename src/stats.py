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


def ternary_search(f, left, right, eps=1e-1, nan_default=1.0, logger=None):
    """
    Ternary search for the minimum of `f` in [left, right].

    Reports failure (returns None, None, None) if
    - the final bracket still touches the original left or right boundary,
      i.e. the minimum may lie at or beyond the search range, or
    - all evaluated points returned the same value (no information, e.g.
      every evaluation failed and got `nan_default`).

    Returns
    -------
    x_min, y_min, tried
        Position and value of the minimum, and the list of (x, y) evaluated.
    """
    tried = []
    left_0, right_0 = left, right

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

    # all points returned the same value: no minimum to find
    ys = np.array([y for _, y in tried], dtype=float)
    if np.allclose(ys, ys[0], rtol=1e-9, atol=0.0):
        if logger:
            logger.warning(
                f"Ternary search: all {len(ys)} evaluations returned the "
                f"same value ({ys[0]:.6g}), reporting failure"
            )
        return None, None, None

    # the bracket never moved away from one of the edges: the minimum
    # can be at or beyond the boundary of the search range
    if left == left_0 or right == right_0:
        if logger:
            logger.warning(
                f"Ternary search reached the boundary of the search range "
                f"[{left_0:.4f}, {right_0:.4f}] (final bracket "
                f"[{left:.4f}, {right:.4f}]), reporting failure"
            )
        return None, None, None

    return x_min, y_min, tried