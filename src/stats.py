"""Statistics and search helpers: runs, correlation, ternary search and robust thresholds."""

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
    """Normalised cross-correlation of `a` and `b` over the pixels where `mask` is True (0 if either is constant)."""
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
        """Evaluate `f` at x (NaN counts as `nan_default`) and record the point."""
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


def robust_residual_threshold(
    residuals, n_sigma=3.0, minimum=0.05, maximum=0.5, min_points=5, n_parameters=0
):
    """
    Outlier threshold (pixels) on the LENGTH of 2D fit residuals, scaled to
    the scatter of the residuals themselves instead of a fixed number.

    Why
    ---
    A fixed threshold (e.g. 0.5 px) means a different thing for every frame
    pair: for precise residuals (scatter ~0.05 px) it is a 10 sigma cut and
    lets clearly wrong points pull the fit, for noisy residuals it may cut
    into the good points. Cutting on the length |r| (a circle) instead of
    |dx| or |dy| (a box) also makes the cut independent of how the frame is
    oriented relative to the pixel axes.

    How
    ---
    For isotropic gaussian residuals with sigma per axis, |r| follows a
    Rayleigh distribution with median sigma * sqrt(2 ln 2) = 1.1774 sigma, so

        sigma = median(|r|) / 1.1774

    Like the MAD this is robust to up to 50% outliers. Residuals of a
    least-squares fit are smaller than the true scatter, since the fit
    follows the points a bit: with N points (2N coordinates) and
    `n_parameters` fitted parameters, sigma is corrected by
    sqrt(2N / (2N - n_parameters)) (1.15x for 6 point sources and a rotation
    + translation fit, negligible for thousands of flow vectors). The
    threshold is
    n_sigma * sigma (3 sigma: 1.1% of good points fall outside, exp(-9/2)),
    clipped to [minimum, maximum]:
    - `minimum` keeps a noise-free or quantised case from collapsing to 0
      and rejecting points over rounding-level differences;
    - `maximum` is the old fixed threshold, so the cut is never looser than
      before, also when the fit is bad and the scatter large.

    Pass the residuals of the CURRENT inliers (already roughly clean), and
    apply the threshold to all points so rejected points can come back.

    Parameters
    ----------
    residuals : array
        (N, 2) residual vectors, or (N,) residual lengths.

    n_sigma, minimum, maximum : float
        See above.

    min_points : int
        With fewer finite residuals the scatter is not trusted and `maximum`
        is returned.

    n_parameters : int
        Number of fitted parameters, for the correction above (0: none).

    Returns
    -------
    threshold, sigma
        Threshold in pixels and the estimated sigma per axis (NaN if not
        estimated).
    """
    residuals = np.asarray(residuals, dtype=float)

    if residuals.ndim == 2:
        lengths = np.linalg.norm(residuals, axis=1)
    else:
        lengths = np.abs(residuals)

    lengths = lengths[np.isfinite(lengths)]

    if len(lengths) < min_points:
        return float(maximum), np.nan

    sigma = float(np.median(lengths) / np.sqrt(2 * np.log(2)))

    n_coordinates = 2 * len(lengths)
    if 0 < n_parameters < n_coordinates:
        sigma *= float(np.sqrt(n_coordinates / (n_coordinates - n_parameters)))

    threshold = float(np.clip(n_sigma * sigma, minimum, maximum))

    return threshold, sigma
