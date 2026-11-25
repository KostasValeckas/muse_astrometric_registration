"""
Experiment: estimate the orientation of the MUSE slicer pattern on the sky from
the residual stripes in the difference images of a finished run.

The exposures keep a faint pattern of straight lines from the instrument (the
slices and the slicer stacks / IFU channels), which is not in the reference.
It is a near-orthogonal grid of two line families. Subtracting the reference
removes the galaxy and leaves the pattern, and its orientation is measured
relative to the pixel grid of the exposure, which is assumed to be exactly
North up (East left).

For every exposure with a result in the output table of the run:

#. The frames are prepared as in the pipeline (``prepare_pointing`` of
   ``replot_final_difference_images.py``: exposures of the pointing in the
   frame of the first one, reference in that frame, converted to the MUSE unit
   with the pixel area ratio), and ``reference = FLUX_SCALE * (exposure +
   BACKGROUND)`` is fitted with the final offsets of the table.
#. The difference ``exposure - (reference / FLUX_SCALE - BACKGROUND)`` is made
   in the frame of the exposure: the *reference* is moved onto the exposure
   with the inverse of the final transformation, so the pattern keeps the
   orientation it has in the exposure (rotating the exposure would rotate the
   pattern by ROTANGLE).
#. The difference is high pass filtered (minus a median filter; not with
   ``--highpass 0``), the edges of
   the frame are eroded (they are aligned with the pixel grid and would pull
   the result to 0), outliers are clipped, and the result is smoothed a little
   (a nearly band limited image is rotated without the interpolation changing
   its variance, which otherwise makes a cusp at exactly 0 deg).
#. The residual is rotated over a range of angles (cubic spline) and the
   variance of its mean profiles across the rows (near-horizontal lines) and
   across the columns (near-vertical lines) is computed. Each variance peaks
   where its line family is aligned with the axis; the peak is located with a
   Gaussian fit. The error is from the scatter of the four quadrants.
#. The individual lines of both families are found as the peaks of the
   profiles at the measured angles (``find_lines``), and the angle of every
   line is fitted from its position in segments along it.
#. As a check of the geometry the same measurement is made on the difference
   in the frame of the reference (the exposure rotated by ROTANGLE, as in the
   final difference images), which must give the native angle - ROTANGLE.

Angles
------
``angle`` is the orientation of a line family relative to the image axes, in
degrees, positive counter-clockwise as displayed North up, East left (origin
lower, as DS9). The pipeline ROTANGLE (scipy convention, see
``docs/output.rst``, ``rotation_convention``) is a clockwise correction in this display,
i.e. the sky in the exposure is rotated by +ROTANGLE in the same sense as
``angle``.

``PA`` is the position angle of the lines (from North through East, modulo
180): ``PA = 90 + angle`` for the near-horizontal and ``PA = angle`` for the
near-vertical lines. ``deviation`` is the angle minus the nominal orientation
of the instrument, ``HIERARCH ESO INS DROT POSANG`` (modulo 90, since it is
not known which family is which).

Written to ``--out_dir`` (default ``<output_dir>/stripe_orientation``):

- ``stripe_orientation_{pointing}_dither_{i}.png`` per exposure: residual with
  the measured line directions, variance curves, power spectrum and profiles,
- ``stripe_orientation_summary.png``: deviations of all exposures, with
  ROTANGLE of the table,
- ``stripe_orientation.csv``: the results,
- ``stripe_lines.csv``: the individual lines,
- the log.

Example::

    python dev/stripe_orientation.py \\
        NGC1365_P1/secondary_OpticalFlow NGC1365_P1.json --table OpticalFlow
"""

import json
import os
import pathlib
import sys

import numpy as np

# the pipeline modules in ../src, and replot_final_difference_images.py here
_HERE = pathlib.Path(__file__).resolve().parent
for _path in (_HERE.parent / "src", _HERE):
    if _path.is_dir() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from replot_final_difference_images import (  # noqa: E402
    find_run_manifest,
    prepare_pointing,
    METHOD_COLUMN,
)

# keyword with the nominal position angle of the instrument
POSANG_KEYWORD = "HIERARCH ESO INS DROT POSANG"

# minimum sigma (deg) of the Gaussian fitted to the variance peaks (the peaks
# are ~0.5 - 1 deg wide: the lines are segments of ~75 pixels)
MIN_PEAK_WIDTH = 0.15

# pixels blanked along the cuts of the quadrants (a cut is a data edge aligned
# with the grid, see prepare_residual)
QUADRANT_BORDER = 4

# names of the two line families (index 0: rows, index 1: columns)
FAMILIES = ("horizontal", "vertical")


def reference_on_exposure(reference, tx, ty, rotation_angle):
    """
    Move the reference onto the exposure with the inverse of the final
    transformation of the exposure (which is: rotate by `rotation_angle` about
    the centre, then shift by (`tx`, `ty`), scipy convention). The inverse is:
    shift by (-`tx`, -`ty`), then rotate by -`rotation_angle`.

    Parameters
    ----------
    reference : np.2darray
        The reference in the common frame of the pointing.

    tx, ty : float
        Final translation of the exposure, in pixels.

    rotation_angle : float
        Final rotation of the exposure (ROTANGLE), in degrees.

    Returns
    -------
    moved : np.2darray
        The reference in the frame of the exposure, NaN where it has no data.
    """

    from scipy.ndimage import shift, rotate

    finite = np.isfinite(reference)

    data = rotate(
        shift(np.where(finite, reference, 0.0), (-ty, -tx), order=3, mode="nearest"),
        -rotation_angle,
        reshape=False,
        order=3,
        mode="nearest",
    )
    covered = rotate(
        shift(finite.astype(float), (-ty, -tx), order=1, cval=0.0),
        -rotation_angle,
        reshape=False,
        order=1,
        cval=0.0,
    )

    return np.where(covered > 0.999, data, np.nan)


def prepare_residual(difference, highpass=15, edge=12, clip=4.0, smooth=1.0):
    """
    Prepare a difference image for the line measurement: high pass filter,
    erode the edges, clip outliers, smooth.

    Parameters
    ----------
    difference : np.2darray
        The difference image, NaN where it has no data.

    highpass : int, optional
        Size (pixels) of the median filter subtracted as the background; 0 for
        no high pass filter (only the median is subtracted). Default is 15.

    edge : int, optional
        Pixels removed along the edges of the data (they are aligned with the
        pixel grid and would bias the angle to 0). Default is 12.

    clip : float, optional
        Pixels further than `clip` MAD standard deviations from 0 (residuals of
        stars and of the galaxy) are masked. Default is 4.

    smooth : float, optional
        Sigma (pixels) of the Gaussian smoothing. Default is 1.

    Returns
    -------
    residual : np.2darray
        The prepared residual, 0 where masked.

    mask : np.2darray
        1 where the residual is used, 0 where masked (float, for the rotation).
    """

    from scipy.ndimage import median_filter, binary_erosion, gaussian_filter
    from astropy.stats import mad_std

    good = np.isfinite(difference)
    if not np.any(good):
        raise ValueError("The difference image has no finite pixels.")

    filled = np.where(good, difference, np.nanmedian(difference))
    if highpass and highpass > 0:
        residual = filled - median_filter(filled, size=highpass)
    else:
        # no high pass filter: only the median is subtracted
        residual = filled - np.nanmedian(difference)

    good = binary_erosion(good, iterations=edge, border_value=0)
    if np.sum(good) < 100:
        raise ValueError(f"Fewer than 100 pixels left after removing {edge} edge pixels.")

    good &= np.abs(residual) < clip * mad_std(residual[good])

    residual = np.where(good, residual, 0.0)
    mask = good.astype(float)

    if smooth:
        smoothed = gaussian_filter(residual, smooth)
        weight = gaussian_filter(mask, smooth)
        residual = np.where(weight > 0.5, smoothed / np.maximum(weight, 1e-6), 0.0) * mask

    return residual, mask


def line_family_scores(residual, mask, angle, order=3):
    """
    Variance of the mean profiles of `residual` rotated by `angle`: across the
    rows (peaks when the near-horizontal lines are horizontal) and across the
    columns (near-vertical lines).

    ``scipy.ndimage.rotate`` by +`angle` turns a line at `angle` (counter-
    clockwise, origin lower) onto the axis.

    Returns
    -------
    scores : np.ndarray
        [horizontal, vertical].
    """

    from scipy.ndimage import rotate

    rotated = rotate(residual, angle, reshape=True, order=order)
    rotated_mask = rotate(mask, angle, reshape=True, order=1)

    scores = np.empty(2)
    # axis 1: mean along x for every row (horizontal lines), axis 0: columns
    for k, axis in enumerate((1, 0)):
        counts = rotated_mask.sum(axis=axis)
        sums = rotated.sum(axis=axis)
        use = counts > 0.5 * counts.max()
        scores[k] = np.var(sums[use] / counts[use]) if np.sum(use) > 2 else np.nan

    return scores


def _gaussian(x, amplitude, centre, sigma, offset):
    return amplitude * np.exp(-0.5 * ((x - centre) / sigma) ** 2) + offset


def fit_peak(angles, scores):
    """
    Locate the peak of a variance curve with a Gaussian + constant fit.

    Returns
    -------
    centre : float
        Angle of the peak, NaN if the fit failed or the peak is not inside the
        scanned range.

    parameters : np.ndarray or None
        Parameters of the fit (for the plot).
    """

    from scipy.optimize import curve_fit

    good = np.isfinite(scores)
    if np.sum(good) < 5:
        return np.nan, None

    x, y = angles[good], scores[good]
    k = np.argmax(y)
    start = [y[k] - y.min(), x[k], 0.5, y.min()]

    # the width is bounded from below, so a narrow spike (e.g. from data edges
    # aligned with the grid) cannot be fitted instead of the peak
    try:
        parameters, _ = curve_fit(
            _gaussian,
            x,
            y,
            p0=start,
            bounds=(
                [0.0, x.min(), MIN_PEAK_WIDTH, -np.inf],
                [np.inf, x.max(), np.inf, np.inf],
            ),
            maxfev=5000,
        )
    except (RuntimeError, ValueError):
        return np.nan, None

    # a peak at (or next to) the end of the range is not a peak in the range
    centre = parameters[1]
    margin = 2 * np.median(np.diff(x))
    if not (x.min() + margin < centre < x.max() - margin) or parameters[0] <= 0:
        return np.nan, parameters

    return centre, parameters


def coarse_line_angle(residual, mask, step=0.5):
    """
    Coarse orientation of the grid of lines (modulo 90 deg): the angle in
    [-45, 45) where the sum of the normalised variances of both families
    peaks. Linear interpolation (fast; its cusp at 0 is narrower than `step`).
    """

    angles = np.arange(-45.0, 45.0, step)
    scores = np.array(
        [line_family_scores(residual, mask, a, order=1) for a in angles]
    )
    total = np.nansum(scores / np.nanmax(scores, axis=0), axis=1)

    return angles[np.argmax(total)]


def measure_line_angles(residual, mask, centre=0.0, half_range=2.5, step=0.02):
    """
    Measure the orientation of both line families in the angle range
    `centre` +- `half_range`.

    Returns
    -------
    result : dict
        ``angles`` (scanned), ``scores`` (n x 2), ``angle`` (2, NaN if not
        found), ``fit`` (parameters of the two fits).
    """

    angles = np.arange(centre - half_range, centre + half_range + 1e-9, step)
    scores = np.array([line_family_scores(residual, mask, a) for a in angles])

    fits = [fit_peak(angles, scores[:, k]) for k in range(2)]

    return {
        "angles": angles,
        "scores": scores,
        "angle": np.array([f[0] for f in fits]),
        "fit": [f[1] for f in fits],
    }


def quadrant_angles(residual, mask, centre=0.0, half_range=2.5, step=0.05):
    """
    The measurement of `measure_line_angles` in the four quadrants of the
    residual, for the error.

    Returns
    -------
    angles : np.ndarray
        4 x 2, NaN where the peak was not found.
    """

    ny, nx = residual.shape
    result = []
    for ys in (slice(0, ny // 2), slice(ny // 2, ny)):
        for xs in (slice(0, nx // 2), slice(nx // 2, nx)):
            quadrant = np.zeros_like(residual[ys, xs])
            quadrant_mask = np.zeros_like(quadrant)
            b = QUADRANT_BORDER
            quadrant[b:-b, b:-b] = residual[ys, xs][b:-b, b:-b]
            quadrant_mask[b:-b, b:-b] = mask[ys, xs][b:-b, b:-b]
            if np.sum(quadrant_mask) < 0.2 * quadrant_mask.size:
                result.append([np.nan, np.nan])
                continue
            result.append(
                measure_line_angles(
                    quadrant, quadrant_mask, centre, half_range, step
                )["angle"]
            )

    return np.array(result)


def angle_error(quadrants):
    """
    Error of the angles from the scatter of the quadrants (standard deviation
    / sqrt(n)); NaN with fewer than 2 quadrants.
    """

    errors = np.full(2, np.nan)
    for k in range(2):
        q = quadrants[:, k][np.isfinite(quadrants[:, k])]
        if len(q) >= 2:
            errors[k] = np.std(q, ddof=1) / np.sqrt(len(q))
    return errors


def wrap_angle(angle, period=90.0):
    """Wrap `angle` to [-period/2, period/2)."""
    return (np.asarray(angle) + period / 2) % period - period / 2


def position_angles(angle):
    """
    Position angles (from North through East, modulo 180) of the two line
    families at `angle` = [horizontal, vertical] (see the module docstring).
    """
    return np.array([(90.0 + angle[0]) % 180.0, angle[1] % 180.0])


def rotated_profile(residual, mask, angle, family):
    """
    Rotate the residual by `angle` (the line family `family` then lies along
    the axis) and take the mean profile across the lines.

    Returns
    -------
    profile : np.ndarray
        Mean across the lines, NaN where less than half of the maximum
        coverage.

    rotated, rotated_mask : np.2darray
        The rotated residual and mask (``reshape=True``, rotation about the
        centres of the input and the output).
    """

    from scipy.ndimage import rotate

    rotated = rotate(residual, angle, reshape=True, order=3)
    rotated_mask = rotate(mask, angle, reshape=True, order=1)

    # horizontal lines: mean along x for every row; vertical: every column
    axis = 1 if family == FAMILIES[0] else 0
    counts = rotated_mask.sum(axis=axis)
    use = counts > 0.5 * counts.max()
    profile = np.where(use, rotated.sum(axis=axis) / np.maximum(counts, 1e-6), np.nan)

    return profile, rotated, rotated_mask


def rotated_to_image(points, angle, rotated_shape, image_shape):
    """
    Coordinates (x, y) in the frame rotated by `angle` (``rotated_profile``)
    back to the image. ``scipy.ndimage.rotate`` maps p - c_in to
    R(-angle) (p - c_in) + c_out (counter-clockwise, origin lower), so the
    inverse is R(+angle).
    """

    a = np.deg2rad(angle)
    c_rot = (np.array(rotated_shape[::-1]) - 1) / 2
    c_img = (np.array(image_shape[::-1]) - 1) / 2
    p = np.atleast_2d(points) - c_rot
    x = np.cos(a) * p[:, 0] - np.sin(a) * p[:, 1]
    y = np.sin(a) * p[:, 0] + np.cos(a) * p[:, 1]

    return np.column_stack([x, y]) + c_img


def find_lines(
    residual,
    mask,
    angle,
    family,
    threshold=3.0,
    detrend=9,
    separation=3,
    segment=30,
    half_width=3,
):
    """
    Find the individual lines of a family and fit the angle of each.

    The lines are the peaks (bright or dark) of the profile across the family
    at `angle`, minus a running median of `detrend` pixels (for the detection
    only, so lines on a slope are found), above `threshold` MAD standard
    deviations, at least `separation` pixels apart. For every line, the strip
    of +-`half_width` pixels around it is cut into segments of `segment`
    pixels along the line; the position of the line in each segment is the
    sub-pixel extremum of the segment profile, and a straight line fitted to
    these positions gives the angle of the line.

    Returns
    -------
    lines : list of dict
        ``position`` (pixels from the centre, perpendicular to the family),
        ``sign`` (+1 bright, -1 dark), ``significance``, ``angle`` (of the
        line, as `angle`; NaN if fewer than 4 segments), ``angle_error``,
        ``n_segments``, ``ends`` (2 x 2, end points in the image) and
        ``segments`` (positions of the segments in the image).

    profile, detrended : np.ndarray
        The profile and the detrended profile (for the plot).
    """

    from scipy.ndimage import median_filter
    from scipy.signal import find_peaks
    from astropy.stats import mad_std

    profile, rotated, rotated_mask = rotated_profile(residual, mask, angle, family)

    # work on rows: the vertical family is transposed
    if family != FAMILIES[0]:
        rotated, rotated_mask = rotated.T, rotated_mask.T

    finite = np.isfinite(profile)
    filled = np.where(finite, profile, np.nanmedian(profile))
    detrended = filled - median_filter(filled, size=detrend, mode="nearest")
    detrended[~finite] = np.nan
    noise = mad_std(detrended[finite])

    strength = np.where(finite, np.abs(detrended), 0.0)
    peaks, _ = find_peaks(strength, height=threshold * noise, distance=separation)

    # the ends of the profile are the edges of the data, not lines
    used = np.flatnonzero(finite)
    margin = half_width + detrend // 2
    peaks = peaks[(peaks > used.min() + margin) & (peaks < used.max() - margin)]

    n_rows, n_columns = rotated.shape
    centre = (n_rows - 1) / 2
    columns = np.arange(n_columns)

    lines = []
    for peak in peaks:

        sign = np.sign(detrended[peak])
        low, high = peak - half_width, peak + half_width + 1
        if low < 0 or high > n_rows:
            continue

        strip = rotated[low:high]
        strip_mask = rotated_mask[low:high]
        covered = strip_mask.min(axis=0) > 0.5

        x_seg, y_seg = [], []
        for start in range(0, n_columns - segment + 1, segment):
            part = slice(start, start + segment)
            if np.mean(covered[part]) < 0.7:
                continue
            weight = strip_mask[:, part] * covered[part]
            seg_profile = sign * (strip[:, part] * weight).sum(axis=1) / weight.sum(axis=1)
            k = int(np.argmax(seg_profile))
            if k == 0 or k == len(seg_profile) - 1:
                continue
            # sub-pixel extremum (parabola through three points)
            a, b, c = seg_profile[k - 1 : k + 2]
            denominator = a - 2 * b + c
            delta = 0.5 * (a - c) / denominator if denominator < 0 else 0.0
            x_seg.append(np.mean(columns[part][covered[part]]))
            y_seg.append(low + k + delta)

        x_seg, y_seg = np.array(x_seg), np.array(y_seg)
        line_angle, line_error = np.nan, np.nan
        slope, intercept = 0.0, float(peak)
        x_centre = (n_columns - 1) / 2
        if len(x_seg) >= 4:
            (slope, intercept), cov = np.polyfit(x_seg - x_centre, y_seg, 1, cov=True)
            tilt = np.rad2deg(np.arctan(slope))
            tilt_error = np.rad2deg(np.sqrt(cov[0, 0]) / (1 + slope**2))
            # rows of the transposed (vertical) frame: the tilt is clockwise
            line_angle = angle + (tilt if family == FAMILIES[0] else -tilt)
            line_error = tilt_error

        # end points of the line, and the segments, in the rotated frame
        x_used = columns[covered] if np.any(covered) else columns
        x_ends = np.array([x_used.min(), x_used.max()], dtype=float)
        y_ends = intercept + slope * (x_ends - x_centre)
        ends = np.column_stack([x_ends, y_ends])
        segments = np.column_stack([x_seg, y_seg]) if len(x_seg) else np.empty((0, 2))
        if family != FAMILIES[0]:
            ends, segments = ends[:, ::-1], segments[:, ::-1]

        lines.append(
            {
                "position": float(peak - centre),
                "sign": int(sign),
                "significance": float(abs(detrended[peak]) / noise),
                "angle": line_angle,
                "angle_error": line_error,
                "n_segments": len(x_seg),
                "ends": rotated_to_image(ends, angle, rotated_mask.shape
                    if family == FAMILIES[0] else rotated_mask.T.shape, residual.shape),
                "segments": rotated_to_image(segments, angle, rotated_mask.shape
                    if family == FAMILIES[0] else rotated_mask.T.shape, residual.shape)
                    if len(segments) else segments,
            }
        )

    return lines, profile, detrended


def weighted_line_angle(lines):
    """
    Weighted mean (1 / error^2) of the angles of the lines with a fit, and its
    error; NaN without lines.
    """

    angles = np.array([l["angle"] for l in lines], dtype=float)
    errors = np.array([l["angle_error"] for l in lines], dtype=float)
    use = np.isfinite(angles) & np.isfinite(errors) & (errors > 0)
    if not np.any(use):
        return np.nan, np.nan
    weights = 1.0 / errors[use] ** 2

    return np.sum(weights * angles[use]) / np.sum(weights), 1.0 / np.sqrt(np.sum(weights))


def power_spectrum(residual, mask):
    """
    Power spectrum (shifted, log10) of the residual with a Hann window, for the
    QA plot.
    """

    ny, nx = residual.shape
    window = np.outer(np.hanning(ny), np.hanning(nx))
    power = np.abs(np.fft.fftshift(np.fft.fft2(residual * mask * window))) ** 2

    return np.log10(power + 1e-12 * power.max())


def plot_stripe_qa(
    residual,
    mask,
    measurement,
    quadrants,
    lines,
    profiles,
    nominal,
    title,
    save_path,
):
    """
    QA plot of one exposure: residual with all lines found (each at its own
    fitted angle), variance curves with the fits, angles of the individual
    lines, power spectrum, profiles with the lines found, histogram of the
    line angles.

    Parameters
    ----------
    lines : list of 2 lists
        The lines of the two families (``find_lines``).

    profiles : list of 2 tuples
        (profile, detrended) of the two families (``find_lines``).
    """

    import matplotlib.pyplot as plt

    angle = measurement["angle"]
    colours = ("tab:blue", "tab:orange")

    fig, axes = plt.subplots(2, 3, figsize=(21, 12.5))

    # residual with all lines, North up, East left
    ax = axes[0, 0]
    shown = np.where(mask > 0, residual, np.nan)
    low, high = np.nanpercentile(shown, [1, 99])
    ax.imshow(shown, origin="lower", cmap="gray", vmin=low, vmax=high)
    for k in range(2):
        for line in lines[k]:
            ends = line["ends"]
            fitted = np.isfinite(line["angle"])
            ax.plot(ends[:, 0], ends[:, 1], "-" if fitted else ":", color=colours[k],
                    lw=0.8, alpha=0.9)
            if len(line["segments"]):
                ax.plot(line["segments"][:, 0], line["segments"][:, 1], ".",
                        color=colours[k], ms=2)
    ax.set_xlim(0, residual.shape[1] - 1)
    ax.set_ylim(0, residual.shape[0] - 1)
    # compass
    ax.annotate("", xy=(0.92, 0.2), xytext=(0.92, 0.05), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color="red"))
    ax.text(0.92, 0.21, "N", color="red", transform=ax.transAxes, ha="center")
    ax.annotate("", xy=(0.77, 0.05), xytext=(0.92, 0.05), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color="red"))
    ax.text(0.75, 0.05, "E", color="red", transform=ax.transAxes, ha="right", va="center")
    ax.set_title(
        f"Residual with the lines found ({len(lines[0])} horizontal, "
        f"{len(lines[1])} vertical;\ndots: segment positions, dotted: no angle fit)"
    )

    # residual alone, for comparison
    ax = axes[1, 2]
    ax.imshow(shown, origin="lower", cmap="gray", vmin=low, vmax=high)
    ax.set_title("Residual")

    # variance curves
    ax = axes[0, 1]
    angles = measurement["angles"]
    for k in range(2):
        scores = measurement["scores"][:, k]
        norm = np.nanmax(scores)
        ax.plot(angles, scores / norm, color=colours[k], label=f"{FAMILIES[k]} lines")
        if measurement["fit"][k] is not None:
            ax.plot(angles, _gaussian(angles, *measurement["fit"][k]) / norm, ":",
                    color=colours[k])
        if np.isfinite(angle[k]):
            ax.axvline(angle[k], color=colours[k], ls="--")
        for q in quadrants[:, k]:
            if np.isfinite(q):
                ax.plot(q, 0.02 + 0.03 * k, "|", color=colours[k], ms=15, mew=2)
    ax.axvline(nominal, color="k", ls=":", label="nominal (POSANG)")
    ax.set_xlabel("angle [deg] (counter-clockwise, North up East left)")
    ax.set_ylabel("variance of the profile (normalised)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left")
    ax.set_title("Variance vs. angle (ticks: quadrants, dotted: Gaussian fit)")

    # angles of the individual lines
    ax = axes[0, 2]
    for k in range(2):
        fitted = [l for l in lines[k] if np.isfinite(l["angle"])]
        if fitted:
            ax.errorbar([l["position"] for l in fitted], [l["angle"] for l in fitted],
                        yerr=[l["angle_error"] for l in fitted], fmt="o", ms=4,
                        color=colours[k], capsize=2, label=f"{FAMILIES[k]} lines")
            mean, error = weighted_line_angle(fitted)
            if np.isfinite(mean):
                ax.axhspan(mean - error, mean + error, color=colours[k], alpha=0.2)
        if np.isfinite(angle[k]):
            ax.axhline(angle[k], color=colours[k], ls="--")
    ax.axhline(nominal, color="k", ls=":")
    ax.set_xlabel("position of the line (pixels from the centre, perpendicular)")
    ax.set_ylabel("angle of the line [deg]")
    ax.set_ylim(nominal - 5, nominal + 5)
    ax.legend(loc="upper right")
    ax.set_title("Individual lines (band: weighted mean, dashed: variance peak)")

    # power spectrum, lines perpendicular to the families
    ax = axes[1, 0]
    spectrum = power_spectrum(residual, mask)
    ax.imshow(spectrum, origin="lower", cmap="magma",
              vmin=np.percentile(spectrum, 50), vmax=np.percentile(spectrum, 99.9))
    sy, sx = (spectrum.shape[0] - 1) / 2, (spectrum.shape[1] - 1) / 2
    for k in range(2):
        if not np.isfinite(angle[k]):
            continue
        a = np.deg2rad(angle[k] + 90.0 * k + 90.0)
        r = 0.45 * min(spectrum.shape)
        ax.plot([sx - r * np.cos(a), sx + r * np.cos(a)],
                [sy - r * np.sin(a), sy + r * np.sin(a)], "--", color=colours[k], lw=1)
    ax.set_title("Power spectrum (Hann window); dashed: perpendicular to the lines")

    # profiles with the lines found
    ax = axes[1, 1]
    for k in range(2):
        profile, detrended = profiles[k]
        if profile is None:
            continue
        offset = 4 * np.nanstd(detrended) * k
        x = np.arange(len(profile)) - (len(profile) - 1) / 2
        ax.plot(x, detrended + offset, color=colours[k],
                label=f"across the {FAMILIES[k]} lines (detrended)")
        for line in lines[k]:
            i = int(round(line["position"] + (len(profile) - 1) / 2))
            ax.plot(line["position"], detrended[i] + offset, "v" if line["sign"] < 0 else "^",
                    color="k", ms=4)
    ax.set_xlabel("pixels from the centre (rotated frame)")
    ax.set_ylabel("mean residual (offset for clarity)")
    ax.legend(loc="upper right")
    ax.set_title("Profiles at the measured angles (markers: lines found)")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)


def plot_summary(results, save_path):
    """
    Deviations of the line families from the nominal orientation for all
    exposures, with ROTANGLE of the table and the aligned frame check.
    """

    import matplotlib.pyplot as plt

    labels = [f"{r['pointing']}.{r['dither']}" for r in results]
    x = np.arange(len(results))

    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(results) + 4), 6))

    for k, (family, colour, dx) in enumerate(
        zip(FAMILIES, ("tab:blue", "tab:orange"), (-0.15, 0.15))
    ):
        ax.errorbar(
            x + dx,
            [r[f"deviation_{family}"] for r in results],
            yerr=[r[f"error_{family}"] for r in results],
            fmt="o",
            color=colour,
            capsize=3,
            label=f"{family} lines (native frame)",
        )
    ax.plot(x, [r["rotangle"] for r in results], "k*", ms=12,
            label="ROTANGLE of the table (sky rotation of the exposure)")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("pointing.dither")
    ax.set_ylabel("deviation from POSANG [deg] (counter-clockwise, North up East left)")
    ax.legend()
    ax.set_title("Orientation of the slicer pattern relative to the pixel grid")
    fig.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)


if __name__ == "__main__":

    import argparse
    import csv
    import time

    start_time = time.perf_counter()

    parser = argparse.ArgumentParser(
        description="Estimate the orientation of the MUSE slicer pattern from the "
        "residual stripes in the difference images of a finished run."
    )

    parser.add_argument(
        "output_dir",
        type=str,
        help="Output directory of the run with the output table (for a hybrid run: "
        "its top directory, or a method sub-directory with --table).",
    )
    parser.add_argument(
        "config_file",
        type=str,
        help="Configuration file of the run (in the order of the rows of the table).",
    )
    parser.add_argument(
        "--table",
        type=str,
        default="Hybrid",
        help="Method name of the output table (<table>_alignment_output_table.fits). "
        "Default is 'Hybrid'.",
    )
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Directory of the exposures. Default: run manifest.")
    parser.add_argument("--ref_file_path", type=str, default=None,
                        help="Reference file. Default: run manifest.")
    parser.add_argument("--ref_filter", type=str, default="WFI_BB",
                        help="Filter of the reference. Default is 'WFI_BB'.")
    parser.add_argument("--ref_unit", type=str, default=None,
                        help="Unit of the reference, used instead of its BUNIT.")
    parser.add_argument("--pointings", type=str, nargs="+", default=None,
                        help="Pointings to measure. Default: all.")
    parser.add_argument("--highpass", type=int, default=15,
                        help="Size of the high pass median filter (pixels). Default 15.")
    parser.add_argument("--edge", type=int, default=12,
                        help="Pixels removed along the edges of the data. Default 12.")
    parser.add_argument("--clip", type=float, default=4.0,
                        help="Outlier clip (MAD standard deviations). Default 4.")
    parser.add_argument("--smooth", type=float, default=1.0,
                        help="Gaussian smoothing sigma (pixels). Default 1.")
    parser.add_argument("--half_range", type=float, default=2.5,
                        help="Half width of the fine angle scan (deg). Default 2.5.")
    parser.add_argument("--step", type=float, default=0.02,
                        help="Step of the fine angle scan (deg). Default 0.02.")
    parser.add_argument("--line_threshold", type=float, default=3.0,
                        help="Detection threshold of the individual lines (MAD standard "
                        "deviations of the detrended profile). Default 3.")
    parser.add_argument("--line_detrend", type=int, default=9,
                        help="Running median (pixels) subtracted from the profile for "
                        "the line detection only. Default 9.")
    parser.add_argument("--line_segment", type=int, default=30,
                        help="Length (pixels) of the segments for the angle of each "
                        "line. Default 30.")
    parser.add_argument("--no_check", action="store_true",
                        help="Skip the measurement in the frame of the reference.")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Default: <output_dir>/stripe_orientation.")

    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")

    from astropy.io import fits
    from astropy.table import Table

    from logger import get_logger, log_run_environment, format_elapsed
    from IO import read_fits_image, extract_image_wcs
    from geometry import sky_translation_to_detector_translation
    from flux import (
        reference_conversion_factor,
        fit_flux_normalisation,
        shift_image_with_nans,
    )

    out_dir = pathlib.Path(
        args.out_dir or os.path.join(args.output_dir, "stripe_orientation")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(out_dir)
    log_run_environment(logger)

    with open(args.config_file, "r") as f:
        config = json.load(f)

    manifest = find_run_manifest(args.output_dir)
    input_dir = args.input_dir or manifest.get("input_dir")
    reference_file = args.ref_file_path or manifest.get("reference_file")
    if input_dir is None or reference_file is None:
        raise ValueError("No run manifest found: give --input_dir and --ref_file_path.")

    table_file = pathlib.Path(args.output_dir) / f"{args.table}_alignment_output_table.fits"
    table = Table.read(table_file, mask_invalid=False)

    logger.info(f"Stripe orientation from {table_file}")
    logger.info(f"Configuration: {args.config_file}")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Reference file: {reference_file}")
    logger.info(
        f"Settings: highpass {args.highpass}, edge {args.edge}, clip {args.clip}, "
        f"smooth {args.smooth}, scan +-{args.half_range} deg in steps of {args.step} deg"
    )

    n_rows = sum(len(e) for p, e in config.items() if p != "headers")
    if n_rows != len(table):
        raise ValueError(
            f"{table_file} has {len(table)} rows, but {args.config_file} has {n_rows} "
            f"exposures."
        )

    reference_wcs = extract_image_wcs(reference_file, logger=logger)
    read = read_fits_image(reference_file, logger=logger)
    if reference_wcs is None or read is None:
        raise ValueError(f"Could not read the reference {reference_file}.")
    reference_data, reference_hdul = read
    reference_data = reference_data * reference_conversion_factor(
        args.ref_filter, header=reference_hdul[0].header, logger=logger,
        unit=args.ref_unit,
    )

    results = []
    line_results = []

    row = 0
    for pointing, exposures in config.items():

        if pointing == "headers":
            continue

        rows = list(range(row, row + len(exposures)))
        row += len(exposures)

        if args.pointings is not None and pointing not in args.pointings:
            continue

        files = [os.path.join(input_dir, exposure[2]) for exposure in exposures]

        try:
            arrays, reference, wcs = prepare_pointing(
                files, reference_data, reference_wcs, logger=logger
            )
        except Exception:
            logger.exception(f"Pointing {pointing}: could not prepare the frames, skipping.")
            continue

        for i, r in enumerate(rows):

            name = f"pointing {pointing}, dither {i+1}"
            offsets = [float(table[c][r]) for c in ("RA_OFFSET", "DEC_OFFSET", "ROTANGLE")]
            if not np.all(np.isfinite(offsets)):
                logger.info(f"{name}: no result, skipping.")
                continue

            try:
                tx, ty = sky_translation_to_detector_translation(wcs, offsets[0], offsets[1])
                rotation_angle = offsets[2]

                flux_scale, background = fit_flux_normalisation(
                    shift_image_with_nans(arrays[i], tx, ty, rotation_angle),
                    reference,
                    logger=logger,
                    name=f"({name})",
                )
                if not np.isfinite(flux_scale):
                    logger.warning(f"{name}: flux normalisation not usable, skipping.")
                    continue

                reference_flux = reference / flux_scale - background

                posang = fits.getheader(files[i]).get(POSANG_KEYWORD, np.nan)
                nominal = float(wrap_angle(posang)) if np.isfinite(posang) else 0.0
                if not np.isfinite(posang):
                    logger.warning(f"{name}: no {POSANG_KEYWORD}, nominal angle 0.")

                # native frame: the reference moved onto the exposure
                residual, mask = prepare_residual(
                    arrays[i] - reference_on_exposure(reference_flux, tx, ty, rotation_angle),
                    args.highpass, args.edge, args.clip, args.smooth,
                )
                coarse = coarse_line_angle(residual, mask)
                if abs(wrap_angle(coarse - nominal)) > args.half_range:
                    logger.warning(
                        f"{name}: the coarse grid angle {coarse:.2f} deg is more than "
                        f"{args.half_range} deg from the nominal {nominal:.2f} deg; "
                        f"scanning around the coarse angle."
                    )
                    centre = coarse
                else:
                    centre = nominal

                measurement = measure_line_angles(
                    residual, mask, centre, args.half_range, args.step
                )
                quadrants = quadrant_angles(residual, mask, centre, args.half_range)
                errors = angle_error(quadrants)
                angle = measurement["angle"]
                deviation = wrap_angle(angle - nominal)
                pa = position_angles(angle)

                # check: frame of the reference (exposure rotated by ROTANGLE)
                check = np.full(2, np.nan)
                if not args.no_check:
                    aligned_residual, aligned_mask = prepare_residual(
                        shift_image_with_nans(arrays[i], tx, ty, rotation_angle)
                        - reference_flux,
                        args.highpass, args.edge, args.clip, args.smooth,
                    )
                    check = measure_line_angles(
                        aligned_residual, aligned_mask, centre - rotation_angle,
                        args.half_range, args.step,
                    )["angle"]

                # the individual lines of both families, at the measured angles
                lines, profiles = [], []
                for k, family in enumerate(FAMILIES):
                    if not np.isfinite(angle[k]):
                        lines.append([])
                        profiles.append((None, None))
                        continue
                    found, profile, detrended = find_lines(
                        residual, mask, angle[k], family,
                        threshold=args.line_threshold, detrend=args.line_detrend,
                        segment=args.line_segment,
                    )
                    lines.append(found)
                    profiles.append((profile, detrended))
                line_mean = [weighted_line_angle(found) for found in lines]

                method = (
                    str(table[METHOD_COLUMN][r]).strip()
                    if METHOD_COLUMN in table.colnames
                    else args.table
                )

                logger.info(
                    f"{name} ({exposures[i][2]}): POSANG {posang}, {method} ROTANGLE "
                    f"{rotation_angle:+.3f} deg, coarse grid angle {coarse:+.2f} deg"
                )
                for k, family in enumerate(FAMILIES):
                    logger.info(
                        f"    {family} lines: angle {angle[k]:+.3f} +- {errors[k]:.3f} deg "
                        f"(quadrants {np.array2string(quadrants[:, k], precision=3)}), "
                        f"PA {pa[k]:.3f} deg, deviation from POSANG {deviation[k]:+.3f} deg"
                        + (
                            f"; frame of the reference {check[k]:+.3f} deg (expected "
                            f"{angle[k] - rotation_angle:+.3f})"
                            if not args.no_check
                            else ""
                        )
                    )
                    fitted = [l for l in lines[k] if np.isfinite(l["angle"])]
                    logger.info(
                        f"    {family} lines: {len(lines[k])} lines found, {len(fitted)} with "
                        f"an angle; weighted mean angle {line_mean[k][0]:+.3f} +- "
                        f"{line_mean[k][1]:.3f} deg"
                        + (
                            f", median {np.median([l['angle'] for l in fitted]):+.3f} deg"
                            if fitted
                            else ""
                        )
                    )
                    for line in lines[k]:
                        line_results.append(
                            {
                                "pointing": pointing,
                                "dither": i + 1,
                                "family": family,
                                "position": line["position"],
                                "sign": line["sign"],
                                "significance": line["significance"],
                                "angle": line["angle"],
                                "angle_error": line["angle_error"],
                                "n_segments": line["n_segments"],
                            }
                        )

                plot_stripe_qa(
                    residual,
                    mask,
                    measurement,
                    quadrants,
                    lines,
                    profiles,
                    nominal,
                    f"Stripe orientation, {name} ({exposures[i][2]})\n"
                    f"POSANG {posang}; horizontal {deviation[0]:+.3f} +- {errors[0]:.3f}, "
                    f"vertical {deviation[1]:+.3f} +- {errors[1]:.3f} deg from nominal; "
                    f"{method} ROTANGLE {rotation_angle:+.3f} deg",
                    out_dir / f"stripe_orientation_{pointing}_dither_{i+1}.png",
                )

                result = {
                    "pointing": pointing,
                    "dither": i + 1,
                    "file": exposures[i][2],
                    "method": method,
                    "posang": posang,
                    "rotangle": rotation_angle,
                    "flux_scale": flux_scale,
                    "background": background,
                    "coarse_angle": coarse,
                }
                for k, family in enumerate(FAMILIES):
                    result[f"angle_{family}"] = angle[k]
                    result[f"error_{family}"] = errors[k]
                    result[f"pa_{family}"] = pa[k]
                    result[f"deviation_{family}"] = deviation[k]
                    result[f"aligned_angle_{family}"] = check[k]
                    result[f"n_lines_{family}"] = len(lines[k])
                    result[f"line_angle_{family}"] = line_mean[k][0]
                    result[f"line_angle_error_{family}"] = line_mean[k][1]
                results.append(result)

            except Exception:
                logger.exception(f"{name}: unexpected error, continuing.")

    if not results:
        logger.error("No exposure was measured.")
        sys.exit(1)

    csv_file = out_dir / "stripe_orientation.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"Wrote {csv_file}")

    if line_results:
        lines_file = out_dir / "stripe_lines.csv"
        with open(lines_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(line_results[0].keys()))
            writer.writeheader()
            writer.writerows(line_results)
        logger.info(f"Wrote {lines_file}")

    plot_summary(results, out_dir / "stripe_orientation_summary.png")

    for family in FAMILIES:
        d = np.array([r[f"deviation_{family}"] for r in results], dtype=float)
        d = d[np.isfinite(d)]
        if len(d):
            logger.info(
                f"Summary, {family} lines: deviation from POSANG median {np.median(d):+.3f} "
                f"deg, mean {np.mean(d):+.3f}, std {np.std(d):.3f} ({len(d)} exposures)"
            )
    logger.info(f"Finished in {format_elapsed(time.perf_counter() - start_time)}")
