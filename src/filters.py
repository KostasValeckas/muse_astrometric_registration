"""Module for filtering and processing images."""

from venv import logger
from stats import longest_run
from scipy.ndimage import gaussian_filter, map_coordinates
from numpy.fft import fft2, fftshift
from scipy.signal.windows import tukey
from scipy.interpolate import griddata

import numpy as np


def remove_nans(image, fill_value=0, logger=None):
    """
    Wrapper function to np.nan_to_num to replace NaN values in the image with zeros.
    """

    if logger:
        logger.info(f"Replacing NaN values in the image with fill_value: {fill_value}")

    return np.nan_to_num(image, nan=fill_value)


def interpolate_nans(image, logger=None):
    """
    Interpolate NaN values in a 2D image using cubic interpolation.

    Falls back to nearest-neighbor interpolation for NaN pixels
    that cubic interpolation cannot fill, such as pixels outside
    the convex hull of valid points.
    """
    if logger:
        logger.info("Interpolating NaN values in the image using cubic interpolation.")

    image = np.asarray(image, dtype=float)
    nan_mask = np.isnan(image)

    # Return a copy if there are no NaNs
    if not np.any(nan_mask):
        return image.copy()

    # Coordinates of valid and missing pixels
    valid_coords = np.array(np.where(~nan_mask)).T
    nan_coords = np.array(np.where(nan_mask)).T

    # Values at valid pixels
    valid_values = image[~nan_mask]

    # Ensure there are enough valid points for interpolation
    if len(valid_values) < 4:
        if logger:
            logger.warning(
                "Too few valid pixels for cubic interpolation; "
                "using nearest-neighbor interpolation."
            )
        interpolated_values = griddata(
            valid_coords, valid_values, nan_coords, method="nearest"
        )
    else:
        # Cubic interpolation
        interpolated_values = griddata(
            valid_coords, valid_values, nan_coords, method="cubic"
        )

        # Fill any remaining NaNs using nearest-neighbor interpolation
        missing = np.isnan(interpolated_values)
        if np.any(missing):
            interpolated_values[missing] = griddata(
                valid_coords,
                valid_values,
                nan_coords[missing],
                method="nearest",
            )

    # Fill missing pixels without modifying the original image
    interpolated_image = image.copy()
    interpolated_image[nan_mask] = interpolated_values

    return interpolated_image


# default threshold of remove_bright_sources_simple, in robust standard
# deviations above the median of the frame. On the MUSE frames of NGC1365 the
# old absolute threshold (500, in MUSE units) was 7 - 118 robust sigmas above
# the median, ~55 on most frames.
BRIGHT_SOURCE_SIGMA = 50.0


def bright_source_threshold(image, threshold_sigma=BRIGHT_SOURCE_SIGMA):
    """
    Threshold for ``remove_bright_sources_simple`` relative to the frame:
    median + `threshold_sigma` * MAD standard deviation of its finite, non-zero
    pixels (zeros are filled NaNs or masked pixels). Scales with the frame, so
    it does not depend on the unit of the image.

    Returns
    -------
    float
        The threshold, or inf (nothing removed) if the frame has no valid
        pixels or no scatter.
    """

    from astropy.stats import mad_std

    values = np.asarray(image, dtype=float)
    values = values[np.isfinite(values) & (values != 0)]

    if values.size == 0:
        return np.inf

    scatter = mad_std(values)

    if not np.isfinite(scatter) or scatter <= 0:
        return np.inf

    return float(np.median(values) + threshold_sigma * scatter)


def remove_bright_sources_simple(
    image,
    threshold=None,
    radius=15,
    fill_value=0,
    logger=None,
    mask=None,
    threshold_sigma=BRIGHT_SOURCE_SIGMA,
):
    """
    Remove bright sources from the image by setting pixels above a certain threshold to a fill value
    including a radius around the bright source. This is a simple approach and may not be suitable for all cases.

    The threshold is relative to the frame by default (see
    ``bright_source_threshold``): `threshold_sigma` robust standard deviations
    above its median, so the same sources are removed whatever the unit of the
    image. An absolute `threshold` (in the unit of the image; the old default
    was 500, for frames in MUSE units) is used instead if given. An absolute
    threshold removed the whole reference when it was 1000 times too bright
    (wrong BUNIT), and nothing when the reference was not converted to MUSE
    units.
    """

    if threshold is None:
        threshold = bright_source_threshold(image, threshold_sigma=threshold_sigma)
        source = f"median + {threshold_sigma:g} robust sigma"
    else:
        source = "absolute"

    if logger:
        logger.info(
            f"Removing bright sources with threshold: {threshold:.6g} ({source}), radius: {radius}, and fill_value: {fill_value}"
        )

    if mask is None:
        # Create a mask for pixels above the threshold
        mask = image > threshold

    else:
        mask = mask | (image > threshold)

    if logger:
        logger.info(f"{int(np.sum(mask))} pixels above the bright source threshold.")

    # Apply the mask to the image, setting pixels above the threshold to the fill value
    filtered_image = np.where(mask, fill_value, image)

    # Apply a circular mask around each bright source
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if mask[i, j]:
                # Create a circular mask
                y, x = np.ogrid[: image.shape[0], : image.shape[1]]
                circular_mask = (x - j) ** 2 + (y - i) ** 2 <= radius**2

                # Apply the circular mask to the filtered image
                filtered_image[circular_mask] = fill_value

    return filtered_image, mask


def mask_zero_as_bpm(image, logger=None):
    """
    Create a bad pixel mask (BPM) for the image where pixels with a value of zero are considered bad.
    """

    if logger:
        logger.info("Creating a bad pixel mask (BPM) for pixels with value zero.")

    # Create a BPM where pixels with value zero are marked as True (bad)
    bpm = image == 0

    masked_image = np.ma.masked_array(image, mask=bpm)

    return masked_image, bpm


def _shrink(finite, window, min_run, logger=None):
    """
    Shrink window = [top, bottom, left, right] (trims from each side) until
    every edge passes. min_run=None requires the whole edge to be finite.
    """
    nrows, ncols = finite.shape
    top, bottom, left, right = window

    def ok(edge):
        """True if the edge is usable: no NaN at all (`min_run` None), or a long enough finite run."""
        if min_run is None:
            return edge.all()
        return longest_run(edge) >= min(min_run, edge.size)

    changed = True
    while changed:
        changed = False
        r0, r1 = top, nrows - bottom
        c0, c1 = left, ncols - right
        if r1 - r0 < 1 or c1 - c0 < 1:
            if logger:
                logger.warning(
                    "Window collapsed; no valid region found. Returning original window."
                )
            return window

        if not ok(finite[r0, c0:c1]):
            top += 1
            changed = True
        if not ok(finite[r1 - 1, c0:c1]):
            bottom += 1
            changed = True
        if not ok(finite[r0:r1, c0]):
            left += 1
            changed = True
        if not ok(finite[r0:r1, c1 - 1]):
            right += 1
            changed = True

    return [top, bottom, left, right]


def estimate_nan_window(image, min_run=100, logger=None):
    """
    Find the window of the image whose edges are free of NaNs.

    The edges are shrunk in two passes: first until every edge has at least
    `min_run` consecutive finite pixels, then until no edge contains a NaN.

    Parameters
    ----------
    image : 2D array
        Image with non-finite borders.
    min_run : int
        Minimum number of consecutive finite pixels per edge in the first pass.
    logger : logging.Logger, optional
        Logger for the progress messages.

    Returns
    -------
    (top, bottom, left, right), (height, width)
        Number of rows / columns cut from each side, and the size of the window.
    """

    if logger:
        logger.info(
            f"Estimating window of finite pixels with minimum finite array size: {min_run}"
        )

    finite = np.isfinite(image)
    nrows, ncols = image.shape

    # Pass 1: each edge needs >= min_run consecutive finite pixels
    w1 = _shrink(finite, [0, 0, 0, 0], min_run, logger=logger)
    if logger:
        logger.info(
            "Pass 1 (run criterion): top=%d, bottom=%d, left=%d, right=%d" % tuple(w1)
        )

    # Pass 2: continue from pass-1 window, each edge must contain no NaN at all
    w2 = _shrink(finite, w1, None, logger=logger)
    if logger:
        logger.info(
            "Pass 2 (no NaN on edges): top=%d, bottom=%d, left=%d, right=%d" % tuple(w2)
        )

    top, bottom, left, right = w2
    height = nrows - top - bottom
    width = ncols - left - right
    if logger:
        logger.info(f"Final window: height={height}, width={width}")

    return (top, bottom, left, right), (height, width)


def dog_passband(sigma, ratio=1.6, level=0.2):
    """Frequency range (cycles/pixel) where the DoG response >= level * peak."""
    f = np.linspace(1e-4, 0.5, 4000)
    h = np.exp(-2 * (np.pi * sigma * f) ** 2) - np.exp(
        -2 * (np.pi * ratio * sigma * f) ** 2
    )
    keep = f[h >= level * h.max()]
    return keep[0], keep[-1]


def dog(img, sigma, ratio=1.6):
    """DoG with a fixed sigma *ratio*: sigma sets the band centre, ratio the
    (fractional) bandwidth.  Absolute sigma differences only scale amplitude."""
    return gaussian_filter(img, sigma, mode="nearest") - gaussian_filter(
        img, ratio * sigma, mode="nearest"
    )


# --------------------------------------------------------------------------
# spectrum -> rings -> angular correlation
# --------------------------------------------------------------------------
def polar_rings(img, f_lo, f_hi, n_theta, pad=2):
    """Whitened rings of sqrt(power spectrum) on a linear (|f|, theta) grid.

    * theta covers [0, 180 deg) only (power spectrum is point-symmetric),
      so the angular axis is exactly periodic with period 180 deg.
    * 2x zero padding makes |F|^2 (FT of the autocorrelation) essentially
      band-limited, so cubic interpolation of it is accurate.
    * Each ring is mean-subtracted and unit-variance so that the steep radial
      power-law of the spectrum cannot dominate the correlation.
    """
    ny, nx = img.shape
    P = pad * max(ny, nx)
    pw = np.abs(fftshift(fft2(img, s=(P, P)))) ** 2
    c = P // 2  # DC bin after fftshift
    n_r = max(8, int(round((f_hi - f_lo) * P)))
    f = np.linspace(f_lo, f_hi, n_r)
    theta = np.arange(n_theta) * (np.pi / n_theta)
    rr = (f * P)[:, None]
    rows = c + rr * np.sin(theta)[None, :]
    cols = c + rr * np.cos(theta)[None, :]
    ring = map_coordinates(pw, [rows, cols], order=3, mode="nearest")
    ring = np.sqrt(np.clip(ring, 0, None))
    ring -= ring.mean(axis=1, keepdims=True)
    sd = ring.std(axis=1, keepdims=True)
    ring /= np.where(sd > 0, sd, 1.0)
    return ring, f


def radial_window(shape, taper=0.4):
    """Rotation-invariant (disk) Tukey window, centred on (N-1)/2 like
    scipy.ndimage.rotate.  Unlike a separable Hann it has no frame-fixed
    cross in its spectrum, and a disk inscribed in the frame never sees the
    NaN/empty corners that rotation creates."""
    ny, nx = shape
    y, x = np.indices(shape)
    r = np.hypot(y - (ny - 1) / 2, x - (nx - 1) / 2)
    r_out = min(ny, nx) / 2
    r_in = r_out * (1 - taper)
    w = np.ones(shape)
    ramp = (r > r_in) & (r < r_out)
    w[ramp] = 0.5 * (1 + np.cos(np.pi * (r[ramp] - r_in) / (r_out - r_in)))
    w[r >= r_out] = 0.0
    return w


def tukey2d(shape, taper=0.25):
    """2D Tukey (tapered cosine) window of `shape`, the outer product of two 1D windows."""
    return np.outer(tukey(shape[0], taper), tukey(shape[1], taper))
