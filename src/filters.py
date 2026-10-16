from venv import logger
from stats import longest_run

import numpy as np

"""
Module for filtering and processing images.
"""


def remove_nans(image, fill_value=0, logger=None):
    """
    Wrapper function to np.nan_to_num to replace NaN values in the image with zeros.
    """

    if logger:
        logger.info(f"Replacing NaN values in the image with fill_value: {fill_value}")

    return np.nan_to_num(image, nan=fill_value)


def remove_bright_sources_simple(
    image, threshold=500, radius=15, fill_value=0, logger=None, mask=None
):
    """
    Remove bright sources from the image by setting pixels above a certain threshold to a fill value
    including a radius around the bright source. This is a simple approach and may not be suitable for all cases.
    """

    if logger:
        logger.info(
            f"Removing bright sources with threshold: {threshold}, radius: {radius}, and fill_value: {fill_value}"
        )

    if mask is None:
        # Create a mask for pixels above the threshold
        mask = image > threshold

    else:
        mask = mask | (image > threshold)

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

def _shrink(finite, window, min_run):
    """
    Shrink window = [top, bottom, left, right] (trims from each side) until
    every edge passes. min_run=None requires the whole edge to be finite.
    """
    nrows, ncols = finite.shape
    top, bottom, left, right = window

    def ok(edge):
        if min_run is None:
            return edge.all()
        return longest_run(edge) >= min(min_run, edge.size)

    changed = True
    while changed:
        changed = False
        r0, r1 = top, nrows - bottom
        c0, c1 = left, ncols - right
        if r1 - r0 < 1 or c1 - c0 < 1:
            raise ValueError("Window collapsed; no valid region found.")

        if not ok(finite[r0, c0:c1]):      top += 1;    changed = True
        if not ok(finite[r1 - 1, c0:c1]):  bottom += 1; changed = True
        if not ok(finite[r0:r1, c0]):      left += 1;   changed = True
        if not ok(finite[r0:r1, c1 - 1]):  right += 1;  changed = True

    return [top, bottom, left, right]

def estimate_nan_window(image, min_run=100):
    finite = np.isfinite(image)
    nrows, ncols = image.shape

    # Pass 1: each edge needs >= min_run consecutive finite pixels
    w1 = _shrink(finite, [0, 0, 0, 0], min_run)
    print("Pass 1 (run criterion): top=%d, bottom=%d, left=%d, right=%d" % tuple(w1))

    # Pass 2: continue from pass-1 window, each edge must contain no NaN at all
    w2 = _shrink(finite, w1, None)
    print("Pass 2 (no NaN on edges): top=%d, bottom=%d, left=%d, right=%d" % tuple(w2))

    top, bottom, left, right = w2
    height = nrows - top - bottom
    width = ncols - left - right
    print(f"Final window: height={height}, width={width}")

    return (top, bottom, left, right), (height, width)
