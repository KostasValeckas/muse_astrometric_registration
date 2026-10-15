from venv import logger

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
