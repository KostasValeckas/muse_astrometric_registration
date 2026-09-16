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
