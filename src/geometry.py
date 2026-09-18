import numpy as np
from astropy.wcs import WCS
from reproject import reproject_interp, reproject_exact, reproject_adaptive

"""
Module for reprojections and WCS conversions.
"""


def reproject_image_to_target_wcs(source_image, source_wcs, target_wcs, shape_out=None, oversample_factor = 1, method='interpolation', logger=None):
    """
    Reprojects a source image to the WCS of a target image.

    Parameters
    ----------
    source_image : np.ndarray
        The source image data.
    source_wcs : astropy.wcs.WCS
        The WCS of the source image.
    target_wcs : astropy.wcs.WCS
        The WCS of the target image.
    shape_out : tuple, optional
        The shape of the output image. If None, it will be inferred from the target WCS.
    method : str, optional
        The reprojection method to use. Options are 'interpolation', 'exact', or 'adaptive'.
        Default is 'interpolation'.
        See https://reproject.readthedocs.io/en/stable/ for more details (reproject_interp, reproject_adaptive, reproject_exact).

    Returns
    -------
    reprojected_image : np.ndarray
        The reprojected image data.
    footprint : np.ndarray
        The footprint of the reprojected image.
    """

    if shape_out is None:
        shape_out = (target_wcs.array_shape[0], target_wcs.array_shape[1])


    if logger:
        logger.info(f"Reprojecting image to target WCS with shape {shape_out} using method '{method}'...")

    if method == 'interpolation':
        reprojected_image, footprint = reproject_interp((source_image, source_wcs), target_wcs, shape_out=shape_out)
    elif method == 'exact':
        reprojected_image, footprint = reproject_exact((source_image, source_wcs), target_wcs, shape_out=shape_out)
    elif method == 'adaptive':
        reprojected_image, footprint = reproject_adaptive((source_image, source_wcs), target_wcs, shape_out=shape_out)
    else:
        if logger:
            logger.error(f"Invalid reprojection method: {method}. Choose from 'interpolation', 'exact', or 'adaptive'.")
            logger.error("Returning the source image without reprojection...")
            return source_image, np.zeros(shape_out)

    if logger:
        logger.info("Reprojection completed successfully.")

    return reprojected_image, footprint