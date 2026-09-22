import numpy as np
from astropy.wcs import WCS
from reproject import reproject_interp, reproject_exact, reproject_adaptive

"""
Module for reprojections and WCS conversions.
"""


def reproject_image_to_target_wcs(
    source_image,
    source_wcs,
    target_wcs,
    shape_out=None,
    oversample_factor=1,
    method="interpolation",
    logger=None,
):
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
        logger.info(
            f"Reprojecting image to target WCS with shape {shape_out} using method '{method}'..."
        )

    if method == "interpolation":
        reprojected_image, footprint = reproject_interp(
            (source_image, source_wcs), target_wcs, shape_out=shape_out
        )
    elif method == "exact":
        reprojected_image, footprint = reproject_exact(
            (source_image, source_wcs), target_wcs, shape_out=shape_out
        )
    elif method == "adaptive":
        reprojected_image, footprint = reproject_adaptive(
            (source_image, source_wcs), target_wcs, shape_out=shape_out
        )
    else:
        if logger:
            logger.error(
                f"Invalid reprojection method: {method}. Choose from 'interpolation', 'exact', or 'adaptive'."
            )
            logger.error("Returning the source image without reprojection...")
            return source_image, np.zeros(shape_out)

    if logger:
        logger.info("Reprojection completed successfully.")

    return reprojected_image, footprint


# some helper methods for euclidean transforms

import numpy as np


def transformation_matrix(tx, ty, rotation_deg, centre):
    """
    Convert the alignment result returned by AlignmentMethod.run()
    into a 3x3 homogeneous transformation matrix.

    The transformation is:

        p_target =
            centre + R(theta) @ (p_input - centre) + [tx, ty]

    where rotation_deg is the rotation returned by run().
    """

    # Your AlignmentMethod returns:
    #
    #     rot = -degrees(theta)
    #
    # so convert back to theta here.
    theta = np.radians(-rotation_deg)

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    R = np.array(
        [
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta],
        ]
    )

    centre = np.asarray(centre, dtype=float)
    translation = np.array([tx, ty], dtype=float)

    # Convert the centre-based transformation into:
    #
    # p_target = R @ p_input + b
    #
    b = centre + translation - R @ centre

    T = np.eye(3)

    T[:2, :2] = R
    T[:2, 2] = b

    return T


def compose_transformations(T1, T2):
    """
    Compose two transformations.

    T1 maps:
        frame A -> frame B

    T2 maps:
        frame B -> frame C

    Returns:
        frame A -> frame C
    """

    return T2 @ T1


def inverse_transformation(T):
    """
    Return the inverse transformation.
    """

    return np.linalg.inv(T)


def matrix_to_parameters(T, centre):
    """
    Convert a 3x3 homogeneous transformation back to the
    (tx, ty, rotation_deg) convention used by AlignmentMethod.
    """

    R = T[:2, :2]
    b = T[:2, 2]

    # Recover theta from R.
    theta = np.arctan2(R[1, 0], R[0, 0])

    # Your AlignmentMethod reports:
    #
    #     rot = -degrees(theta)
    #
    rotation_deg = -np.degrees(theta)

    centre = np.asarray(centre, dtype=float)

    # b = centre + t - R @ centre
    #
    # therefore:
    #
    # t = b - centre + R @ centre
    translation = b - centre + R @ centre

    tx, ty = translation

    return tx, ty, rotation_deg
