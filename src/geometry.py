import numpy as np
from astropy.wcs import WCS
from reproject import reproject_interp, reproject_exact, reproject_adaptive
from astropy.wcs.utils import proj_plane_pixel_scales

"""
Module for reprojections and WCS conversions.
"""

def make_window_wcs(wcs, shape, window, oversample_factor=1):
    """
    Slice `wcs` (for an image of `shape`) to the window (top, bottom, left, right),
    then oversample. Returns (wcs_out, shape_out) on the cropped grid.
    """
    ny, nx = shape
    top, bottom, left, right = (int(w) for w in window)

    wcs_out = wcs.slice((slice(top, ny - bottom), slice(left, nx - right)))
    shape_out = (ny - top - bottom, nx - left - right)

    if oversample_factor > 1:
        if wcs_out.wcs.has_cd():
            wcs_out.wcs.cd = wcs_out.wcs.cd / oversample_factor
        else:
            wcs_out.wcs.cdelt = wcs_out.wcs.cdelt / oversample_factor
        wcs_out.wcs.crpix = (np.array(wcs_out.wcs.crpix) - 0.5) * oversample_factor + 0.5
        shape_out = (shape_out[0] * oversample_factor, shape_out[1] * oversample_factor)

    wcs_out.array_shape = shape_out
    return wcs_out, shape_out


def reproject_image_to_target_wcs(
    source_image,
    source_wcs,
    target_wcs,
    oversample_factor=1,
    method="interpolation",
    window=None,
    shape_out=None,
    logger=None,
):
    """
    window : (top, bottom, left, right), optional
        Trims in *source* pixels from estimate_nan_window. If given, the
        output shape is the window shape and the output grid is centred on
        the window centre. shape_out is ignored.
    """

    target_wcs = target_wcs.deepcopy()

    if window is not None:
        ny, nx = source_image.shape
        top, bottom, left, right = (int(w) for w in window)
        height = ny - top - bottom
        width = nx - left - right

        # centre of the clean window in source pixels (0-based) -> sky
        cx = left + (width - 1) / 2
        cy = top + (height - 1) / 2
        crval = source_wcs.wcs_pix2world([[cx, cy]], 0)[0]

        # account for a different pixel scale between source and target
        ratio = (proj_plane_pixel_scales(source_wcs)[0]
                 / proj_plane_pixel_scales(target_wcs)[0])
        out_h = int(round(height * ratio))
        out_w = int(round(width * ratio))

        target_wcs.wcs.crval = crval
        target_wcs.wcs.crpix = [(out_w + 1) / 2, (out_h + 1) / 2]  # 1-based FITS
        shape_out = (out_h, out_w)

    elif shape_out is None:
        if target_wcs.array_shape is None:
            raise ValueError("Need window, shape_out or target_wcs.array_shape.")
        shape_out = tuple(target_wcs.array_shape)

    if oversample_factor > 1:
        if target_wcs.wcs.has_cd():
            target_wcs.wcs.cd = target_wcs.wcs.cd / oversample_factor
        else:
            target_wcs.wcs.cdelt = target_wcs.wcs.cdelt / oversample_factor
        target_wcs.wcs.crpix = (np.array(target_wcs.wcs.crpix) - 0.5) * oversample_factor + 0.5
        shape_out = (shape_out[0] * oversample_factor, shape_out[1] * oversample_factor)

    target_wcs.array_shape = shape_out

    if logger:
        logger.info(f"Reprojecting to shape {shape_out} using '{method}'...")

    funcs = {"interpolation": reproject_interp,
             "exact": reproject_exact,
             "adaptive": reproject_adaptive}
    if method not in funcs:
        if logger:
            logger.error(f"Invalid reprojection method: {method}. Returning source image.")
        return source_image, np.zeros(shape_out)

    return funcs[method]((source_image, source_wcs), target_wcs, shape_out=shape_out)


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


def detector_translation_to_sky_translation(wcs, dx, dy, logger=None):
    """
    Convert a detector/pixel translation (dx, dy) into
    RA/Dec coordinate offsets using the image WCS.

    Parameters
    ----------
    wcs : astropy.wcs.WCS
        WCS of the image.
    dx, dy : float
        Translation in pixels.

    Returns
    -------
    dRA : float
        RA offset in degrees.
    dDec : float
        Dec offset in degrees.
    """

    # WCS reference pixel, converted from FITS 1-based
    # coordinates to Astropy 0-based pixel coordinates.
    x0 = wcs.wcs.crpix[0] - 1
    y0 = wcs.wcs.crpix[1] - 1

    # Sky coordinate at reference position
    sky0 = wcs.pixel_to_world(x0, y0)

    # Sky coordinate after applying the detector translation
    sky1 = wcs.pixel_to_world(x0 + dx, y0 + dy)

    # RA difference, wrapped to [-180, +180] degrees
    dRA = (sky1.ra.deg - sky0.ra.deg + 180.0) % 360.0 - 180.0

    # Dec difference
    dDec = sky1.dec.deg - sky0.dec.deg

    if logger:
        logger.info(
            f"Converted detector translation (dx={dx}, dy={dy}) to sky offsets (dRA={dRA}, dDec={dDec})"
        )

    return dRA, dDec
