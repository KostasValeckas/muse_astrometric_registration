"""Module for reprojections and WCS conversions."""

import numpy as np
from astropy.wcs import WCS
from reproject import reproject_interp, reproject_exact, reproject_adaptive
from muse_astrometric_registration.filters import estimate_nan_window


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
        wcs_out.wcs.crpix = (
            np.array(wcs_out.wcs.crpix) - 0.5
        ) * oversample_factor + 0.5
        shape_out = (shape_out[0] * oversample_factor, shape_out[1] * oversample_factor)

    wcs_out.array_shape = shape_out
    return wcs_out, shape_out


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


def reproject_dithers(dither_set, wcs_set, logger=None, method="interpolation"):
    """
    Reprojects a stack of frames to the referrence frame of the first frame in the stack.

    Keeps the minimum width and height from all frames as the shape.
    """
    if logger:
        logger.info(
            f"Reprojecting dithered frames to the reference frame of the first frame using method '{method}'..."
        )

    min_height = min([dither.shape[0] for dither in dither_set])
    min_width = min([dither.shape[1] for dither in dither_set])

    # use reprojection to crop all dithers to the same size and WCS
    target_wcs = wcs_set[0]

    for i, dither in enumerate(dither_set):
        reprojected_dither, _ = reproject_image_to_target_wcs(
            dither,
            wcs_set[i],
            target_wcs,
            shape_out=(min_height, min_width),
            method=method,
        )

        dither_set[i] = reprojected_dither

        wcs_set[i] = target_wcs

    if logger:
        logger.info("Reprojection of dithered frames completed successfully.")

    return dither_set, wcs_set


def remove_nan_edges_dithers(dither_set, logger=None):
    """
    Estimates the NaN window for each dither, makes the most conservative window,
    applies to all dithers, and returns the cropped dither set and the cropping values
    to be used for the reference later on.
    """

    tops = []
    bottoms = []
    lefts = []
    rights = []

    if logger:
        logger.info("Estimating NaN windows for each dither...")

    for i, dither in enumerate(dither_set):
        all_edges, __ = estimate_nan_window(dither, logger=logger)

        top, bottom, left, right = all_edges

        tops.append(top)
        bottoms.append(bottom)
        lefts.append(left)
        rights.append(right)

    # find the max of all:

    top = np.max(tops)
    bottom = np.max(bottoms)
    left = np.max(lefts)
    right = np.max(rights)

    total_max = np.max(np.array([top, bottom, left, right]))

    if logger:
        logger.info(
            f"Cropping a total of {total_max} pixels from each edge to remove NaN edges from all dithers."
        )

    # crop all dithers to the same size
    for i, dither in enumerate(dither_set):
        dither_set[i] = dither[
            total_max : dither.shape[0] - total_max,
            total_max : dither.shape[1] - total_max,
        ]

    return dither_set, total_max


def transformation_matrix(tx, ty, rotation_deg, centre):
    """
    Convert the alignment result returned by AlignmentMethod.run()
    into a 3x3 homogeneous transformation matrix.

    The transformation is:

        p_target =
            centre + R(theta) @ (p_input - centre) + [tx, ty]

    for points p = [x, y] (x = column, y = row).

    Rotation convention (the same everywhere in the pipeline): rotation_deg is
    positive counter-clockwise as the frame is displayed with origin upper
    (the numpy array as read by astropy.io.fits), i.e. the convention of
    scipy.ndimage.rotate. In (x, y) with y = row this is a clockwise rotation,
    so theta = -rotation_deg.
    """

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

    # same convention as transformation_matrix: theta = -rotation_deg
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


def detector_errors_to_sky_errors(wcs, sigma_x, sigma_y, logger=None):
    """
    Convert 1-sigma errors of a detector/pixel translation (sigma_x, sigma_y)
    into 1-sigma errors of the RA/Dec offsets, in the same units as
    `detector_translation_to_sky_translation` (degrees of RA and Dec).

    Errors are not a displacement: pushing (sigma_x, sigma_y) through
    `detector_translation_to_sky_translation` gives a negative RA "error" when
    RA increases to the left (CD1_1 < 0), and lets the x and y errors cancel
    or add up when the WCS is rotated. Instead they are propagated through the
    local Jacobian J = d(RA, Dec) / d(x, y) of the WCS at the reference pixel
    (central differences of one pixel), assuming independent x and y errors:

        sigma_RA^2  = (dRA/dx  sigma_x)^2 + (dRA/dy  sigma_y)^2
        sigma_Dec^2 = (dDec/dx sigma_x)^2 + (dDec/dy sigma_y)^2

    Parameters
    ----------
    wcs : astropy.wcs.WCS
        WCS of the image (the same one used for the offsets).
    sigma_x, sigma_y : float
        1-sigma errors of the translation in pixels.

    Returns
    -------
    sigma_RA, sigma_Dec : float
        1-sigma errors of the RA and Dec offsets in degrees (always >= 0).
    """

    # WCS reference pixel, converted from FITS 1-based
    # coordinates to Astropy 0-based pixel coordinates.
    x0 = wcs.wcs.crpix[0] - 1
    y0 = wcs.wcs.crpix[1] - 1

    def sky(x, y):
        """Pixel (x, y) to (RA, Dec) in degrees."""
        ra, dec = wcs.pixel_to_world_values(x, y)
        return float(ra), float(dec)

    def difference(a, b):
        """Sky difference a - b in degrees, with the RA difference wrapped to [-180, 180]."""
        # RA difference wrapped to [-180, +180] degrees, Dec plain
        return (
            (a[0] - b[0] + 180.0) % 360.0 - 180.0,
            a[1] - b[1],
        )

    dRA_dx, dDec_dx = (v / 2.0 for v in difference(sky(x0 + 1, y0), sky(x0 - 1, y0)))
    dRA_dy, dDec_dy = (v / 2.0 for v in difference(sky(x0, y0 + 1), sky(x0, y0 - 1)))

    sigma_RA = float(np.hypot(dRA_dx * sigma_x, dRA_dy * sigma_y))
    sigma_Dec = float(np.hypot(dDec_dx * sigma_x, dDec_dy * sigma_y))

    if logger:
        logger.info(
            f"Converted detector errors (sigma_x={sigma_x}, sigma_y={sigma_y}) to "
            f"sky errors (sigma_RA={sigma_RA}, sigma_Dec={sigma_Dec})"
        )

    return sigma_RA, sigma_Dec


def sky_translation_to_detector_translation(wcs, dRA, dDec, logger=None):
    """
    Inverse of `detector_translation_to_sky_translation`: convert RA/Dec
    coordinate offsets back into a detector/pixel translation (dx, dy)
    using the image WCS.

    Parameters
    ----------
    wcs : astropy.wcs.WCS
        WCS of the image (the same one used for the forward conversion).
    dRA, dDec : float
        RA and Dec offsets in degrees.

    Returns
    -------
    dx, dy : float
        Translation in pixels.
    """

    # WCS reference pixel, converted from FITS 1-based
    # coordinates to Astropy 0-based pixel coordinates.
    x0 = wcs.wcs.crpix[0] - 1
    y0 = wcs.wcs.crpix[1] - 1

    # Sky coordinate at reference position (degrees)
    ra0, dec0 = wcs.pixel_to_world_values(x0, y0)

    # pixel position of the offset sky coordinate
    x1, y1 = wcs.world_to_pixel_values(ra0 + dRA, dec0 + dDec)

    dx = float(x1 - x0)
    dy = float(y1 - y0)

    if logger:
        logger.info(
            f"Converted sky offsets (dRA={dRA}, dDec={dDec}) to detector translation (dx={dx}, dy={dy})"
        )

    return dx, dy


def compare_distance(
    tx, ty, rot, con_tx, con_ty, con_rot, point_from_center=(100, 100)
):
    """
    For the point `point_from_center` in the input frame, compute the distance between the transformed point using (tx, ty, rot) and (con_tx, con_ty, con_rot).

    Rotates the point center, then translates it, and computes the distance between the two transformed points.
    """

    # Convert rotation from degrees to radians, same convention as
    # transformation_matrix (theta = -rotation)
    theta = np.radians(-rot)
    con_theta = np.radians(-con_rot)

    # Rotation matrices
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    con_R = np.array(
        [
            [np.cos(con_theta), -np.sin(con_theta)],
            [np.sin(con_theta), np.cos(con_theta)],
        ]
    )

    # Transform the point using the first transformation
    transformed_point = R @ point_from_center + np.array([tx, ty])

    # Transform the point using the second transformation
    con_transformed_point = con_R @ point_from_center + np.array([con_tx, con_ty])

    # Compute the distance between the two transformed points
    distance = np.linalg.norm(transformed_point - con_transformed_point)

    return distance
