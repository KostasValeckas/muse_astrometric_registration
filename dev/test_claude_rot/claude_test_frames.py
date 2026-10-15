"""
Rotation estimation for scale == 1, from DFT power spectra.
 
Replaces the log-polar / 2D phase_cross_correlation step.  Because scale is
known, the radial axis carries no shift, so we:
 
  * sample the spectrum on a *linear* polar grid (rings of constant |f|),
  * cross-correlate ring-by-ring along theta only and sum over rings,
  * read the angle off a 1D, trigonometrically-interpolated correlation.
 
Angle convention of the returned value: scipy.ndimage.rotate(reference, angle)
~= image   (i.e. `angle` is what you must apply to `reference` to get `image`).
"""

import numpy as np
from numpy.fft import fft2, fftshift, rfft, irfft
from scipy.ndimage import gaussian_filter, map_coordinates, rotate
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS


import numpy as np
from astropy.wcs import WCS
from reproject import reproject_interp, reproject_exact, reproject_adaptive
from astropy.wcs.utils import proj_plane_pixel_scales

import matplotlib.pyplot as plt

from pathlib import Path

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
        ratio = (
            proj_plane_pixel_scales(source_wcs)[0]
            / proj_plane_pixel_scales(target_wcs)[0]
        )
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
        target_wcs.wcs.crpix = (
            np.array(target_wcs.wcs.crpix) - 0.5
        ) * oversample_factor + 0.5
        shape_out = (shape_out[0] * oversample_factor, shape_out[1] * oversample_factor)

    target_wcs.array_shape = shape_out

    if logger:
        logger.info(f"Reprojecting to shape {shape_out} using '{method}'...")

    funcs = {
        "interpolation": reproject_interp,
        "exact": reproject_exact,
        "adaptive": reproject_adaptive,
    }
    if method not in funcs:
        if logger:
            logger.error(
                f"Invalid reprojection method: {method}. Returning source image."
            )
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


# --------------------------------------------------------------------------
# pre-processing
# --------------------------------------------------------------------------
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


def dog(img, sigma, ratio=1.6):
    """DoG with a fixed sigma *ratio*: sigma sets the band centre, ratio the
    (fractional) bandwidth.  Absolute sigma differences only scale amplitude."""
    return gaussian_filter(img, sigma, mode="nearest") - gaussian_filter(
        img, ratio * sigma, mode="nearest"
    )


def dog_passband(sigma, ratio=1.6, level=0.2):
    """Frequency range (cycles/pixel) where the DoG response >= level * peak."""
    f = np.linspace(1e-4, 0.5, 4000)
    h = np.exp(-2 * (np.pi * sigma * f) ** 2) - np.exp(
        -2 * (np.pi * ratio * sigma * f) ** 2
    )
    keep = f[h >= level * h.max()]
    return keep[0], keep[-1]


def prep(img):
    """Fill NaNs with the median (not 0) and remove the background level."""
    img = np.asarray(img, dtype=float)
    med = np.nanmedian(img)
    return np.where(np.isfinite(img), img, med) - med


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


def angular_shift(A, B, w, expected=0.0, halfwidth=None, upsample=64):
    """Shift m (deg) such that A(theta) ~ B(theta - m), plus a peak SNR.

    C(m) = sum_r w_r sum_t A_r[t+m] B_r[t]  computed with an FFT along theta
    only; zero-padding the angular spectrum gives *exact* band-limited
    interpolation (no parabola/centroid bias).
    """
    n = A.shape[1]
    cross = (w[:, None] * rfft(A, axis=1) * np.conj(rfft(B, axis=1))).sum(axis=0)
    if n % 2 == 0:
        cross[-1] = 0.0  # Nyquist term is ambiguous
    c = irfft(cross, n=n * upsample)
    ang = (np.arange(c.size) * (180.0 / c.size) + 90.0) % 180.0 - 90.0
    if halfwidth is not None:
        d = (ang - expected + 90.0) % 180.0 - 90.0
        cs = np.where(np.abs(d) <= halfwidth, c, -np.inf)
    else:
        cs = c
    k = int(np.argmax(cs))
    med = np.median(c)
    mad = 1.4826 * np.median(np.abs(c - med))
    return ang[k], (c[k] - med) / mad


# --------------------------------------------------------------------------
# driver (replaces the while-loop in run())
# --------------------------------------------------------------------------
def estimate_rotation_fixed_scale(
    reference,
    image,
    extra_rotation=0.0,
    sigmas=None,
    ratio=1.6,
    ctrl_tol=0.03,
    snr_min=30.0,
    search_halfwidth=1.0,
    min_ok=3,
    pad=2,
    upsample=64,
):
    """
    Returns dict(rot, rot_err, success, table).

    rot  : degrees, scipy convention: rotate(reference, rot) ~ image.
    table: per-sigma rows (sigma, a_main, a_ctrl, eps, snr_main, snr_ctrl, ok).

    `extra_rotation` should be larger than search_halfwidth + peak width so the
    true peak is separated from frame-fixed features (sensor striping, pixel-grid
    anisotropy) that all pile up at dtheta = 0.
    """
    ref, img = prep(reference), prep(image)
    L = max(ref.shape)
    win = radial_window(ref.shape)

    # control image: the *same* data rotated by a known angle (spline, no NaNs
    # inside the inscribed disk)
    img_rot = rotate(
        img, extra_rotation, reshape=False, order=3, mode="constant", cval=0.0
    )

    if sigmas is None:  # geometric: band centre ~ 1/sigma
        sigmas = np.geomspace(1.0, 16.0, 32)

    # my polar parameterisation rotates opposite to scipy's, hence the minus
    S = -1.0
    rows, ok_eps = [], []
    for s in sigmas:
        f_lo, f_hi = dog_passband(s, ratio)
        n_theta = max(128, 2 * int(2 * np.pi * f_hi * L))  # even
        rings = {}
        for name, im in (("ref", ref), ("img", img), ("rot", img_rot)):
            rings[name], f = polar_rings(
                dog(im, s, ratio) * win, f_lo, f_hi, n_theta, pad
            )
        w = f / f.sum()  # ring information ~ |f|

        m_main, snr_main = angular_shift(
            rings["rot"],
            rings["ref"],
            w,
            S * extra_rotation,
            search_halfwidth,
            upsample,
        )
        m_ctrl, snr_ctrl = angular_shift(
            rings["rot"],
            rings["img"],
            w,
            S * extra_rotation,
            search_halfwidth,
            upsample,
        )
        a_main, a_ctrl = S * m_main, S * m_ctrl
        eps = a_main - a_ctrl  # bias-cancelled
        ok = (
            abs(a_ctrl - extra_rotation) < ctrl_tol
            and snr_main > snr_min
            and snr_ctrl > snr_min
        )
        rows.append((s, a_main, a_ctrl, eps, snr_main, snr_ctrl, ok))
        if ok:
            ok_eps.append(eps)

    table = np.array(rows, dtype=float)
    if len(ok_eps) >= min_ok:
        rot = float(np.median(ok_eps))
        err = float(np.std(ok_eps, ddof=1))  # conservative: sigmas correlated
        return dict(rot=rot, rot_err=err, success=True, table=table)
    return dict(rot=np.nan, rot_err=np.nan, success=False, table=table)


# --------------------------------------------------------------------------
# synthetic self-test:  python fixed_scale_rotation.py
# --------------------------------------------------------------------------
def _render(xy, flux, rot_deg, shift, N, psf, rng, noise=4.0):
    """Analytic star field; rot_deg in scipy's convention (about (N-1)/2)."""
    c = (N - 1) / 2
    phi = -np.deg2rad(rot_deg)  # scipy +angle == my -phi (checked numerically)
    x, y = xy[:, 0] - c, xy[:, 1] - c
    xr = np.cos(phi) * x - np.sin(phi) * y + c + shift[0]
    yr = np.sin(phi) * x + np.cos(phi) * y + c + shift[1]
    img = np.full((N, N), 50.0)
    for xi, yi, fl in zip(xr, yr, flux):
        x0, x1 = int(max(0, xi - 9)), int(min(N, xi + 10))
        y0, y1 = int(max(0, yi - 9)), int(min(N, yi + 10))
        if x1 <= x0 or y1 <= y0:
            continue
        gx = np.arange(x0, x1)[None, :]
        gy = np.arange(y0, y1)[:, None]
        img[y0:y1, x0:x1] += (
            fl
            / (2 * np.pi * psf**2)
            * np.exp(-((gx - xi) ** 2 + (gy - yi) ** 2) / (2 * psf**2))
        )
    return img + rng.normal(0, noise, img.shape)


ref = "/home/kostas/Documents/ESO/muse_astrometric_registration/dev/test_claude_rot/reference"
input_dir = "/home/kostas/Documents/ESO/For_Kostas/PHANGS_LP/MUSE"


if __name__ == "__main__":

    ref_hdul = fits.open(ref)

    fits_in_input_dir = list(Path(input_dir).glob("*.fits"))

    # sort:
    fits_in_input_dir.sort(key=lambda x: x.name)

    all_rots = []

    for i, im2 in enumerate(fits_in_input_dir):
        print("Processing: ", im2)
        img_hdul = fits.open(im2)

        img_wcs = WCS(img_hdul[1].header)

        ref = ref_hdul[0].data
        ref_wcs = WCS(ref_hdul[0].header)

        # reproject reference onto image WCS
        ref, _ = reproject_image_to_target_wcs(
            ref, ref_wcs, img_wcs, shape_out=img_hdul[1].data.shape
        )

        img = img_hdul[1].data

        out = estimate_rotation_fixed_scale(ref, img, extra_rotation=6.0)

        n_ok = int(out["table"][:, -1].sum())

        snr_main = out["table"][:, 4]
        snr_control = out["table"][:, 5]
        rot_main = out["table"][:, 3]
        plt.plot(snr_main, "-o", label="SNR Main")
        plt.plot(snr_control, "-o", label="SNR Control")
        plt.xlabel("Index")
        plt.ylabel("SNR")
        plt.title("SNR Main vs Index")
        plt.axhline(30.0, color="r", linestyle="--", label="SNR Threshold")
        plt.legend(loc="upper left")
        # new y axis for rotation
        ax2 = plt.gca().twinx()
        ax2.plot(rot_main, "-o", color="orange", label="Rotation")
        ax2.set_ylabel("Rotation (degrees)")
        ax2.legend(loc="upper right")
        plt.savefig(f"snr_rotation_plot_{fits_in_input_dir[i].name}.png")
        plt.close()

        print(
            "\nper-sigma table for last case: sigma, a_main, a_ctrl, eps, snr_main, snr_ctrl, ok"
        )
        np.set_printoptions(precision=4, suppress=True)
        print(out["table"])

        print(out["rot"])

        all_rots.append(out["rot"])

    plt.plot(all_rots, "-o", label="Estimated Rotations")

    # write all rots to disc as an npy array

    all_rots = np.array(all_rots)

    np.save("all_rotations.npy", all_rots)
