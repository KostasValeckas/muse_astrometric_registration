"""
Flux scaling of the reference image to the MUSE images (``--scale-ref``).

Borrowed from pymusepipe (https://github.com/emsellem/pymusepipe, MIT
license, Copyright (c) 2019 Eric Emsellem), so that the FLUX_SCALE and
BACKGROUND written to the output table have the same meaning as pymusepipe's:

1. Unit conversion (``align_pipe.get_conversion_factor`` with
   ``config_pipe.dict_equivalencies``): the reference is converted from its
   unit to the MUSE unit (1e-20 erg/s/cm2/A) with the spectral density
   equivalency at the pivot wavelength of its filter. The unit is read from
   the BUNIT keyword of the reference header; only if there is none, the unit
   of the filter in ``REFERENCE_FILTERS`` is used (pymusepipe's default).

2. Pixel area (``align_pipe._align_reference_hdu``): after the reprojection to
   the MUSE frame, the reference is multiplied by the ratio of the pixel areas
   (MUSE / reference).

3. Normalisation (``util_image.get_normfactor`` / ``get_polynorm`` /
   ``chunk_stats`` / ``regress_odr`` with the defaults of
   ``AlignMuseDataset.get_normfactor_ima``): both images are median filtered
   (NaN set to 0 first), cut into chunks of 10 x 10 pixels, and the chunk
   medians are fitted with ``reference = FLUX_SCALE * (MUSE + BACKGROUND)``,
   with the chunk MAD standard deviations as errors on both axes (orthogonal
   distance regression).

pymusepipe uses ``scipy.odr`` for the regression, which is removed from SciPy
1.19. Here the same orthogonal distance problem is solved with
``scipy.optimize.least_squares``: for a straight line, minimising the
orthogonal distances over the x corrections gives in closed form the
residuals ``(y - b1 (x + b0)) / sqrt(sy^2 + b1^2 sx^2)``, so the solution is
the same as ODRPACK's (to the precision of the fit, not bit for bit).

In the pipeline, only the unit conversion and the pixel area are applied
before the alignment; the normalisation is fitted after it, with the final
offsets, and ``reference / FLUX_SCALE - BACKGROUND`` (the reference on the flux
scale of the MUSE image) is only used for the final difference images.
"""

import warnings

import numpy as np
import astropy.units as u
from astropy.stats import mad_std
from astropy.wcs.utils import proj_plane_pixel_scales
from scipy.ndimage import median_filter, rotate, shift
from scipy.optimize import least_squares


# unit of the MUSE images (pymusepipe ``config_pipe.default_muse_unit``)
MUSE_UNIT = 1.0e-20 * u.erg / (u.cm * u.cm * u.second * u.AA)

# unit (in microJansky) and pivot wavelength (Angstrom) of the reference image
# per filter (pymusepipe ``config_pipe.dict_equivalencies``). The pivot
# wavelength is always used; the unit only if the reference header has no
# BUNIT
REFERENCE_FILTERS = {
    "WFI_BB": (1.000, 6483.58),
    "DUPONT_R": (1.000, 6483.58),
    "Johnson_B": (1.000, 4402.1095),
    "Johnson_V": (1.000, 5492.9237),
    "Cousins_R": (1.000, 6449.8565),
    "Cousins_I": (1.000, 8046.6135),
    "SDSS_u": (3.631, 3556.5240),
    "SDSS_g": (3.631, 4702.4953),
    "SDSS_r": (3.631, 6175.5789),
    "SDSS_i": (3.631, 7489.9768),
    "SDSS_z": (3.631, 8946.7096),
    "ACS_F475W": (3.631, 4715.0024),
    "ACS_F550M": (3.631, 5578.5129),
    "ACS_F555W": (3.631, 5344.3102),
    "ACS_F606W": (3.631, 5849.2354),
    "ACS_F625W": (3.631, 6279.7071),
    "ACS_F775W": (3.631, 7679.5302),
    "ACS_F814W": (3.631, 8036.7563),
    "WFPC2_F555W": (3.631, 5391.3705),
    "WFPC2_F675W": (3.631, 6697.8244),
    "WFPC2_F814W": (3.631, 7947.5087),
    "WFC3_F502N": (3.631, 5009.0078),
    "WFC3_F555W": (3.631, 5309.9095),
    "WFC3_F606W": (3.631, 5907.0092),
    "WFC3_F625W": (3.631, 6253.9998),
    "WFC3_F656N": (3.631, 6561.1205),
    "WFC3_F775W": (3.631, 7732.9840),
    "WFC3_F814W": (3.631, 8304.6816),
    "Kron_V": (3.631, 5416.2200),
    "LEGACY_G": (3.631, 4808.5075),
    "LEGACY_R": (3.631, 6417.6456),
    "LEGACY_I": (3.631, 7814.5752),
    "LEGACY_Z": (3.631, 9168.8545),
    "PANSTARRS_O": (3.631, 6550.0000),
    "PANSTARRS_G": (3.631, 4810.0000),
    "PANSTARRS_R": (3.631, 6170.0000),
    "PANSTARRS_I": (3.631, 7520.0000),
    "PANSTARRS_Z": (3.631, 8660.0000),
    "PANSTARRS_Y": (3.631, 9620.0000),
    "PANSTARRS_W": (3.631, 6080.0000),
}

# pymusepipe's default filter
DEFAULT_REFERENCE_FILTER = "WFI_BB"

# filters pymusepipe warns about (normalisation may need attention)
FILTER_WARNING_LIST = ["PANSTARRS"]

# parameters of the normalisation fit (pymusepipe defaults)
NORM_MEDIAN_FILTER_SIZE = 2  # median filter of both images (pixels)
NORM_CHUNK_SIZE = 10  # chunk size (pixels)
NORM_VALID_FRACTION = 0.3  # minimum fraction of finite pixels in a chunk
NORM_THRESHOLD = 0.0  # chunks with a median at or below this are not used

# not in pymusepipe: fewer chunks than this is not a usable fit
NORM_MIN_CHUNKS = 10

# not in pymusepipe: a FLUX_SCALE further than this factor from 1 means the
# reference unit is most likely wrong (e.g. BUNIT = 'mJy' on an image in
# microJansky gives FLUX_SCALE ~ 1000). Before the alignment this stops the run
# (``check_flux_scale_before_alignment``), after it the median of the run is
# checked again (``check_flux_scales``)
FLUX_SCALE_WARNING_FACTOR = 10.0


class FluxScaleError(RuntimeError):
    """
    The reference and the exposures are on very different flux scales after
    the unit conversion (wrong reference unit), see
    ``check_flux_scale_before_alignment``. Ends the run.
    """


def _parse_unit(text, source):
    """
    Parse a unit string (``BUNIT`` or ``--ref_unit``), see ``reference_unit``.

    Returns
    -------
    unit : astropy.units.Unit
        The unit, with an explicit "per pixel" dropped.

    per_pixel : bool
        Whether a "per pixel" was dropped.
    """

    try:
        with warnings.catch_warnings():
            # e.g. "multiple slashes" in erg/s/cm2/Angstrom
            warnings.simplefilter("ignore", u.UnitsWarning)
            unit = u.Unit(str(text).strip(), parse_strict="raise")
    except ValueError as error:
        raise ValueError(
            f"Cannot parse the unit of the reference, {source} = {text!r}: {error}"
        ) from None

    # the MUSE unit is a flux per pixel without saying so: an explicit "per
    # pixel" (e.g. uJy/pix) is the same thing
    pix_power = dict(zip(unit.bases, unit.powers)).get(u.pix, 0)
    if pix_power == -1:
        unit = unit * u.pix

    return unit, pix_power == -1


def reference_unit(
    header=None, filter_name=DEFAULT_REFERENCE_FILTER, logger=None, unit=None
):
    """
    Unit of the reference image: `unit` if given (``--ref_unit``), else the
    BUNIT keyword of its header, or, if there is none, the unit of its filter
    in ``REFERENCE_FILTERS``.

    `unit` is for references with a wrong BUNIT: e.g. the WFI images with
    BUNIT = 'mJy' whose data are in microJansky (FLUX_SCALE ~ 1000 in the
    output table).

    Parameters
    ----------
    header : astropy.io.fits.Header, optional
        Header of the reference image.

    filter_name : str, optional
        Filter of the reference image, one of ``REFERENCE_FILTERS``.

    unit : str, optional
        Unit of the reference, used instead of its BUNIT (e.g. "uJy").

    An explicit "per pixel" (e.g. ``uJy/pix``) is dropped: the MUSE unit is a
    flux per pixel as well, without saying so.

    Returns
    -------
    astropy.units.Quantity
        The unit, as a quantity (e.g. 1 uJy, or 3.631 uJy for a filter in
        nanomaggies).

    Raises
    ------
    ValueError
        If BUNIT (or `unit`) cannot be parsed.
    """

    bunit = None if header is None else header.get("BUNIT")

    if unit is not None:
        parsed, per_pixel = _parse_unit(unit, "--ref_unit")
        if logger:
            logger.warning(
                f"Reference unit set by hand: {unit!r} ({parsed}"
                + (" per pixel" if per_pixel else "")
                + f"), the BUNIT of the reference ({bunit!r}) is not used."
            )
        return 1.0 * parsed

    if bunit is None or not str(bunit).strip():
        value, _ = REFERENCE_FILTERS[filter_name]
        if logger:
            logger.warning(
                f"The reference has no BUNIT keyword: assuming the unit of filter "
                f"{filter_name}, {value} microJansky (pymusepipe's default)."
            )
        return value * u.microJansky

    parsed, per_pixel = _parse_unit(bunit, "BUNIT")

    if logger:
        logger.info(
            f"Reference unit from its header: BUNIT = {bunit!r} ({parsed}"
            + (" per pixel" if per_pixel else "")
            + ")"
        )

    return 1.0 * parsed


def reference_conversion_factor(
    filter_name=DEFAULT_REFERENCE_FILTER, header=None, logger=None, unit=None
):
    """
    Factor that converts the reference image from its unit (see
    ``reference_unit``) to the MUSE unit (pymusepipe ``get_conversion_factor``),
    with the spectral density equivalency at the pivot wavelength of its
    filter.

    Parameters
    ----------
    filter_name : str, optional
        Filter of the reference image, one of ``REFERENCE_FILTERS``. Default is
        ``DEFAULT_REFERENCE_FILTER``.

    header : astropy.io.fits.Header, optional
        Header of the reference image (for BUNIT).

    unit : str, optional
        Unit of the reference, used instead of its BUNIT (``--ref_unit``).

    Returns
    -------
    float
        The conversion factor.

    Raises
    ------
    ValueError
        If the filter is unknown, or the unit of the reference cannot be
        converted to the MUSE unit (pymusepipe would silently use 1.0).
    """

    if filter_name not in REFERENCE_FILTERS:
        raise ValueError(
            f"Unknown reference filter {filter_name!r}. Choose from "
            f"{', '.join(REFERENCE_FILTERS)}."
        )

    _, pivot = REFERENCE_FILTERS[filter_name]
    unit = reference_unit(header, filter_name=filter_name, logger=logger, unit=unit)

    try:
        factor = float(
            unit.to(MUSE_UNIT.unit, equivalencies=u.spectral_density(pivot * u.AA)).value
            / MUSE_UNIT.value
        )
    except u.UnitConversionError:
        raise ValueError(
            f"The unit of the reference ({unit.unit}) cannot be converted to the MUSE "
            f"unit (1e-20 erg/s/cm2/Angstrom per pixel). Note that 'A' in a unit is "
            f"Ampere (Angstrom is 'Angstrom'), and that a surface brightness (per "
            f"arcsec2) is not a flux per pixel."
        ) from None

    if logger:
        logger.info(
            f"Reference unit conversion ({unit} at the pivot wavelength {pivot} A of "
            f"filter {filter_name} to 1e-20 erg/s/cm2/A): factor {factor:.6g}"
        )
        if any(name in filter_name for name in FILTER_WARNING_LIST):
            logger.warning(
                f"pymusepipe flags filter {filter_name}: its normalisation may need "
                f"attention."
            )

    return factor


def pixel_area_factor(target_wcs, reference_wcs):
    """
    Ratio of the pixel areas target / reference, to keep the flux per pixel
    after reprojecting the reference to the target frame (pymusepipe
    ``_align_reference_hdu``).
    """

    target = proj_plane_pixel_scales(target_wcs.celestial)
    reference = proj_plane_pixel_scales(reference_wcs.celestial)

    return float((target[0] / reference[0]) * (target[1] / reference[1]))


def _prepare_image(image):
    """Median filter an image, with NaN set to 0 first (pymusepipe ``prepare_image``)."""
    return median_filter(np.nan_to_num(image), NORM_MEDIAN_FILTER_SIZE)


def chunk_stats(arrays, chunk_size=NORM_CHUNK_SIZE, valid_fraction=NORM_VALID_FRACTION):
    """
    Median and MAD standard deviation of every chunk of `chunk_size` x
    `chunk_size` pixels of the arrays (pymusepipe ``chunk_stats``).

    Parameters
    ----------
    arrays : list of np.2darray
        Arrays of the same shape.

    Returns
    -------
    median, std : np.ndarray
        Shape (number of arrays, number of chunks). Chunks with less than
        `valid_fraction` finite pixels are 0.
    """

    stack = np.asarray(arrays, dtype=np.float64)
    n_arrays, ny, nx = stack.shape
    n_y, n_x = ny // chunk_size, nx // chunk_size

    chunks = stack[:, : n_y * chunk_size, : n_x * chunk_size].reshape(
        n_arrays, n_y, chunk_size, n_x, chunk_size
    )
    chunks = chunks.transpose(0, 1, 3, 2, 4).reshape(
        n_arrays, n_y * n_x, chunk_size * chunk_size
    )

    valid = np.mean(np.isfinite(chunks), axis=-1) > valid_fraction

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(chunks, axis=-1)
        std = mad_std(chunks, axis=-1, ignore_nan=True)

    median[~valid] = np.nan
    std[~valid] = np.nan

    return np.nan_to_num(median), np.nan_to_num(std)


def _odr_linear(x, y, sx, sy, beta0=(0.0, 1.0)):
    """
    Orthogonal distance regression of ``y = b1 (x + b0)`` with errors `sx`
    and `sy` (see the module docstring). As in pymusepipe, x is measured from
    its minimum during the fit.

    Returns (b0, b1).
    """

    minx = np.min(x)
    xs = x - minx

    def residuals(beta):
        """Residuals of the orthogonal distance problem for a straight line."""
        b0, b1 = beta
        return (y - b1 * (xs + b0)) / np.sqrt(sy**2 + (b1 * sx) ** 2)

    result = least_squares(residuals, np.asarray(beta0, dtype=float), method="lm")

    b0, b1 = result.x
    return float(b0 - minx), float(b1)


def fit_flux_normalisation(muse_image, reference_image, logger=None, name=""):
    """
    Fit ``reference = FLUX_SCALE * (MUSE + BACKGROUND)`` (pymusepipe
    ``get_normfactor`` with the defaults of ``get_normfactor_ima``).

    Parameters
    ----------
    muse_image, reference_image : np.2darray
        The MUSE image and the reference (unit converted, reprojected to the
        same frame, same shape).

    name : str, optional
        Name of the exposure for the log.

    Returns
    -------
    flux_scale, background : float
        NaN if the fit is not usable (too few chunks, or a scale <= 0).
    """

    muse = _prepare_image(muse_image)
    reference = _prepare_image(reference_image)

    median, std = chunk_stats([muse, reference])

    use = (
        (median[0] > NORM_THRESHOLD)
        & (std[0] > 0.0)
        & (std[1] > 0.0)
        & (median[1] > 0.0)
    )
    n_chunks = int(np.sum(use))

    if n_chunks < NORM_MIN_CHUNKS:
        if logger:
            logger.warning(
                f"Flux normalisation {name}: only {n_chunks} usable chunks (minimum "
                f"{NORM_MIN_CHUNKS}), no fit."
            )
        return np.nan, np.nan

    background, flux_scale = _odr_linear(
        median[0][use], median[1][use], std[0][use], std[1][use]
    )

    if not (np.isfinite(flux_scale) and flux_scale > 0 and np.isfinite(background)):
        if logger:
            logger.warning(
                f"Flux normalisation {name}: fit not usable (FLUX_SCALE {flux_scale}, "
                f"BACKGROUND {background})."
            )
        return np.nan, np.nan

    if logger:
        logger.info(
            f"Flux normalisation {name}: FLUX_SCALE {flux_scale:.6g}, BACKGROUND "
            f"{background:.6g} ({n_chunks} chunks)"
        )

    return flux_scale, background


def shift_image_with_nans(image, dx, dy, dtheta):
    """
    Rotate (about the centre) and then shift an image as
    ``AligmentMethod.shift_image`` does, but with NaN where the result has no
    data (NaN pixels and outside the frame) instead of 0 / the nearest edge.
    Used to refit the normalisation with the final offsets.
    """

    finite = np.isfinite(image)

    data = shift(
        rotate(np.where(finite, image, 0.0), angle=dtheta, reshape=False, mode="nearest"),
        shift=(dy, dx),
        mode="nearest",
    )
    covered = shift(
        rotate(finite.astype(float), angle=dtheta, reshape=False, order=1, cval=0.0),
        shift=(dy, dx),
        order=1,
        cval=0.0,
    )

    return np.where(covered > 0.999, data, np.nan)


def check_flux_scales(flux_scales, logger=None, unit_source="its BUNIT"):
    """
    Warn if the median FLUX_SCALE of a run is further than
    ``FLUX_SCALE_WARNING_FACTOR`` from 1. The exposures and the reference are
    then on very different scales after the unit conversion, which almost
    always means the unit of the reference is wrong (e.g. BUNIT = 'mJy' on an
    image in microJansky gives FLUX_SCALE ~ 1000). FLUX_SCALE is then off by
    this factor (BACKGROUND, in the unit of the exposure, is not), and
    PointSources can fail: its bright-source filter
    (``filters.remove_bright_sources_simple``) cuts at an absolute value, 500,
    which removes the whole reference if it is ~1000 times too bright.
    OpticalFlow and FFT are insensitive to a constant factor.

    Parameters
    ----------
    flux_scales : array_like
        FLUX_SCALE of the exposures of the run (NaN without a result).

    unit_source : str, optional
        Where the unit of the reference came from, for the message.

    Returns
    -------
    float
        The median FLUX_SCALE (NaN if there is none).
    """

    flux_scales = np.asarray(flux_scales, dtype=float)
    flux_scales = flux_scales[np.isfinite(flux_scales)]

    if flux_scales.size == 0:
        return np.nan

    median = float(np.median(flux_scales))

    if logger and not (
        1.0 / FLUX_SCALE_WARNING_FACTOR <= median <= FLUX_SCALE_WARNING_FACTOR
    ):
        logger.warning(
            f"The median FLUX_SCALE of the run is {median:.4g} (from "
            f"{flux_scales.size} exposures), more than a factor "
            f"{FLUX_SCALE_WARNING_FACTOR:g} from 1: the unit of the reference "
            f"(from {unit_source}) is most likely wrong. FLUX_SCALE is off by "
            f"this factor (BACKGROUND is not), and PointSources may have failed "
            f"(its bright-source filter cuts at an absolute value; OpticalFlow and "
            f"FFT are not affected). Set the correct unit with --ref_unit (e.g. a "
            f"factor ~1000 with BUNIT = 'mJy' usually means --ref_unit uJy)."
        )

    return median


def unit_prefix_correction(flux_scale, factor=FLUX_SCALE_WARNING_FACTOR):
    """
    Factor the reference is off by if `flux_scale` is off by a power of 1000,
    the signature of a wrong SI prefix in its unit (e.g. BUNIT = 'mJy' on data
    in microJansky gives FLUX_SCALE ~ 1000, i.e. 1e-3). Only used for the
    error message of ``check_flux_scale_before_alignment``: a wrong unit is
    never corrected automatically.

    Returns
    -------
    float or None
        1000**-k for the k that brings `flux_scale` within `factor` of 1 (1.0
        if it already is), or None if no power of 1000 does (or `flux_scale` is
        not a positive number).
    """

    if not np.isfinite(flux_scale) or flux_scale <= 0:
        return None

    k = int(np.round(np.log10(flux_scale) / 3.0))
    corrected = flux_scale / 1000.0**k

    if 1.0 / factor <= corrected <= factor:
        return 1000.0 ** (-k)

    return None


# the known header inconsistency of some WFI references (``--scale-ref``):
# BUNIT = 'mJy' while the data are in microJansky, as the ZUNIT keyword of the
# same header says (e.g. NGC1365_Rc_flux_nosky.fits: pixel values 1.03 times
# those of the reference with BUNIT = 'uJy/pixel'). See ``wfi_unit_fix``.
WFI_MJY_BUNIT = u.mJy
# ZUNIT values meaning microJansky (compared in lower case, without spaces)
WFI_ZUNIT_MICROJANSKY = ("microjansky", "ujy", "microjy")
WFI_MJY_FIX = 1e-3  # mJy -> uJy: the reference is multiplied by this


def wfi_unit_fix(header):
    """
    Check if the header of the reference is the known WFI case: BUNIT = 'mJy'
    (with or without "per pixel") and ZUNIT = 'micro Jansky'. The header
    contradicts itself, and the data are in microJansky.

    Parameters
    ----------
    header : astropy.io.fits.Header or None
        Header of the reference.

    Returns
    -------
    factor : float or None
        ``WFI_MJY_FIX`` (to multiply the reference with) if it is this case,
        else None.

    reason : str
        Why it is or is not this case, for the log.
    """

    if header is None:
        return None, "no header of the reference"

    bunit = header.get("BUNIT")
    zunit = header.get("ZUNIT")

    if bunit is None:
        return None, "the header has no BUNIT"

    try:
        unit, _ = _parse_unit(bunit, "BUNIT")
    except ValueError:
        return None, f"BUNIT = {bunit!r} cannot be parsed"

    if unit != WFI_MJY_BUNIT:
        return None, f"BUNIT = {bunit!r} is not mJy"

    if zunit is None:
        return None, "BUNIT is mJy but the header has no ZUNIT to confirm microJansky"

    if "".join(str(zunit).lower().split()) not in WFI_ZUNIT_MICROJANSKY:
        return None, f"BUNIT is mJy but ZUNIT = {zunit!r} is not microJansky"

    return WFI_MJY_FIX, f"BUNIT = {bunit!r} but ZUNIT = {zunit!r}"


def _plausible_flux_scale(flux_scale):
    """Whether `flux_scale` is within ``FLUX_SCALE_WARNING_FACTOR`` of 1."""
    return bool(
        np.isfinite(flux_scale)
        and 1.0 / FLUX_SCALE_WARNING_FACTOR <= flux_scale <= FLUX_SCALE_WARNING_FACTOR
    )


def _flux_scale_report(
    name, muse_image, reference_image, flux_scale, background, header, details
):
    """Lines describing a failed flux scale check, for the log and the error."""

    def _median(image):
        values = np.asarray(image, dtype=float)
        values = values[np.isfinite(values) & (values != 0)]
        return (float(np.median(values)), values.size) if values.size else (np.nan, 0)

    muse_median, muse_n = _median(muse_image)
    reference_median, reference_n = _median(reference_image)

    lines = [f"Flux scale check before the alignment {name} FAILED:"]

    for key, value in (details or {}).items():
        lines.append(f"    {key}: {value}")

    if header is not None:
        for keyword in ("BUNIT", "ZUNIT", "ZPOINT", "FILTER", "ORIGIN", "DATE"):
            lines.append(f"    header {keyword}: {header.get(keyword)!r}")

    lines += [
        f"    median of the exposure (MUSE unit): {muse_median:.6g} ({muse_n} pixels)",
        f"    median of the reference (converted, reprojected): {reference_median:.6g} "
        f"({reference_n} pixels)",
        f"    ratio of the medians reference / exposure: "
        f"{reference_median / muse_median if muse_median else np.nan:.6g}",
        f"    fitted FLUX_SCALE: {flux_scale:.6g}, BACKGROUND: {background:.6g} "
        f"(reference = FLUX_SCALE * (exposure + BACKGROUND), plausible within a "
        f"factor {FLUX_SCALE_WARNING_FACTOR:g} of 1)",
    ]

    prefix = unit_prefix_correction(flux_scale)
    if prefix is not None and prefix != 1.0:
        lines.append(
            f"    the scale is off by a factor {1.0 / prefix:g}, a power of 1000 (a "
            f"wrong SI prefix of the unit?)"
        )

    return lines


def check_flux_scale_before_alignment(
    muse_image,
    reference_image,
    logger=None,
    name="",
    unit_source="its BUNIT",
    header=None,
    allow_fix=True,
    details=None,
):
    """
    Check the unit conversion of the reference before the alignment: fit
    ``reference = FLUX_SCALE * (MUSE + BACKGROUND)`` with the exposure as
    reprojected with its WCS (not aligned yet). The fit uses the medians of
    10 x 10 pixel chunks of the median filtered images, so the remaining WCS
    offset (a few pixels) changes it little, while a wrong unit (a factor 1000
    for mJy / uJy) is obvious.

    1. The unit from the header (or ``--ref_unit``) is used if FLUX_SCALE is
       plausible (within ``FLUX_SCALE_WARNING_FACTOR`` of 1).
    2. If not, and `allow_fix`: the known WFI case (``wfi_unit_fix``: BUNIT =
       'mJy' but ZUNIT = 'micro Jansky') is fixed by converting the reference
       as microJansky, if the fit then gives a plausible FLUX_SCALE.
    3. Otherwise everything known about the conversion is logged as an error
       and ``FluxScaleError`` is raised: the run must not go on.

    Parameters
    ----------
    muse_image, reference_image : np.2darray
        The exposure and the reference (unit converted, reprojected to the
        same frame, same shape).

    name : str, optional
        Name of the exposure for the log.

    unit_source : str, optional
        Where the unit of the reference came from, for the message.

    header : astropy.io.fits.Header, optional
        Header of the reference, for the fix and the error report.

    allow_fix : bool, optional
        Try the fix of step 2 (default True). False with ``--ref_unit`` (an
        explicit unit is not changed) and once a run has decided on its unit.

    details : dict, optional
        Further information for the error report (file, filter, conversion
        and area factors).

    Returns
    -------
    flux_scale : float
        The FLUX_SCALE of the check, after the fix if one was applied (NaN if
        the fit is not usable: then nothing is checked).

    fix : float or None
        Factor to multiply the reference with: 1.0 if the unit is right,
        ``WFI_MJY_FIX`` if the fix was applied, None if not checked.

    Raises
    ------
    FluxScaleError
        If FLUX_SCALE is not plausible and the fix does not apply or does not
        help.
    """

    flux_scale, background = fit_flux_normalisation(
        muse_image, reference_image, name=name
    )

    if not np.isfinite(flux_scale):
        if logger:
            logger.warning(
                f"Flux scale check before the alignment {name}: the fit is not "
                f"usable, not checked."
            )
        return flux_scale, None

    # 1) the unit of the header
    if _plausible_flux_scale(flux_scale):
        if logger:
            logger.info(
                f"Flux scale check before the alignment {name}: FLUX_SCALE = "
                f"{flux_scale:.4g} with the unit from {unit_source}, the unit "
                f"conversion of the reference is plausible."
            )
        return flux_scale, 1.0

    report = _flux_scale_report(
        name, muse_image, reference_image, flux_scale, background, header, details
    )
    report.insert(1, f"    unit of the reference from {unit_source}")

    # 2) the known WFI case
    if not allow_fix:
        report.append(
            "    no fix tried (the unit was set with --ref_unit, or the unit of the "
            "run was already decided at an earlier pointing)"
        )
    else:
        fix, reason = wfi_unit_fix(header)

        if fix is None:
            report.append(f"    the known WFI fix does not apply: {reason}")
        else:
            fixed_scale, fixed_background = fit_flux_normalisation(
                muse_image, reference_image * fix, name=name
            )

            if _plausible_flux_scale(fixed_scale):
                if logger:
                    logger.warning(
                        f"Flux scale check before the alignment {name}: FLUX_SCALE "
                        f"= {flux_scale:.4g} with the unit from {unit_source}, not "
                        f"plausible. The header is the known WFI case ({reason}): "
                        f"the data are in microJansky. The reference is converted "
                        f"as microJansky (multiplied by {fix:g}): FLUX_SCALE = "
                        f"{fixed_scale:.4g}, BACKGROUND = {fixed_background:.4g}."
                    )
                return fixed_scale, fix

            report.append(
                f"    the known WFI fix applies ({reason}) but does not help: with "
                f"the reference multiplied by {fix:g}, FLUX_SCALE = "
                f"{fixed_scale:.6g}, BACKGROUND = {fixed_background:.6g}"
            )

    # 3) stop
    report.append(
        "    Stopping the run: with a wrong reference unit FLUX_SCALE is wrong and "
        "PointSources fails. Check the unit of the reference (BUNIT) or set it "
        "with --ref_unit."
    )
    message = "\n".join(report)

    if logger:
        logger.error(message)

    raise FluxScaleError(message)
