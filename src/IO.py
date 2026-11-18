"""Reading of FITS images and their WCS, and writing of the alignment output tables."""

import contextlib
import os

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
from astropy.table import Table


def read_fits_image(file_path, logger=None, data_extension=0):
    """
    Wrapper for astropy.io.fits.open.

    Parameters
    ----------
    file_path : str or pathlib.Path
        The path to the FITS file.

    data_extension : int, optional
        The extension of the FITS file to read. Default is 0 (primary HDU).

    Returns
    -------
    np.ndarray
        The image data as a 2D numpy array.
    """

    if logger:
        logger.info(f"Reading FITS image from {file_path} (extension {data_extension})")

    try:
        with fits.open(file_path) as hdul:
            image_data = hdul[data_extension].data

    except Exception as e:
        if logger:
            logger.error(f"Error reading FITS image from {file_path}: {e}")

        return None

    if image_data is None:
        if logger:
            logger.error(
                f"No data found in FITS image at {file_path} (extension {data_extension})"
            )
        return None

    if logger:
        logger.info(
            f"Successfully read FITS image from {file_path} (extension {data_extension})"
        )

    return image_data, hdul


def extract_image_wcs(file_path, logger=None, wcs_extension=0):
    """
    Extracts the WCS information from a FITS file.

    Parameters
    ----------
    file_path : str or pathlib.Path
        The path to the FITS file.

    wcs_extension : int, optional
        The extension of the FITS file to extract WCS from. Default is 0 (primary HDU).

    Returns
    -------
    astropy.wcs.WCS
        The WCS object containing the World Coordinate System information.
    """

    if logger:
        logger.info(
            f"Extracting WCS from FITS image at {file_path} (extension {wcs_extension})"
        )

    try:
        with fits.open(file_path) as hdul:
            wcs_info = WCS(hdul[wcs_extension].header)

    except Exception as e:
        if logger:
            logger.error(f"Error extracting WCS from FITS image at {file_path}: {e}")
        return None

    if logger:
        logger.info(
            f"Successfully extracted WCS from FITS image at {file_path} (extension {wcs_extension})"
        )

    return wcs_info


# list of what header entries to extract from the frames
OUTPUT_TABLE_ENTRIES = [
    "DATE-OBS",
    "MJD-OBS",
    "HIERARCH ESO TPL START",
    "HIERARCH MUSEPIPE_IEXPO",
    "HIERARCH MUSEPIPE_DATASET",  # for NGC3221
    # "HIERARCH MUSEPIPE_POINTING" # for NGC1365
]

OUTPUT_TABLE_HEADERS = [
    "DATE_OBS",
    "MJD_OBS",
    "TPL_START",
    "IEXPO_OBS",
    "POINTING_OBS",
    "RA_OFFSET",
    "RA_OFFSET_ERROR",
    "DEC_OFFSET",
    "DEC_OFFSET_ERROR",
    "ROTANGLE",
    "ROTATION_ERROR",
]


def make_table_entry(
    hdul,
    dRA,
    dRA_sigma,
    dDEC,
    dDEC_sigma,
    dROT,
    dROT_sigma,
    output_table_entries,
    extension=0,
    logger=None,
):
    """
    Creates a table entry for the pipeline output, which is a tupple of
    OUTPUT_TABLE_ENTRIES and the values of dRA, dRA_sigma, dDEC, dDEC_sigma, dROT, dROT_sigma.

    Parameters
    ----------
    hdul : astropy.io.fits.HDUList
        The HDU list containing the FITS file data.
    extension : int
        The extension number of the HDU to create the table entry for.
    dRA : float
        The change in Right Ascension (RA).
    dRA_sigma : float
        The uncertainty in the change in RA.
    dDEC : float
        The change in Declination (DEC).
    dDEC_sigma : float
        The uncertainty in the change in DEC.
    dROT : float
        The change in rotation angle.
    dROT_sigma : float
        The uncertainty in the change in rotation angle.

    Returns
    -------
    dict
        A dictionary representing the table entry with the provided values.
    """

    table_entry = {}
    for entry in output_table_entries:
        try:
            table_entry[entry] = hdul[extension].header[entry]
        except KeyError:
            if logger:
                logger.warning(
                    f"Header entry '{entry}' not found in FITS file (extension {extension}). Setting to None."
                )
            table_entry[entry] = None

    table_entry["dRA"] = dRA
    table_entry["dRA_sigma"] = dRA_sigma
    table_entry["dDEC"] = dDEC
    table_entry["dDEC_sigma"] = dDEC_sigma
    table_entry["dROT"] = dROT
    table_entry["dROT_sigma"] = dROT_sigma

    if logger:
        logger.info(f"Created table entry for output:\n      {table_entry}")

    return table_entry


def write_final_results(
    output_table, method_name="", path=".", logger=None, pymusepipe_flux=None
):
    """
    Writes the final results to a FITS table file.

    Needs to be adapted to OUTPUT_FILE_HEADERS.

    Parameters
    ----------
    output_table : list of tuples
        The list of tuples containing the final results to be written to the FITS table.
        (See ``make_table_entry`` for the structure of each tuple.)

    path : str, optional
        The path where the output FITS file will be saved. Default is the current directory.

    logger : logging.Logger, optional
        The logger to use for logging messages. If None, no logging will be performed.

    pymusepipe_flux : dict, optional
        Inputs for the FLUX_SCALE and BACKGROUND columns of the pymusepipe
        offset table, see ``write_pymusepipe_offset_table``. If None, they are
        written as 1 and 0.
    """

    # integer columns cannot hold None (a header entry missing in a file, or
    # an exposure that could not be read): write -1 instead of failing the
    # whole table
    def _int_column(index, name):
        """Integer column from field `index` of the output rows, -1 where the value is None."""
        values = [row[index] for row in output_table]
        n_missing = sum(v is None for v in values)
        if n_missing and logger:
            logger.warning(f"{n_missing} row(s) have no {name}, writing -1 for them.")
        return np.array([-1 if v is None else v for v in values], dtype=np.int32)

    # Build columns explicitly so dtypes are correct (esp. the string columns)
    columns = {
        "DATE_OBS": np.array([row[0] for row in output_table], dtype="U23"),
        "MJD_OBS": np.array([row[1] for row in output_table], dtype=np.float64),
        "TPL_START": np.array([row[2] for row in output_table], dtype="U19"),
        "IEXPO_OBS": _int_column(3, "IEXPO_OBS"),
        "POINTING_OBS": _int_column(4, "POINTING_OBS"),
        "RA_OFFSET": np.array([row[5] for row in output_table], dtype=np.float64),
        "RA_OFFSET_ERROR": np.array([row[6] for row in output_table], dtype=np.float64),
        "DEC_OFFSET": np.array([row[7] for row in output_table], dtype=np.float64),
        "DEC_OFFSET_ERROR": np.array(
            [row[8] for row in output_table], dtype=np.float64
        ),
        "ROTANGLE": np.array([row[9] for row in output_table], dtype=np.float64),
        "ROTATION_ERROR": np.array([row[10] for row in output_table], dtype=np.float64),
    }

    table = Table(columns, names=OUTPUT_TABLE_HEADERS)

    if logger:
        logger.info(f"Writing final results to {path}")

    table.write(
        path + f"/{method_name}_alignment_output_table.fits",
        format="fits",
        overwrite=True,
    )

    if logger:
        logger.info(f"Successfully wrote final results to {path}")

    write_pymusepipe_offset_table(
        table,
        path + f"/{method_name}_pymusepipe_offset_table.fits",
        pymusepipe_flux=pymusepipe_flux,
        logger=logger,
    )


# ROTANGLE(pymusepipe) = PYMUSEPIPE_ROTATION_SIGN * ROTANGLE(this pipeline).
# -1 follows from the pymusepipe code (it uses minus the spacepylot rotation,
# and spacepylot rotates in the same direction as this pipeline), not yet
# verified on data.
PYMUSEPIPE_ROTATION_SIGN = -1

# filter of the MUSE images and the reference for pymusepipe's flux
# normalisation (sets the unit conversion of the reference, see
# `pymusepipe.config_pipe.dict_equivalencies`); pymusepipe's default
PYMUSEPIPE_FILTER_NAME = "WFI_BB"


def write_pymusepipe_offset_table(
    table,
    filename,
    rotation_sign=PYMUSEPIPE_ROTATION_SIGN,
    pymusepipe_flux=None,
    logger=None,
):
    """
    Writes the results as an OFFSET table in the format pymusepipe expects
    (``name_offset_table`` of ``finalise_reduction`` / ``mosaic``).

    Columns: DATE_OBS, MJD_OBS, TPL_START, IEXPO_OBS, DATASET_OBS, RA_OFFSET,
    DEC_OFFSET (degrees, same convention as pymusepipe), ROTANGLE (degrees,
    `rotation_sign` times ours), FLUX_SCALE and BACKGROUND. The errors (and
    ALIGNMENT_METHOD, if present) are kept as extra columns, which pymusepipe
    ignores.

    FLUX_SCALE and BACKGROUND are computed by pymusepipe itself if
    `pymusepipe_flux` is given (see ``pymusepipe_flux_columns``), else they are
    1 and 0. If pymusepipe fails, the error is logged and they are 1 and 0 as
    well; the table header keyword FLUXSRC says which ("pymusepipe" or "none").

    pymusepipe matches the pixel tables to the rows by MJD-OBS and DATE-OBS.
    Exposures without a result (failed or inconsistent alignments) are left
    out, so pymusepipe leaves them out of the combination.

    Parameters
    ----------
    table : astropy.table.Table
        Results with the columns of ``OUTPUT_TABLE_HEADERS`` (the method or
        hybrid output table).

    filename : str
        Path of the output FITS file.

    rotation_sign : int, optional
        +1 or -1, see ``PYMUSEPIPE_ROTATION_SIGN``.

    pymusepipe_flux : dict, optional
        Inputs for ``pymusepipe_flux_columns``: "config_dict", "input_dir",
        "reference_file" and optionally "filter_name". If None, FLUX_SCALE is
        1 and BACKGROUND is 0.

    logger : logging.Logger, optional
        The logger to use for logging messages. If None, no logging will be performed.

    Returns
    -------
    astropy.table.Table
        The table that was written.
    """

    if rotation_sign not in (1, -1):
        raise ValueError(f"rotation_sign must be +1 or -1, got {rotation_sign}")

    offset_table = Table()
    offset_table["DATE_OBS"] = np.asarray(table["DATE_OBS"], dtype="U23")
    offset_table["MJD_OBS"] = np.asarray(table["MJD_OBS"], dtype=np.float64)
    offset_table["TPL_START"] = np.asarray(table["TPL_START"], dtype="U19")
    offset_table["IEXPO_OBS"] = np.asarray(table["IEXPO_OBS"], dtype=np.int32)
    offset_table["DATASET_OBS"] = np.asarray(table["POINTING_OBS"], dtype=np.int32)
    offset_table["RA_OFFSET"] = np.asarray(table["RA_OFFSET"], dtype=np.float64)
    offset_table["DEC_OFFSET"] = np.asarray(table["DEC_OFFSET"], dtype=np.float64)
    offset_table["ROTANGLE"] = rotation_sign * np.asarray(
        table["ROTANGLE"], dtype=np.float64
    )
    offset_table["FLUX_SCALE"] = np.ones(len(offset_table))
    offset_table["BACKGROUND"] = np.zeros(len(offset_table))

    for column in ("RA_OFFSET_ERROR", "DEC_OFFSET_ERROR", "ROTATION_ERROR"):
        offset_table[column] = np.asarray(table[column], dtype=np.float64)
    if "ALIGNMENT_METHOD" in table.colnames:
        offset_table["ALIGNMENT_METHOD"] = np.asarray(
            table["ALIGNMENT_METHOD"], dtype="U16"
        )

    # never write NaN offsets: pymusepipe would use them
    usable = (
        np.isfinite(offset_table["RA_OFFSET"])
        & np.isfinite(offset_table["DEC_OFFSET"])
        & np.isfinite(offset_table["ROTANGLE"])
        & np.isfinite(offset_table["MJD_OBS"])
    )
    if not np.all(usable) and logger:
        logger.warning(
            f"pymusepipe offset table: leaving out {int(np.sum(~usable))} exposure(s) "
            f"without a result (not combined by pymusepipe): "
            f"{', '.join(str(d) for d in offset_table['DATE_OBS'][~usable])}"
        )
    offset_table = offset_table[usable]

    offset_table.meta["ROTSIGN"] = rotation_sign
    offset_table.meta["FLUXSRC"] = "none"

    # FLUX_SCALE and BACKGROUND from pymusepipe, on the final offsets. This
    # runs at the very end of a (long) alignment run, so a problem here must
    # not lose the offsets: it is logged and the table is written with 1 / 0.
    if pymusepipe_flux is not None and len(offset_table) > 0:
        work_dir = os.path.join(
            os.path.dirname(os.path.abspath(filename)), "pymusepipe_flux"
        )
        try:
            flux_scale, background = pymusepipe_flux_columns(
                offset_table, work_dir=work_dir, logger=logger, **pymusepipe_flux
            )
            offset_table["FLUX_SCALE"] = flux_scale
            offset_table["BACKGROUND"] = background
            offset_table.meta["FLUXSRC"] = "pymusepipe"
        except Exception:
            if logger:
                logger.exception(
                    "pymusepipe flux normalisation failed - writing the pymusepipe "
                    "offset table with FLUX_SCALE 1 and BACKGROUND 0."
                )
    elif logger:
        logger.info(
            "pymusepipe offset table: FLUX_SCALE 1 and BACKGROUND 0 (flux "
            "normalisation not requested for this table)."
        )

    offset_table.write(filename, format="fits", overwrite=True)

    if logger:
        logger.info(
            f"Wrote pymusepipe offset table ({len(offset_table)} exposures, "
            f"ROTANGLE sign {rotation_sign:+d}, FLUX_SCALE / BACKGROUND from "
            f"{offset_table.meta['FLUXSRC']}) to {filename}"
        )

    return offset_table


class _LogWriter:
    """File-like object that writes every printed line to the logger."""

    def __init__(self, logger, prefix=""):
        self.logger = logger
        self.prefix = prefix
        self._buffer = ""

    def write(self, message):
        """Log every complete, non-empty line (print writes a line in pieces)."""
        self._buffer += message
        *lines, self._buffer = self._buffer.split("\n")
        for line in lines:
            if self.logger and line.strip():
                self.logger.info(f"{self.prefix}{line.rstrip()}")

    def flush(self):
        """Log what is left in the buffer."""
        if self.logger and self._buffer.strip():
            self.logger.info(f"{self.prefix}{self._buffer.rstrip()}")
        self._buffer = ""


def pymusepipe_flux_columns(
    offset_table,
    config_dict,
    input_dir,
    reference_file,
    work_dir,
    filter_name=PYMUSEPIPE_FILTER_NAME,
    logger=None,
):
    """
    Computes FLUX_SCALE and BACKGROUND with pymusepipe's own normalisation,
    on the offsets of `offset_table`.

    pymusepipe's ``AlignMuseDataset`` with ``firstguess="fits"`` reads the
    offsets and rotations of the table (matched to the images by MJD-OBS),
    shifts every MUSE image accordingly, reprojects the reference onto it
    (with the unit conversion of `filter_name`), and fits
    ``reference = FLUX_SCALE * (MUSE + BACKGROUND)`` to the chunk medians of
    the two images (``get_normfactor``). This is the same code the pymusepipe
    alignment uses for these columns. No offsets are measured by pymusepipe.

    Only the images of the exposures in `offset_table` are passed to
    pymusepipe (images without a result would be fitted at zero offset).

    Parameters
    ----------
    offset_table : astropy.table.Table
        The pymusepipe offset table (``write_pymusepipe_offset_table``),
        without the failed exposures.

    config_dict : dict
        The configuration of the run (pointing -> [exposure, DATE-OBS,
        file name]), used to find the image of every row by DATE_OBS.

    input_dir : str
        Directory of the MUSE images.

    reference_file : str
        Path of the reference image.

    work_dir : str
        Directory for pymusepipe's input table and its (empty) header and
        figure folders.

    filter_name : str, optional
        Filter for pymusepipe's unit conversion of the reference. Default is
        ``PYMUSEPIPE_FILTER_NAME``.

    logger : logging.Logger, optional
        The logger to use for logging messages (pymusepipe's printed output is
        sent to it as well).

    Returns
    -------
    flux_scale, background : np.ndarray
        One value per row of `offset_table`. Rows pymusepipe gives no finite
        value for get 1 and 0, with a warning.
    """

    # imported here: only needed for this step, and the entry points set the
    # matplotlib backend before (pymusepipe imports pyplot)
    from pymusepipe.align_pipe import AlignMuseDataset

    # image of every exposure, by DATE-OBS (as the table)
    files_by_date = {}
    for pointing, exposures in config_dict.items():
        if pointing == "headers":
            continue
        for exposure in exposures:
            files_by_date[str(exposure[1]).strip()] = exposure[2]

    dates = [str(date).strip() for date in offset_table["DATE_OBS"]]
    missing = [date for date in dates if date not in files_by_date]
    if missing:
        raise ValueError(
            f"No image in the configuration for DATE_OBS {', '.join(missing)}."
        )
    name_images = [files_by_date[date] for date in dates]

    os.makedirs(work_dir, exist_ok=True)
    name_input_table = "input_offset_table.fits"
    offset_table.write(
        os.path.join(work_dir, name_input_table), format="fits", overwrite=True
    )

    reference_file = os.path.abspath(reference_file)

    if logger:
        logger.info(
            f"pymusepipe flux normalisation: {len(name_images)} images, filter "
            f"{filter_name}, reference {reference_file}, working directory {work_dir}"
        )

    writer = _LogWriter(logger, prefix="pymusepipe: ")
    with contextlib.redirect_stdout(writer):
        dataset = AlignMuseDataset(
            name_reference=os.path.basename(reference_file),
            folder_reference=os.path.dirname(reference_file),
            folder_muse_images=os.path.abspath(input_dir),
            name_muse_images=name_images,
            firstguess="fits",
            name_offset_table=name_input_table,
            folder_offset_table=work_dir,
            folder_output_table=work_dir,
            header_folder_name=os.path.join(work_dir, "AlignHeaders"),
            figures_folder_name=os.path.join(work_dir, "AlignFigures"),
            filter_name=filter_name,
            plot=False,
        )
    writer.flush()

    # AlignMuseDataset returns early (without raising) if it cannot open the
    # images; then no normalisation was fitted
    polypar = getattr(dataset, "ima_polypar", None)
    if polypar is None or len(polypar) != len(name_images) or any(
        p is None for p in polypar
    ):
        raise RuntimeError(
            "pymusepipe did not fit the flux normalisation (see its messages above)."
        )

    by_mjd = {
        float(mjd): (float(scale), float(back))
        for mjd, scale, back in zip(
            dataset.ima_mjdobs, dataset.ima_norm_factors, dataset.ima_background
        )
    }

    flux_scale = np.ones(len(offset_table))
    background = np.zeros(len(offset_table))
    for i, (date, mjd) in enumerate(zip(dates, offset_table["MJD_OBS"])):
        scale, back = by_mjd.get(float(mjd), (np.nan, np.nan))
        if np.isfinite(scale) and np.isfinite(back):
            flux_scale[i], background[i] = scale, back
        elif logger:
            logger.warning(
                f"pymusepipe gave no flux normalisation for {date} (MJD {mjd}), "
                f"writing FLUX_SCALE 1 and BACKGROUND 0 for it."
            )

    if logger:
        logger.info(
            f"pymusepipe flux normalisation: FLUX_SCALE median "
            f"{np.median(flux_scale):.4f} (range {flux_scale.min():.4f} - "
            f"{flux_scale.max():.4f}), BACKGROUND median {np.median(background):.4g}"
        )

    return flux_scale, background
