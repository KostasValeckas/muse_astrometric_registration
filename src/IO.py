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


def write_final_results(output_table, method_name="", path=".", logger=None):
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


    """

    # Build columns explicitly so dtypes are correct (esp. the string columns)
    columns = {
        "DATE_OBS": np.array([row[0] for row in output_table], dtype="U23"),
        "MJD_OBS": np.array([row[1] for row in output_table], dtype=np.float64),
        "TPL_START": np.array([row[2] for row in output_table], dtype="U19"),
        "IEXPO_OBS": np.array([row[3] for row in output_table], dtype=np.int32),
        "POINTING_OBS": np.array([row[4] for row in output_table], dtype=np.int32),
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
