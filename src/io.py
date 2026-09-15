from astropy.io import fits

def read_fits_image(file_path, logger = None, data_extension=0):
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
            logger.error(f"No data found in FITS image at {file_path} (extension {data_extension})")
        return None

    if logger:
        logger.info(f"Successfully read FITS image from {file_path} (extension {data_extension})")

    return image_data