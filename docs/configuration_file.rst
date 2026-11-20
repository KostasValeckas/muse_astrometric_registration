.. _conf:

Configuration files
===================

The pipeline input is a ``.json`` file with the following formatting:

.. code-block:: text

    {
        Pointing number (str): [
            [
                Dither number (int),
                Observation time (str),
                Filename (str)
            ],
            ...
        ],
        ...
        "headers": {
            "date_keyword": fits keyword for date of observation (str),
            "mjd_keyword": fits keyword for MJD of observation (str),
            "tpl_start_keyword": fits keyword for template start (str),
            "exp_keyword": fits keyword for exposure number (str),
            "pointing_keyword": fits keyword for pointing number (str)
        }
    }

There are several examples in the ``dev`` directory. In practice, the input
file will look something like this (snippet):

.. code-block:: json

    {
        "1": [
            [
                1,
                "2021-11-27T03:13:14.863",
                "IMAGE_FOV_prealign_OB001_WFI_BB_2021-11-27T03:07:22_0001.fits"
            ],
            [
                2,
                "2021-11-27T03:18:29.277",
                "IMAGE_FOV_prealign_OB001_WFI_BB_2021-11-27T03:07:22_0002.fits"
            ]
        ],
        "headers": {
            "date_keyword": "DATE-OBS",
            "mjd_keyword": "MJD-OBS",
            "tpl_start_keyword": "HIERARCH ESO TPL START",
            "exp_keyword": "HIERARCH MUSEPIPE_IEXPO",
            "pointing_keyword": "MUSEPIPE_DATASET"
        }
    }


.. _sorter:

Preparing the configuration file
--------------------------------

The script ``dev/sorter.py`` produces a configuration file from a directory with
MUSE data. Three FITS keywords control the sorting of the data: the pointing
number, the exposure number and the observation time. The defaults are:

.. code-block:: text

    pointing_keyword = "HIERARCH MUSEPIPE_POINTING"
    exp_keyword = "HIERARCH MUSEPIPE_IEXPO"
    date_keyword = "DATE-OBS"

The script can be called with the default headers by giving the input directory:

.. code-block:: bash

    python sorter.py /path/to/directory/with/muse/data

For some datasets the headers differ. The following flags allow for that:

.. code-block:: text

      --pointing_keyword POINTING_KEYWORD
                            Keyword for the pointing. Default is 'HIERARCH MUSEPIPE_POINTING'.
      --exp_keyword EXP_KEYWORD
                            Keyword for the exposure time. Default is 'HIERARCH MUSEPIPE_IEXPO'.
      --date_keyword DATE_KEYWORD
                            Keyword for the observation date. Default is 'DATE-OBS'.

These flags specify the output directory and filename, and two more headers that
are used for the output table of the pipeline (not needed for the sorting):

.. code-block:: text

      --output_dir OUTPUT_DIR
                            Directory to save the results. Default is current working directory.
      --output_filename OUTPUT_FILENAME
                            Name of the output JSON file. Default is 'sorted_files.json'.
      --mjd_keyword MJD_KEYWORD
                            Keyword for the observation MJD. Default is 'MJD-OBS'.
      --tpl_start_keyword TPL_START_KEYWORD
                            Keyword for the template start. Default is 'HIERARCH ESO TPL START'.
