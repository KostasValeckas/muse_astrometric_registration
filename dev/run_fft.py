###
# This block is used so we can access the src diretory for packages.
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
###

from src.logger import get_logger
from src.methods import FFT
from src.IO import read_fits_image
from src.filters import *

from astropy.wcs import WCS

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np

"""
Main method for now a stump for simple testing.
"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run the Point Sources alignment on two images."
    )

    # TODO: these will later be exchanged with full litsts
    parser.add_argument(
        "reference_image", type=pathlib.Path, help="Path to the reference image."
    )
    parser.add_argument(
        "input_directory", type=pathlib.Path, help="Path to the input directory."
    )

    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=pathlib.Path.cwd(),
        help="Directory to save the results and log file. Default is current working directory.",
    )

    args = parser.parse_args()

    # Initialize logger
    logger = get_logger(args.output_dir)

    logger.info(f"Reference image: {args.reference_image}")
    logger.info(f"Input directory: {args.input_directory}")

    output_dir = args.output_dir

    logger.info(f"Output directory: {output_dir}")

    fits_file_list_input = list(args.input_directory.glob("*.fits"))

    # sort

    fits_file_list_input.sort()

    all_rots = []

    for input_image in fits_file_list_input:
        logger.info(f"Processing input image: {input_image}")

        # Initialize the FFT alignment method
        fft_aligner = FFT(logger=logger)

        # Read the FITS images
        reference_frame, reference_hdul = read_fits_image(
            args.reference_image, logger=logger
        )
        input_frame, input_hdul = read_fits_image(
            input_image, logger=logger, data_extension=1
        )

        reference_wcs = WCS(reference_hdul[0].header)
        input_wcs = WCS(input_hdul[1].header)

        (tx, ty), rot, (tx_err, ty_err), rot_err, success = fft_aligner.run(
            reference_frame,
            input_frame,
            input_image.stem,
            reference_wcs=reference_wcs,
            input_wcs=input_wcs,
            show_QA=True,
        )

        print("Got rot: ", rot)

        all_rots.append(rot)

    # write wrot array to disc

    all_rots = np.array(all_rots)
    all_rots_file = output_dir / "all_rots.npy"
    np.save(all_rots_file, all_rots)

    plt.plot(all_rots)
    plt.show()
