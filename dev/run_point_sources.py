###
# This block is used so we can access the src diretory for packages.
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
###

from src.logger import get_logger
from src.methods import PointSources
from src.io import read_fits_image
from src.filters import remove_nans

import argparse
import pathlib


"""
Main method for now a stump for simple testing.
"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run the Point Sources alignment on two images.")

    # TODO: these will later be exchanged with full litsts
    parser.add_argument(
        "reference_image", type=pathlib.Path, help="Path to the reference image."
    )
    parser.add_argument(
        "input_image", type=pathlib.Path, help="Path to the input image."
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
    logger.info(f"Input image: {args.input_image}")

    output_dir = args.output_dir

    logger.info(f"Output directory: {output_dir}")


    # Initialize the PointSources alignment method
    point_sources_aligner = PointSources(logger=logger)

    # Read the FITS images
    reference_frame = read_fits_image(args.reference_image, logger=logger)
    input_frame = read_fits_image(args.input_image, logger=logger, data_extension=1)

    # Remove NaN values from the images
    reference_frame = remove_nans(reference_frame, logger=logger)
    input_frame = remove_nans(input_frame, logger=logger)


    point_sources_aligner.run(
        reference_frame,
        input_frame,
        "test_obs"
    )
