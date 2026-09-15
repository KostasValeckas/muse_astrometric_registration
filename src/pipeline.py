from logger import get_logger

import argparse
import pathlib


"""
Main method for now a stump for simple testing.
"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run the alignment pipeline.")

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
