import json
import sys 
import os
from astropy.io import fits
import argparse
import pathlib


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Sort MUSE data for aligment pipeline")

    parser.add_argument("input_dir", help="Input directory")

    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=pathlib.Path.cwd(),
        help="Directory to save the results. Default is current working directory.",
    )

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    # all ".fits" files in the input directory
    fits_files = [f for f in os.listdir(input_dir) if f.endswith(".fits")]

    # sort the files by name 
    fits_files.sort()


    sorted_files = {}

    for file in fits_files:
        with fits.open(os.path.join(input_dir, file)) as hdul:
            header = hdul[0].header

            IEXPO_OBS = header.get("HIERARCH MUSEPIPE_IEXPO", None)
            POINTING_OBS = header.get("HIERARCH MUSEPIPE_POINTING", None)
            DATE_OBS = header.get("DATE-OBS", None)

            print(f"File: {file}, IEXPO_OBS: {IEXPO_OBS}, POINTING_OBS: {POINTING_OBS}, DATE_OBS: {DATE_OBS}")

            if POINTING_OBS not in sorted_files:
                sorted_files[POINTING_OBS] = []

            sorted_files[POINTING_OBS].append((IEXPO_OBS, DATE_OBS, file))

