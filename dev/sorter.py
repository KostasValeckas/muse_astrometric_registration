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

    parser.add_argument(
        "--output_filename",
        type=str,
        default="sorted_files.json",
        help="Name of the output JSON file. Default is 'sorted_files.json'.",
    )

    parser.add_argument(
        "--pointing_keyword",
        type=str,
        default="HIERARCH MUSEPIPE_POINTING",
        help="Keyword for the pointing. Default is 'HIERARCH MUSEPIPE_POINTING'.",
    )

    parser.add_argument(
        "--exp_keyword",
        type=str,
        default="HIERARCH MUSEPIPE_IEXPO",
        help="Keyword for the exposure time. Default is 'HIERARCH MUSEPIPE_IEXPO'.",
    )

    parser.add_argument(
        "--date_keyword",
        type=str,
        default="DATE-OBS",
        help="Keyword for the observation date. Default is 'DATE-OBS'.",
    )

    parser.add_argument(
        "--mjd_keyword",
        type=str,
        default="MJD-OBS",
        help="Keyword for the observation MJD. Default is 'MJD-OBS'.",
    )

    parser.add_argument(
        "--tpl_start_keyword",
        type=str,
        default="HIERARCH ESO TPL START",
        help="Keyword for the template start. Default is 'HIERARCH ESO TPL START'.",
    )

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    output_filename = args.output_filename
    os.makedirs(output_dir, exist_ok=True)

    POINTING_OBS_KEYWORD = args.pointing_keyword
    IEXPO_OBS_KEYWORD = args.exp_keyword
    DATE_OBS_KEYWORD = args.date_keyword
    TPL_START_OBS_KEYWORD = args.tpl_start_keyword

    # all ".fits" files in the input directory
    fits_files = [f for f in os.listdir(input_dir) if f.endswith(".fits")]

    # sort the files by name
    fits_files.sort()

    sorted_files = {}

    for file in fits_files:
        with fits.open(os.path.join(input_dir, file)) as hdul:
            header = hdul[0].header

            IEXPO_OBS = header.get(IEXPO_OBS_KEYWORD, None)
            POINTING_OBS = header.get(POINTING_OBS_KEYWORD, None)
            DATE_OBS = header.get(DATE_OBS_KEYWORD, None)
            TPL_START_OBS = header.get(TPL_START_OBS_KEYWORD, None)

            # if any are None, print a warning and skip the file
            if IEXPO_OBS is None:
                print(f"Warning: {IEXPO_OBS_KEYWORD} not found in {file}. Skipping.")
                continue
            if POINTING_OBS is None:
                print(f"Warning: {POINTING_OBS_KEYWORD} not found in {file}. Skipping.")
                continue
            if DATE_OBS is None:
                print(f"Warning: {DATE_OBS_KEYWORD} not found in {file}. Skipping.")
                continue
            if TPL_START_OBS is None:
                print(f"Warning: {TPL_START_OBS_KEYWORD} not found in {file}. Skipping.")
                continue

            print(
                f"File: {file}, IEXPO_OBS: {IEXPO_OBS}, POINTING_OBS: {POINTING_OBS}, DATE_OBS: {DATE_OBS}, TPL_START_OBS: {TPL_START_OBS}"
            )

            if POINTING_OBS not in sorted_files:
                sorted_files[POINTING_OBS] = []

            sorted_files[POINTING_OBS].append((IEXPO_OBS, TPL_START_OBS, DATE_OBS, file))


    # sort the files within each pointing by exposure number, then template start
    for pointing in sorted_files:
        sorted_files[pointing].sort(
            key=lambda x: (
                x[0] if x[0] is not None else -1,
                x[1] if x[1] is not None else "",
                x[2] if x[2] is not None else "",
                x[3],
            )
        )

        # renumber the exposures so different template starts count as extra slots
        sorted_files[pointing] = [
            (exposure_number, date_obs, file)
            for exposure_number, (_, _, date_obs, file) in enumerate(sorted_files[pointing], start=1)
        ]

    # lastly, for bookingeeping, we add the headers used to sort the files to the dictionary
    sorted_files["headers"] = {
        "date_keyword": DATE_OBS_KEYWORD,
        "mjd_keyword": args.mjd_keyword,
        "tpl_start_keyword": args.tpl_start_keyword,
        "exp_keyword": IEXPO_OBS_KEYWORD,
        "pointing_keyword": POINTING_OBS_KEYWORD
    }

    # if .json extension does not exist to the output filename, add it
    if not output_filename.endswith(".json"):
        output_filename += ".json"

    # save the sorted files to a JSON file
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, "w") as f:
        json.dump(sorted_files, f, indent=4)