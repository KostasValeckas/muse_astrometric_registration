import argparse
import json
from pipeline import AlignmentPipeline



"""
Reads the configuration file and runs the alignment pipeline.
"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run aligment pipeline")

    parser.add_argument(
        "config_file", 
        type=str, 
        help="Path to the configuration file. File configuration (JSON format): \n" \
        "{ \"HIERARCH_MUSEPIPE_POINTING\": [HIERARCH_MUSEPIPE_IEXPO, DATE-OBS, FILENAME], ... }, ... }\n" \
    )

    parser.add_argument(
        "ref_file_path",
        type=str,
        help="Path to the reference file."
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        default=".",
        help="Path to the input directory. Default is current working directory."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Path to the output directory. Default is current working directory."
    )

    args = parser.parse_args()

    # for now, just load all the filenames

    filenames = []

    with open(args.config_file, "r") as f:
        config = json.load(f)

    for pointing, exposures in config.items():
        for exposure in exposures:
            filenames.append(exposure[2])

    # create the pipeline
    pipeline = AlignmentPipeline(
        input_files=[f"{args.input_dir}/{filename}" for filename in filenames],
        reference_file=args.ref_file_path,
        output_dir=args.output_dir
    )
    pipeline.run(args.config_file, args.ref_file_path, args.output_dir)
