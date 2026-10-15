import argparse
import json
import os
from pipeline import AlignmentPipeline


"""
Reads the configuration file and runs the alignment pipeline.
"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run aligment pipeline")

    parser.add_argument(
        "config_file",
        type=str,
        help="Path to the configuration file. File configuration (JSON format): \n"
        '{ "HIERARCH_MUSEPIPE_POINTING": [HIERARCH_MUSEPIPE_IEXPO, DATE-OBS, FILENAME], ... }, ... }\n',
    )

    parser.add_argument("ref_file_path", type=str, help="Path to the reference file.")

    parser.add_argument(
        "--input_dir",
        type=str,
        default=".",
        help="Path to the input directory. Default is current working directory.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Path to the output directory. Default is current working directory.",
    )

    args = parser.parse_args()

    # create the output directory if it does not exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(args.config_file, "r") as f:
        config = json.load(f)

    # create the pipeline
    pipeline = AlignmentPipeline(
        config_dict=config,
        input_dir=args.input_dir,
        reference_file=args.ref_file_path,
        output_dir=args.output_dir,
    )

    pipeline.run()
