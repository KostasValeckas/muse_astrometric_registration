import argparse
import json



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
        "--output_dir",
        type=str,
        default=".",
        help="Path to the output directory. Default is current working directory."
    )

    args = parser.parse_args()