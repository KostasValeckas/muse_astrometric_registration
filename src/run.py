import argparse
import json

if __package__ is None or __package__ == "":
    import pathlib
    import sys

    # Allow `python src/run.py ...` by adding the project root to sys.path.
    project_root = pathlib.Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.pipeline import AlignmentPipeline
else:
    from .pipeline import AlignmentPipeline


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

    # for now, just load all the filenames

    filenames = []

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
