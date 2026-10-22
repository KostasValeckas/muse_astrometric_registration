"""
Reads the configuration file and runs the alignment pipeline.
"""

if __name__ == "__main__":


    import argparse
    import os
    import json

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

    parser.add_argument(
        "--plot_headless",
        action="store_true",
        help="If set, plots will be generated in headless mode (no GUI). Default is False.",
    )

    parser.add_argument(
        "--method",
        type=str,
        default="NoAlignment",
        help="The alignment method to use. Default is NoAlignment. Options are NoAlignment, PointSources.",
    )

    parser.add_argument(
        "--distance_threshold",
        type=float,
        default=0.10,
        help="The distance threshold for consistency checks. Default is 0.10.",
    )

    args = parser.parse_args()


    import matplotlib
    if args.plot_headless: matplotlib.use("Agg")

    # create the output directory if it does not exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(args.config_file, "r") as f:
        config = json.load(f)

    # imports happen here so we can set global backends for matplotlib 
    # before importing any other modules that use it
    
    from pipeline import AlignmentPipeline
    from methods import NoAlignment, PointSources, FFT, OpticalFlow

    # TODO: this method selection needs to be refactored
    
    method = None

    switcher = {
        "NoAlignment": NoAlignment,
        "PointSources": PointSources,
        "FFT": FFT,
        "OpticalFlow": OpticalFlow,
    }

    if args.method in switcher:
        method = switcher[args.method]

    else:
        raise ValueError(
            f"Invalid method: {args.method}. Choose from 'NoAlignment', 'PointSources', 'FFT', 'OpticalFlow'."
        )


    # create the pipeline
    pipeline = AlignmentPipeline(
        config_dict=config,
        input_dir=args.input_dir,
        reference_file=args.ref_file_path,
        output_dir=args.output_dir,
        aligment_method=method,
        distance_threshold=args.distance_threshold
    )

    pipeline.run()
