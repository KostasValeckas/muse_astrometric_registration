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

    parser.add_argument(
        "--skip_rotation_estimation",
        action="store_true",
        help="If set, rotation estimation will be skipped. Default is False.",
    )

    parser.add_argument(
        "--scan_for_angles",
        action="store_true",
        help="If set, use the shared angle scan to estimate the rotation. "
        "Default is False (use each method's direct rotation estimator).",
    )

    parser.add_argument(
        "--dither_mode",
        type=str,
        choices=("full", "targeted"),
        default="full",
        help="'full' aligns all dither pairs, 'targeted' only the pairs needed to "
        "recover failed science alignments. Default is 'full'.",
    )

    parser.add_argument(
        "--scale-ref",
        "--scale_ref",
        dest="scale_ref",
        action="store_true",
        help="If set, bring the reference to the flux scale of every exposure before "
        "its science alignment, as pymusepipe does (unit conversion for --ref_filter, "
        "pixel area, and the fit reference = FLUX_SCALE * (exposure + BACKGROUND)), "
        "and write FLUX_SCALE and BACKGROUND to the output table. Dither alignments "
        "are not affected. Default is False.",
    )

    parser.add_argument(
        "--ref_filter",
        type=str,
        default="WFI_BB",
        help="Filter of the reference image, for the unit conversion of --scale-ref "
        "(see REFERENCE_FILTERS in flux.py). Default is 'WFI_BB'.",
    )

    args = parser.parse_args()

    import matplotlib

    if args.plot_headless:
        matplotlib.use("Agg")

    # create the output directory if it does not exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(args.config_file, "r") as f:
        config = json.load(f)

    # imports happen here so we can set global backends for matplotlib
    # before importing any other modules that use it

    from pipeline import AlignmentPipeline
    from logger import exit_on_sigterm

    # a batch time limit (SIGTERM) is logged as ABORTED instead of silently
    exit_on_sigterm()
    from methods import NoAlignment, PointSources, FFT, OpticalFlow

    # TODO: this method selection needs to be refactored

    method = None

    switcher = {
        "NoAlignment": NoAlignment,
        "PointSources": PointSources,
        "FFT": FFT,
        "OpticalFlow": OpticalFlow,
        "All": [PointSources, OpticalFlow, FFT],
    }

    if args.method in switcher:
        method = switcher[args.method]

    else:
        raise ValueError(
            f"Invalid method: {args.method}. Choose from 'NoAlignment', 'PointSources', 'FFT', 'OpticalFlow'."
        )

    if args.method == "All":
        # run the pipeline for each method in the list, this option mostly for testing
        for method in switcher[args.method]:
            # create the pipeline
            pipeline = AlignmentPipeline(
                config_dict=config,
                input_dir=args.input_dir,
                reference_file=args.ref_file_path,
                output_dir=args.output_dir,
                aligment_method=method,
                distance_threshold=args.distance_threshold,
                skip_rotation_estimation=args.skip_rotation_estimation,
                scan_for_angles=args.scan_for_angles,
                dither_mode=args.dither_mode,
            scale_ref=args.scale_ref,
            ref_filter=args.ref_filter,
            )

            pipeline.run()

    # singular cases
    else:

        # create the pipeline
        pipeline = AlignmentPipeline(
            config_dict=config,
            input_dir=args.input_dir,
            reference_file=args.ref_file_path,
            output_dir=args.output_dir,
            aligment_method=method,
            distance_threshold=args.distance_threshold,
            skip_rotation_estimation=args.skip_rotation_estimation,
            scan_for_angles=args.scan_for_angles,
            dither_mode=args.dither_mode,
            scale_ref=args.scale_ref,
            ref_filter=args.ref_filter,
        )

        pipeline.run()
