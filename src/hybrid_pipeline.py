"""
Hybrid pipeline that first runs a secondary and tretary method, and then uses tje results
to give inital guesses to the primary method if available.

Order of execution:

1. The secondary and tertiary methods are run as isolated pipelines (as `run.py`
   would), without initial guesses. Angle scan is off by default for these,
   so the direct rotation estimators are used.

2. The primary method is run with initial guesses read from disk: from the
   secondary results if usable, if not from the tertiary results, and if not
   without initial guess. Angle scan is on by default for the primary.

3. The final table is constructed from the tables written to disk, per exposure:
   primary if it exists, secondary if it exists, tertiary if it exists, or None
   values. An extra column ALIGNMENT_METHOD shows what method has been used.

4. The science difference images that correspond to every reported result in
   the final table are copied to one directory (`FINAL_IMAGES_DIR`).

Every method writes its results and QA to its own sub-directory of the output
directory; the hybrid table is written to the output directory itself.
"""

import pathlib
import shutil

import numpy as np
from astropy.table import Table, Column


# the columns holding the alignment values in the output table
VALUE_COLUMNS = [
    "RA_OFFSET",
    "RA_OFFSET_ERROR",
    "DEC_OFFSET",
    "DEC_OFFSET_ERROR",
    "ROTANGLE",
    "ROTATION_ERROR",
]

# flux normalisation columns (only with --scale-ref), taken from the same
# method as the alignment values (they are fitted with its offsets)
FLUX_COLUMNS = ["FLUX_SCALE", "BACKGROUND"]

# name of the method column in the hybrid table
METHOD_COLUMN = "ALIGNMENT_METHOD"

# directory (inside the output directory) for the difference images of the
# reported results
FINAL_IMAGES_DIR = "final_difference_images"


def read_results_table(results_dir, method_name, logger=None):
    """
    Read the output table written by `AlignmentPipeline.run`.

    Returns the astropy Table, or None if it does not exist.
    """

    filename = pathlib.Path(results_dir) / f"{method_name}_alignment_output_table.fits"

    if not filename.exists():
        if logger:
            logger.warning(f"No output table found for {method_name} at {filename}.")
        return None

    if logger:
        logger.info(f"Reading {method_name} results from {filename}")

    return Table.read(filename, mask_invalid=False)


def merge_results(results, logger=None):
    """
    Construct the hybrid table from the per-method output tables.

    Parameters
    ----------
    results : list
        List of (role, method_name, table) tuples in priority order (primary,
        secondary, tertiary). `table` can be None if the method produced no table.

    Returns
    -------
    merged : astropy.table.Table
        Same columns as the per-method tables, plus `METHOD_COLUMN`. For every
        row, the values are taken from the first method with finite values; if
        none has them, the values are NaN and the method is "None". The
        `FLUX_COLUMNS` (if present) are taken from the same method; rows
        without a result keep those of the first available table.

    roles_used : list
        For every row, the role ("primary", "secondary", "tertiary") the values
        were taken from, or None.
    """

    available = [
        (role, name, table) for role, name, table in results if table is not None
    ]

    if not available:
        raise ValueError("No output tables available to merge.")

    # the first available table gives the header entries (same for all methods)
    _, base_name, base_table = available[0]
    merged = base_table.copy()

    for _, name, table in available[1:]:
        if len(table) != len(base_table):
            raise ValueError(
                f"Output table of {name} has {len(table)} rows, but {base_name} "
                f"has {len(base_table)}. Were they run on the same configuration?"
            )

    methods_used = []
    roles_used = []

    for row in range(len(merged)):

        method_used = "None"
        role_used = None

        for role, name, table in available:

            # sanity check that the rows describe the same exposure
            if table["DATE_OBS"][row] != base_table["DATE_OBS"][row]:
                if logger:
                    logger.warning(
                        f"Row {row}: DATE_OBS of {name} ({table['DATE_OBS'][row]}) does "
                        f"not match {base_name} ({base_table['DATE_OBS'][row]}). Skipping {name}."
                    )
                continue

            values = [table[column][row] for column in VALUE_COLUMNS]

            if np.all(np.isfinite(values)):
                for column, value in zip(VALUE_COLUMNS, values):
                    merged[column][row] = value
                for column in FLUX_COLUMNS:
                    if column in merged.colnames and column in table.colnames:
                        merged[column][row] = table[column][row]
                method_used = name
                role_used = role
                break

        if method_used == "None":
            for column in VALUE_COLUMNS:
                merged[column][row] = np.nan

            if logger:
                logger.warning(
                    f"Row {row} (DATE_OBS {merged['DATE_OBS'][row]}): no method "
                    f"gave a result. Writing None values."
                )

        elif logger:
            logger.info(
                f"Row {row} (DATE_OBS {merged['DATE_OBS'][row]}): using {method_used}."
            )

        methods_used.append(method_used)
        roles_used.append(role_used)

    merged.add_column(Column(np.array(methods_used, dtype="U16"), name=METHOD_COLUMN))

    return merged, roles_used


def collect_difference_images(config, roles_used, pipelines, output_dir, logger=None):
    """
    Copy the science difference image of every reported result in the final
    table to `output_dir`.

    Parameters
    ----------
    config : dict
        The configuration dictionary (gives the row order of the tables).

    roles_used : list
        For every row, the role the result was taken from, or None
        (see `merge_results`).

    pipelines : dict
        The finished AlignmentPipeline objects, per role.

    output_dir : str or pathlib.Path
        Directory to copy the images to. Old images in it are removed first.
    """

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # remove images from earlier runs, so only the current results are in here
    old_images = list(output_dir.glob("*.png"))
    if old_images and logger:
        logger.info(f"Removing {len(old_images)} old images from {output_dir}")
    for old_image in old_images:
        old_image.unlink()

    # rows are written in configuration order, one per exposure
    row_keys = [
        (pointing, i)
        for pointing, exposures in config.items()
        if pointing != "headers"
        for i in range(len(exposures))
    ]

    if len(row_keys) != len(roles_used):
        raise ValueError(
            f"Configuration has {len(row_keys)} exposures, but the final table has "
            f"{len(roles_used)} rows."
        )

    n_copied = 0

    for row, ((pointing, i), role) in enumerate(zip(row_keys, roles_used)):

        if role is None:
            continue

        image = pipelines[role].reported_science_images.get((pointing, i))

        if image is None or not pathlib.Path(image).exists():
            if logger:
                logger.warning(
                    f"Row {row} (pointing {pointing}, dither {i+1}): no difference "
                    f"image found for the reported {role} result ({image})."
                )
            continue

        shutil.copy2(image, output_dir / pathlib.Path(image).name)
        n_copied += 1

    if logger:
        logger.info(f"Copied {n_copied} difference images to {output_dir}")


if __name__ == "__main__":

    import argparse
    import os
    import json
    import time

    # total execution time of the hybrid run, logged as the last line
    hybrid_start_time = time.perf_counter()

    parser = argparse.ArgumentParser(description="Run hybrid aligment pipeline")

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
        "--primary",
        type=str,
        default="FFT",
        help="Primary alignment method. Default is FFT. "
        "Options are NoAlignment, PointSources, FFT, OpticalFlow.",
    )

    parser.add_argument(
        "--secondary",
        type=str,
        default="OpticalFlow",
        help="Secondary alignment method (first source of initial guesses). "
        "Default is OpticalFlow.",
    )

    parser.add_argument(
        "--tertiary",
        type=str,
        default="PointSources",
        help="Tertiary alignment method (second source of initial guesses). "
        "Default is PointSources.",
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
        help="If set, rotation estimation will be skipped for all methods. Default is False.",
    )

    parser.add_argument(
        "--no_primary_angle_scan",
        action="store_true",
        help="If set, the primary method uses its direct rotation estimator instead "
        "of the angle scan. Default is False (angle scan on).",
    )

    parser.add_argument(
        "--secondary_angle_scan",
        action="store_true",
        help="If set, the secondary method uses the angle scan. Default is False.",
    )

    parser.add_argument(
        "--tertiary_angle_scan",
        action="store_true",
        help="If set, the tertiary method uses the angle scan. Default is False.",
    )

    # dither tables: the fast secondary / tertiary methods align all dither
    # pairs, the (slow) primary only the pairs needed to recover failed
    # science alignments
    for role, default in (
        ("primary", "targeted"),
        ("secondary", "full"),
        ("tertiary", "full"),
    ):
        parser.add_argument(
            f"--{role}_dither_mode",
            type=str,
            choices=("full", "targeted"),
            default=default,
            help=f"Dither alignment of the {role} method: 'full' aligns all dither "
            f"pairs, 'targeted' only the pairs needed to recover failed science "
            f"alignments. Default is '{default}'.",
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

    parser.add_argument(
        "--ref_unit",
        type=str,
        default=None,
        help="Unit of the reference image for --scale-ref (e.g. 'uJy'), used instead "
        "of its BUNIT keyword. For references with a wrong BUNIT (e.g. WFI images with "
        "BUNIT = 'mJy' in microJansky, which give FLUX_SCALE ~ 1000). Default: BUNIT.",
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
    from flux import FluxScaleError
    from methods import NoAlignment, PointSources, FFT, OpticalFlow
    from logger import get_logger, format_elapsed, exit_on_sigterm

    # a batch time limit (SIGTERM) is logged as ABORTED instead of silently
    exit_on_sigterm()

    switcher = {
        "NoAlignment": NoAlignment,
        "PointSources": PointSources,
        "FFT": FFT,
        "OpticalFlow": OpticalFlow,
    }

    for role, method_name in (
        ("primary", args.primary),
        ("secondary", args.secondary),
        ("tertiary", args.tertiary),
    ):
        if method_name not in switcher:
            raise ValueError(
                f"Invalid {role} method: {method_name}. Choose from "
                f"'NoAlignment', 'PointSources', 'FFT', 'OpticalFlow'."
            )

    # every method gets its own sub-directory, so QA and tables do not overwrite
    # each other (the role is in the name in case the same method is used twice)
    output_dirs = {
        role: os.path.join(args.output_dir, f"{role}_{method_name}")
        for role, method_name in (
            ("primary", args.primary),
            ("secondary", args.secondary),
            ("tertiary", args.tertiary),
        )
    }

    for output_dir in output_dirs.values():
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # finished pipelines per role, used for the final difference images
    pipelines = {}

    # roles whose run raised an error (see below)
    crashed_roles = []

    # 1) secondary and tertiary: isolated runs, no initial guesses

    for role, method_name, scan_for_angles, dither_mode in (
        (
            "secondary",
            args.secondary,
            args.secondary_angle_scan,
            args.secondary_dither_mode,
        ),
        (
            "tertiary",
            args.tertiary,
            args.tertiary_angle_scan,
            args.tertiary_dither_mode,
        ),
    ):
        pipeline = AlignmentPipeline(
            config_dict=config,
            input_dir=args.input_dir,
            reference_file=args.ref_file_path,
            output_dir=output_dirs[role],
            aligment_method=switcher[method_name],
            distance_threshold=args.distance_threshold,
            skip_rotation_estimation=args.skip_rotation_estimation,
            scan_for_angles=scan_for_angles,
            dither_mode=dither_mode,
            scale_ref=args.scale_ref,
            ref_filter=args.ref_filter,
            ref_unit=args.ref_unit,
        )

        pipeline.logger.info(
            f"Hybrid pipeline: running {role} method {method_name} "
            f"({dither_mode} dither table)."
        )

        # a crashed method must not stop the hybrid run: the next method and
        # the final table only lose its results (pipeline.run already logged
        # the traceback as ABORTED)
        try:
            pipeline.run()
        except FluxScaleError as error:
            # the other methods would stop the same way
            pipeline.logger.error(
                "Hybrid pipeline: stopped, the flux scale of the reference is wrong."
            )
            raise SystemExit(str(error)) from None
        except Exception:
            crashed_roles.append(role)
            pipeline.logger.error(
                f"Hybrid pipeline: the {role} method {method_name} crashed, "
                f"continuing without its (remaining) results."
            )

        pipelines[role] = pipeline

    # 2) primary: initial guesses from secondary, then tertiary (read from disk)

    pipeline = AlignmentPipeline(
        config_dict=config,
        input_dir=args.input_dir,
        reference_file=args.ref_file_path,
        output_dir=output_dirs["primary"],
        aligment_method=switcher[args.primary],
        distance_threshold=args.distance_threshold,
        skip_rotation_estimation=args.skip_rotation_estimation,
        scan_for_angles=not args.no_primary_angle_scan,
        initial_guess_sources=[
            (output_dirs["secondary"], args.secondary),
            (output_dirs["tertiary"], args.tertiary),
        ],
        dither_mode=args.primary_dither_mode,
        scale_ref=args.scale_ref,
        ref_filter=args.ref_filter,
        ref_unit=args.ref_unit,
    )

    pipeline.logger.info(
        f"Hybrid pipeline: running primary method {args.primary} with initial "
        f"guesses from {args.secondary}, then {args.tertiary} "
        f"({args.primary_dither_mode} dither table)."
    )

    try:
        pipeline.run()
    except FluxScaleError as error:
        pipeline.logger.error(
            "Hybrid pipeline: stopped, the flux scale of the reference is wrong."
        )
        raise SystemExit(str(error)) from None
    except Exception:
        crashed_roles.append("primary")
        pipeline.logger.error(
            f"Hybrid pipeline: the primary method {args.primary} crashed, "
            f"constructing the final table from the other methods."
        )

    pipelines["primary"] = pipeline

    # 3) construct the final table: primary, else secondary, else tertiary, else None

    logger = get_logger(args.output_dir)

    logger.info("Hybrid pipeline: constructing final results table...")

    # a crashed run may have left no table, or an old one from an earlier run
    # in the same directory: its table is not used
    for role in crashed_roles:
        logger.warning(
            f"Hybrid pipeline: not using the {role} results (the run crashed)."
        )

    results = [
        (
            role,
            method_name,
            (
                None
                if role in crashed_roles
                else read_results_table(output_dirs[role], method_name, logger=logger)
            ),
        )
        for role, method_name in (
            ("primary", args.primary),
            ("secondary", args.secondary),
            ("tertiary", args.tertiary),
        )
    ]

    merged, roles_used = merge_results(results, logger=logger)

    output_filename = os.path.join(
        args.output_dir, "Hybrid_alignment_output_table.fits"
    )

    merged.write(output_filename, format="fits", overwrite=True)

    logger.info(f"Successfully wrote hybrid results to {output_filename}")

    # 4) the difference images of the reported results, all in one directory
    #    (the table is already written, so a problem here only costs the copies)

    try:
        collect_difference_images(
            config,
            roles_used,
            pipelines,
            os.path.join(args.output_dir, FINAL_IMAGES_DIR),
            logger=logger,
        )
    except Exception:
        logger.exception(
            "Hybrid pipeline: copying the final difference images failed (the "
            "hybrid table is written)."
        )

    n_rows = len(merged)
    usage = {
        name: int(np.sum(merged[METHOD_COLUMN] == name))
        for name in dict.fromkeys(list(merged[METHOD_COLUMN]))
    }
    logger.info(
        f"Hybrid summary: {n_rows} exposures, results from "
        + ", ".join(f"{name}: {count}" for name, count in usage.items())
    )
    if crashed_roles:
        logger.warning(
            f"Hybrid summary: the {', '.join(crashed_roles)} run(s) crashed, see "
            f"ABORTED in their logs."
        )

    logger.info(
        f"Hybrid pipeline finished, total execution time (all three runs): "
        f"{format_elapsed(time.perf_counter() - hybrid_start_time)}"
    )
