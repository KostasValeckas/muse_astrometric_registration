"""
Replot the final difference images of a finished run with the flux
normalisation of ``--scale-ref``, without running the alignment again.

For every exposure with a result in the output table of the run (by default
``Hybrid_alignment_output_table.fits``), the frames are prepared as in the
pipeline (all exposures of the pointing reprojected to the frame of the first
one, the common NaN edges cropped, the reference reprojected to that frame,
converted to the MUSE unit and multiplied by the pixel area ratio, see
``flux.py``), the offsets of the table are converted back to pixels, the
normalisation ``reference = FLUX_SCALE * (exposure + BACKGROUND)`` is fitted
with the exposure shifted by these offsets, and the difference image is drawn
with the reference on the flux scale of the exposure (as the final difference
images of a ``--scale-ref`` run).

Written to ``--images_dir`` (default ``<output_dir>/final_difference_images_flux``):

- ``difference_image_{pointing}_dither_{i}_final_{method}.png`` per exposure,
- ``{table}_alignment_output_table_flux.fits``: the output table of the run
  with the columns FLUX_SCALE and BACKGROUND added (NaN without a result),
- the log.

The run itself is not changed. Example::

    python dev/replot_final_difference_images.py results/NGC0628 NGC0628_sorted.json
"""

import json
import os
import pathlib
import sys

import numpy as np

# the pipeline modules: next to this script (zip), or in ../src (repository)
_SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# rows without a result in the hybrid table have this method name
NO_RESULT = "None"

# name of the method column in the hybrid table (see hybrid_pipeline.py)
METHOD_COLUMN = "ALIGNMENT_METHOD"


def find_run_manifest(output_dir):
    """
    Read the run manifest (reference file and input directory of the run) from
    `output_dir`, or from one of its sub-directories (hybrid run).

    Returns the manifest as a dict, empty if there is none.
    """

    output_dir = pathlib.Path(output_dir)
    candidates = [output_dir / "run_manifest.json"] + sorted(
        output_dir.glob("*/run_manifest.json")
    )

    for candidate in candidates:
        if candidate.exists():
            try:
                with open(candidate) as f:
                    return json.load(f)
            except (OSError, ValueError):
                continue

    return {}


def prepare_pointing(files, reference_frame, reference_wcs, logger=None):
    """
    Prepare the exposures of a pointing and the reference as the pipeline does
    for the science alignment (see ``AlignmentPipeline._run``), with the flux
    scaling of ``--scale-ref`` before the alignment (pixel area ratio; the
    unit conversion is already in `reference_frame`).

    Parameters
    ----------
    files : list
        Paths of the exposures of the pointing, in configuration order.

    reference_frame : np.2darray
        The reference, converted to the MUSE unit.

    reference_wcs : astropy.wcs.WCS
        WCS of the reference.

    Returns
    -------
    exposures : list of np.2darray
        The prepared exposures.

    reference : np.2darray
        The reference in their frame.

    wcs : astropy.wcs.WCS
        WCS of the common frame (to convert the offsets of the table).
    """

    from IO import read_fits_image, extract_image_wcs
    from geometry import (
        reproject_dithers,
        remove_nan_edges_dithers,
        reproject_image_to_target_wcs,
    )
    from flux import pixel_area_factor

    arrays = []
    wcs_set = []

    for path in files:
        wcs = extract_image_wcs(path, logger=logger, wcs_extension=1)
        read = read_fits_image(path, logger=logger, data_extension=1)

        if wcs is None or read is None:
            raise ValueError(f"Could not read the image or the WCS of {path}.")

        arrays.append(read[0])
        wcs_set.append(wcs)

    arrays, wcs_set = reproject_dithers(
        arrays, wcs_set, logger=logger, method="interpolation"
    )

    original_shape = arrays[0].shape

    arrays, edge_cut_value = remove_nan_edges_dithers(arrays, logger=logger)

    reference, _ = reproject_image_to_target_wcs(
        reference_frame, reference_wcs, wcs_set[0], shape_out=original_shape
    )

    reference = reference * pixel_area_factor(wcs_set[0], reference_wcs)

    reference = reference[
        edge_cut_value : reference.shape[0] - edge_cut_value,
        edge_cut_value : reference.shape[1] - edge_cut_value,
    ]

    return arrays, reference, wcs_set[0]


if __name__ == "__main__":

    import argparse
    import time

    start_time = time.perf_counter()

    parser = argparse.ArgumentParser(
        description="Replot the final difference images of a finished run with the "
        "flux normalisation of --scale-ref."
    )

    parser.add_argument(
        "output_dir",
        type=str,
        help="Output directory of the run (hybrid or single method).",
    )

    parser.add_argument(
        "config_file",
        type=str,
        help="Configuration file of the run (gives the exposures of every pointing, "
        "in the order of the rows of the output table).",
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Directory of the exposures. Default: the input directory of the run "
        "(run manifest).",
    )

    parser.add_argument(
        "--ref_file_path",
        type=str,
        default=None,
        help="Path of the reference file. Default: the reference file of the run "
        "(run manifest).",
    )

    parser.add_argument(
        "--table",
        type=str,
        default="Hybrid",
        help="Method name of the output table to replot "
        "(<table>_alignment_output_table.fits). Default is 'Hybrid'.",
    )

    parser.add_argument(
        "--ref_filter",
        type=str,
        default="WFI_BB",
        help="Filter of the reference, for the unit conversion (see "
        "REFERENCE_FILTERS in flux.py). Default is 'WFI_BB'.",
    )

    parser.add_argument(
        "--ref_unit",
        type=str,
        default=None,
        help="Unit of the reference (e.g. 'uJy'), used instead of its BUNIT keyword "
        "(for references with a wrong BUNIT). Default: BUNIT.",
    )

    parser.add_argument(
        "--images_dir",
        type=str,
        default=None,
        help="Directory to write to. Default: <output_dir>/final_difference_images_flux.",
    )

    args = parser.parse_args()

    # no display on the cluster
    import matplotlib

    matplotlib.use("Agg")

    # imports happen here so the matplotlib backend is set before
    from astropy.table import Table

    from logger import get_logger, log_run_environment, format_elapsed
    from IO import read_fits_image, extract_image_wcs
    from geometry import sky_translation_to_detector_translation
    from flux import (
        reference_conversion_factor,
        fit_flux_normalisation,
        shift_image_with_nans,
        check_flux_scales,
    )
    from methods import NoAlignment

    images_dir = pathlib.Path(
        args.images_dir
        if args.images_dir is not None
        else os.path.join(args.output_dir, "final_difference_images_flux")
    )
    images_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(images_dir)
    log_run_environment(logger)

    with open(args.config_file, "r") as f:
        config = json.load(f)

    manifest = find_run_manifest(args.output_dir)
    input_dir = args.input_dir or manifest.get("input_dir")
    reference_file = args.ref_file_path or manifest.get("reference_file")

    if input_dir is None or reference_file is None:
        raise ValueError(
            "No run manifest found: give --input_dir and --ref_file_path."
        )

    table_file = (
        pathlib.Path(args.output_dir) / f"{args.table}_alignment_output_table.fits"
    )
    table = Table.read(table_file, mask_invalid=False)

    logger.info(f"Replotting the final difference images of {table_file}")
    logger.info(f"Configuration: {args.config_file}")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Reference file: {reference_file}")
    logger.info(f"Writing to {images_dir}")

    # rows are written in configuration order, one per exposure
    row_keys = [
        (pointing, i)
        for pointing, exposures in config.items()
        if pointing != "headers"
        for i in range(len(exposures))
    ]

    if len(row_keys) != len(table):
        raise ValueError(
            f"{table_file} has {len(table)} rows, but {args.config_file} has "
            f"{len(row_keys)} exposures. Was the run made with this configuration?"
        )

    # the reference, converted to the MUSE unit (as --scale-ref before the
    # alignment)
    reference_wcs = extract_image_wcs(reference_file, logger=logger)
    read = read_fits_image(reference_file, logger=logger)
    if reference_wcs is None or read is None:
        raise ValueError(f"Could not read the reference {reference_file}.")
    reference_data, reference_hdul = read
    reference_data = reference_data * reference_conversion_factor(
        args.ref_filter,
        header=reference_hdul[0].header,
        logger=logger,
        unit=args.ref_unit,
    )

    flux_scale = np.full(len(table), np.nan)
    background = np.full(len(table), np.nan)

    plotter = NoAlignment(QA_path=images_dir)

    n_plotted = 0
    n_no_result = 0
    failed_pointings = []

    row = 0
    for pointing, exposures in config.items():

        if pointing == "headers":
            continue

        rows = list(range(row, row + len(exposures)))
        row += len(exposures)

        has_result = [
            bool(
                np.all(
                    np.isfinite(
                        [
                            table["RA_OFFSET"][r],
                            table["DEC_OFFSET"][r],
                            table["ROTANGLE"][r],
                        ]
                    )
                )
            )
            for r in rows
        ]
        n_no_result += len(rows) - sum(has_result)

        if not any(has_result):
            logger.warning(f"Pointing {pointing}: no exposure has a result, skipping.")
            continue

        # a problem in one pointing (e.g. an unreadable file) must not stop
        # the others
        try:
            files = [os.path.join(input_dir, exposure[2]) for exposure in exposures]
            arrays, reference, wcs = prepare_pointing(
                files, reference_data, reference_wcs, logger=logger
            )

            for i, r in enumerate(rows):

                if not has_result[i]:
                    logger.info(
                        f"Pointing {pointing}, dither {i+1}: no result, no image."
                    )
                    continue

                date_obs = str(table["DATE_OBS"][r]).strip()
                if date_obs != str(exposures[i][1]).strip():
                    logger.warning(
                        f"Pointing {pointing}, dither {i+1}: DATE_OBS {date_obs} of "
                        f"the table does not match {exposures[i][1]} of the "
                        f"configuration."
                    )

                tx, ty = sky_translation_to_detector_translation(
                    wcs, float(table["RA_OFFSET"][r]), float(table["DEC_OFFSET"][r])
                )
                rotation_angle = float(table["ROTANGLE"][r])

                flux_scale[r], background[r] = fit_flux_normalisation(
                    shift_image_with_nans(arrays[i], tx, ty, rotation_angle),
                    reference,
                    logger=logger,
                    name=f"(pointing {pointing}, dither {i+1})",
                )

                flux = (
                    (flux_scale[r], background[r])
                    if np.isfinite(flux_scale[r])
                    else None
                )
                if flux is None:
                    logger.warning(
                        f"Pointing {pointing}, dither {i+1}: flux normalisation not "
                        f"usable, plotting without it."
                    )

                method = (
                    str(table[METHOD_COLUMN][r]).strip()
                    if METHOD_COLUMN in table.colnames
                    else args.table
                )
                plotter.name = method

                plotter.plot_difference_image(
                    reference,
                    arrays[i],
                    tx,
                    ty,
                    rotation_angle,
                    f"{pointing}_dither_{i+1}_final",
                    show=False,
                    custom_title=(
                        f"Final alignment of pointing {pointing}, dither {i+1} "
                        f"({exposures[i][2]})"
                    ),
                    flux_normalisation=flux,
                )
                n_plotted += 1

        except Exception:
            logger.exception(
                f"Unexpected error in pointing {pointing}, continuing with the next."
            )
            failed_pointings.append(pointing)

    check_flux_scales(
        flux_scale,
        logger=logger,
        unit_source="--ref_unit" if args.ref_unit is not None else "its BUNIT",
    )

    table["FLUX_SCALE"] = flux_scale
    table["BACKGROUND"] = background
    flux_table_file = images_dir / f"{args.table}_alignment_output_table_flux.fits"
    table.write(flux_table_file, format="fits", overwrite=True)
    logger.info(f"Wrote {flux_table_file}")

    finite = np.isfinite(flux_scale)
    logger.info(
        f"Summary: {n_plotted} images, {n_no_result} exposures without a result, "
        f"{int(np.sum(finite))} flux normalisations"
        + (
            f" (FLUX_SCALE median {np.median(flux_scale[finite]):.4g}, range "
            f"{np.min(flux_scale[finite]):.4g} - {np.max(flux_scale[finite]):.4g}; "
            f"BACKGROUND median {np.median(background[finite]):.4g})"
            if np.any(finite)
            else ""
        )
    )
    if failed_pointings:
        logger.warning(
            f"Pointings with an unexpected error (no images): "
            f"{', '.join(failed_pointings)}"
        )
    logger.info(
        f"Replotting finished in {format_elapsed(time.perf_counter() - start_time)}"
    )
