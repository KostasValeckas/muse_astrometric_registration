"""
Manual alignment of one exposure to the reference, set up from the output of
a pipeline run.

The exposure and the reference are prepared exactly as the pipeline does it
(all exposures of the pointing reprojected to the frame of the first one, the
common NaN edges cropped, the reference reprojected to that frame), so the
offsets found here are in the same convention as the output table:
``shift_image(exposure, tx, ty, rot)`` matches the reference, and (tx, ty) is
converted to RA / Dec offsets with the WCS of that frame.

The initial guess is taken, in this order, from ``--guess``, the row of the
exposure in the output table of the run, the median of the other exposures of
the same pointing in that table, or zero.

In the window the offsets can be changed by hand (keys or text boxes), refined
with one of the alignment methods (the current values are its initial guess,
optionally checked with the reverse alignment as in the pipeline), and saved.
Saving writes, in ``<output_dir>/manual_alignment``:

- ``Manual_<exposure>_alignment_output_table.fits`` (and the pymusepipe offset
  table) with the one exposure,
- ``Manual_alignment_output_table.fits``: the output table of the run with the
  row of this exposure replaced (kept between sessions, so several exposures
  can be aligned one after the other), and the pymusepipe offset table made
  from it,
- the difference image of the saved result (the QA plots of the refinements
  are in ``refine_QA``).

Example::

    python src/manual_alignment.py IMAGE_FOV_..._0003.fits \\
        --output_dir results/NGC4535 --config_file NGC4535_sorted.json \\
        --input_dir /data/NGC4535/Alignment --ref_file_path ref.fits
"""

import json
import os
import pathlib

import numpy as np

# name of the method column in the hybrid table (see hybrid_pipeline.py)
METHOD_COLUMN = "ALIGNMENT_METHOD"

# directory (inside the output directory of the run) for the results
MANUAL_DIR = "manual_alignment"

# method name of a result set by hand ("Manual+<method>" if refined)
MANUAL_METHOD_NAME = "Manual"

# output tables looked for in the output directory, in this order
RESULT_TABLE_NAMES = ("Hybrid", "FFT", "OpticalFlow", "PointSources", "NoAlignment")

# columns of the output table with the alignment values
VALUE_COLUMNS = (
    "RA_OFFSET",
    "RA_OFFSET_ERROR",
    "DEC_OFFSET",
    "DEC_OFFSET_ERROR",
    "ROTANGLE",
    "ROTATION_ERROR",
)

# round-trip distance (pixels) for a consistent refinement, the default
# distance_threshold of the pipeline
DISTANCE_THRESHOLD = 0.10

# default step sizes of the keyboard controls
TRANSLATION_STEP = 0.5  # pixels
ROTATION_STEP = 0.05  # degrees

# percentiles used to scale the frames for display
DISPLAY_PERCENTILES = (5, 99.5)


# ============================================================================
# Set-up from the output of the pipeline run
# ============================================================================


def find_results_table(output_dir, table_name=None, logger=None):
    """
    Find the output table of a pipeline run.

    Parameters
    ----------
    output_dir : str or pathlib.Path
        Output directory of the run (hybrid or single method).

    table_name : str, optional
        Method name of the table (``<table_name>_alignment_output_table.fits``).
        If None, the names in ``RESULT_TABLE_NAMES`` are tried in order.

    Returns
    -------
    name : str
        Method name of the table.

    table : astropy.table.Table
        The output table.
    """

    from astropy.table import Table

    names = (table_name,) if table_name else RESULT_TABLE_NAMES

    for name in names:
        filename = pathlib.Path(output_dir) / f"{name}_alignment_output_table.fits"
        if filename.exists():
            if logger:
                logger.info(f"Using output table {filename}")
            return name, Table.read(filename, mask_invalid=False)

    raise FileNotFoundError(
        f"No output table ({', '.join(f'{n}_alignment_output_table.fits' for n in names)}) "
        f"found in {output_dir}."
    )


def find_run_manifest(output_dir):
    """
    Read the run manifest (reference file and input directory of the run) from
    `output_dir`, or from one of its sub-directories (hybrid run).

    Returns the manifest as a dict, empty if there is none.
    """

    from pipeline import RUN_MANIFEST_NAME

    output_dir = pathlib.Path(output_dir)
    candidates = [output_dir / RUN_MANIFEST_NAME] + sorted(
        output_dir.glob(f"*/{RUN_MANIFEST_NAME}")
    )

    for candidate in candidates:
        if candidate.exists():
            try:
                with open(candidate) as f:
                    return json.load(f)
            except (OSError, ValueError):
                continue

    return {}


def find_exposure(config_dict, filename):
    """
    Find an exposure in the configuration by its file name.

    Returns (pointing, index of the exposure in the pointing, row index in the
    output table), or (None, None, None) if it is not in the configuration.
    """

    name = os.path.basename(str(filename))
    row = 0

    for pointing, exposures in config_dict.items():
        if pointing == "headers":
            continue
        for i, exposure in enumerate(exposures):
            if os.path.basename(str(exposure[2])) == name:
                return pointing, i, row + i
        row += len(exposures)

    return None, None, None


def find_table_row(table, date_obs, logger=None):
    """
    Row of the exposure with `date_obs` in the output table, or None.
    """

    rows = [
        k
        for k, value in enumerate(table["DATE_OBS"])
        if str(value).strip() == str(date_obs).strip()
    ]

    if not rows:
        return None

    if len(rows) > 1 and logger:
        logger.warning(
            f"DATE_OBS {date_obs} appears in {len(rows)} rows of the output table, "
            f"using the first one (row {rows[0]})."
        )

    return rows[0]


def prepare_frames(file_path, pointing_files, index, reference_file, logger=None):
    """
    Prepare the exposure and the reference as the pipeline does for the
    science alignment (see ``AlignmentPipeline._run``).

    Parameters
    ----------
    file_path : str or pathlib.Path
        The exposure to align.

    pointing_files : list
        All exposures of its pointing, in configuration order (the exposure
        itself at `index`). If only the exposure is given, its own frame is
        used.

    index : int
        Index of the exposure in `pointing_files`.

    reference_file : str
        Path of the reference image.

    Returns
    -------
    dict
        "reference" and "exposure" (2D arrays of the same shape), "wcs" (WCS
        of the common frame, used for the sky offsets), "hdul" (of the
        exposure, for the header entries of the output table).
    """

    from IO import read_fits_image, extract_image_wcs
    from geometry import (
        reproject_dithers,
        remove_nan_edges_dithers,
        reproject_image_to_target_wcs,
    )

    arrays = []
    wcs_set = []
    hdul = None

    for k, path in enumerate(pointing_files):
        wcs = extract_image_wcs(path, logger=logger, wcs_extension=1)
        read = read_fits_image(path, logger=logger, data_extension=1)

        if wcs is None or read is None:
            raise ValueError(f"Could not read the image or the WCS of {path}.")

        arrays.append(read[0])
        wcs_set.append(wcs)
        if k == index:
            hdul = read[1]

    # same frame and shape for all exposures of the pointing
    arrays, wcs_set = reproject_dithers(
        arrays, wcs_set, logger=logger, method="interpolation"
    )

    original_shape = arrays[0].shape

    arrays, edge_cut_value = remove_nan_edges_dithers(arrays, logger=logger)

    reference_wcs = extract_image_wcs(reference_file, logger=logger)
    read = read_fits_image(reference_file, logger=logger)

    if reference_wcs is None or read is None:
        raise ValueError(f"Could not read the reference {reference_file}.")

    reference, _ = reproject_image_to_target_wcs(
        read[0], reference_wcs, wcs_set[0], shape_out=original_shape
    )

    reference = reference[
        edge_cut_value : reference.shape[0] - edge_cut_value,
        edge_cut_value : reference.shape[1] - edge_cut_value,
    ]

    exposure = arrays[index]

    if not np.any(np.isfinite(exposure)):
        raise ValueError(
            f"{file_path} has no finite pixels after the preparation, it cannot be "
            f"aligned."
        )

    return dict(reference=reference, exposure=exposure, wcs=wcs_set[0], hdul=hdul)


def _row_to_pixels(table, row, wcs):
    """(tx, ty, rot) of a row of the output table in pixels of `wcs`, or None if it has no result."""

    from geometry import sky_translation_to_detector_translation

    dRA, dDEC, rot = (
        float(table["RA_OFFSET"][row]),
        float(table["DEC_OFFSET"][row]),
        float(table["ROTANGLE"][row]),
    )

    if not np.all(np.isfinite([dRA, dDEC, rot])):
        return None

    tx, ty = sky_translation_to_detector_translation(wcs, dRA, dDEC)

    return tx, ty, rot


def initial_guess(table, table_name, row, pointing_rows, wcs, guess=None, logger=None):
    """
    Initial guess (tx, ty, rot) for the manual alignment, see the module
    docstring for the order of the sources.

    Parameters
    ----------
    table : astropy.table.Table
        The output table of the run.

    table_name : str
        Its method name (for the log).

    row : int or None
        Row of the exposure in the table.

    pointing_rows : list
        Rows of the other exposures of the same pointing.

    wcs : astropy.wcs.WCS
        WCS of the common frame of the pointing.

    guess : tuple, optional
        (tx, ty, rot) given by the user, used if not None.

    Returns
    -------
    (tx, ty, rot), source : the guess and a description of where it came from.
    """

    if guess is not None:
        return tuple(float(v) for v in guess), "given on the command line"

    if row is not None:
        values = _row_to_pixels(table, row, wcs)
        if values is not None:
            method = (
                str(table[METHOD_COLUMN][row]).strip()
                if METHOD_COLUMN in table.colnames
                else table_name
            )
            return values, f"{table_name} output table ({method})"

    others = [_row_to_pixels(table, r, wcs) for r in pointing_rows if r != row]
    others = [values for values in others if values is not None]

    if others:
        values = tuple(float(v) for v in np.median(np.array(others), axis=0))
        if logger:
            logger.warning(
                f"No result for this exposure in the {table_name} output table: "
                f"using the median of {len(others)} other exposure(s) of the "
                f"pointing as initial guess. This is only a rough guess."
            )
        return values, f"median of {len(others)} other exposure(s) of the pointing"

    if logger:
        logger.warning(
            "No result for this exposure or its pointing in the output table: "
            "starting from zero offsets."
        )

    return (0.0, 0.0, 0.0), "none (zero)"


# ============================================================================
# Refinement with an alignment method
# ============================================================================


def refine_alignment(
    method_name,
    reference,
    exposure,
    wcs,
    guess,
    obs_name,
    QA_path,
    verify=True,
    logger=None,
):
    """
    Refine the alignment with one of the alignment methods, starting from
    `guess`, and optionally check it with the reverse alignment, as
    ``AlignmentPipeline._science_forward_reverse`` does.

    Returns a dict with "tx", "ty", "rot", "tx_error", "ty_error",
    "rot_error", "method" and "roundtrip" (round-trip distance in pixels, NaN
    if not checked or the reverse failed), or None if the method failed.
    """

    from methods import PointSources, FFT, OpticalFlow
    from geometry import (
        transformation_matrix,
        inverse_transformation,
        matrix_to_parameters,
    )
    from pipeline import _image_centre, _roundtrip_distance, _check_points

    switcher = {"FFT": FFT, "OpticalFlow": OpticalFlow, "PointSources": PointSources}

    aligner = switcher[method_name](QA_path=QA_path, logger=logger)

    tx, ty, rot = guess

    (
        (ftx, fty),
        frot,
        (ftx_error, fty_error),
        frot_error,
        success,
    ) = aligner.run(
        reference,
        exposure,
        f"{obs_name}_to_reference",
        input_wcs=wcs,
        reference_wcs=wcs,
        initial_guess_translation=(tx, ty),
        initial_guess_rotation=rot,
    )

    if success != 1:
        if logger:
            logger.warning(f"{method_name} refinement of {obs_name} failed.")
        return None

    result = dict(
        tx=float(ftx),
        ty=float(fty),
        rot=float(frot),
        tx_error=float(ftx_error),
        ty_error=float(fty_error),
        rot_error=float(frot_error),
        method=method_name,
        roundtrip=np.nan,
    )

    if verify:
        # the inverse of a rotation about the centre is not (-tx, -ty, -rot)
        centre = _image_centre(exposure.shape)
        rtx, rty, rrot = matrix_to_parameters(
            inverse_transformation(transformation_matrix(ftx, fty, frot, centre)),
            centre,
        )

        (btx, bty), brot, _, _, success_rev = aligner.run(
            exposure,
            reference,
            f"reference_to_{obs_name}",
            input_wcs=wcs,
            reference_wcs=wcs,
            initial_guess_translation=(rtx, rty),
            initial_guess_rotation=rrot,
        )

        if success_rev == 1:
            result["roundtrip"] = _roundtrip_distance(
                (ftx, fty, frot), (btx, bty, brot), _check_points(exposure.shape)
            )
            if not result["roundtrip"] <= DISTANCE_THRESHOLD and logger:
                logger.warning(
                    f"{method_name} refinement of {obs_name} is inconsistent: "
                    f"round-trip distance {result['roundtrip']:.3f} px > "
                    f"{DISTANCE_THRESHOLD} px (the pipeline would not accept it)."
                )
        elif logger:
            logger.warning(
                f"Reverse {method_name} alignment of {obs_name} failed, the "
                f"refinement cannot be verified."
            )

    if logger:
        logger.info(
            f"{method_name} refinement of {obs_name}: tx = {result['tx']:.3f}, "
            f"ty = {result['ty']:.3f}, rot = {result['rot']:.4f}, round-trip "
            f"distance {result['roundtrip']:.3f} px"
        )

    return result


# ============================================================================
# Saving
# ============================================================================


def save_alignment(
    state,
    prepared,
    table,
    table_name,
    row,
    obs_name,
    manual_dir,
    header_entries,
    pymusepipe_flux=None,
    logger=None,
):
    """
    Write the result of the manual alignment (see the module docstring).

    Parameters
    ----------
    state : dict
        "tx", "ty", "rot" (pixels, degrees), "tx_error", "ty_error",
        "rot_error" (NaN if set by hand) and "method".

    prepared : dict
        Output of ``prepare_frames``.

    table : astropy.table.Table
        The output table of the run.

    table_name : str
        Its method name.

    row : int or None
        Row of the exposure in the table (None: only the single-exposure
        table is written).

    obs_name : str
        Name of the exposure for the file names.

    manual_dir : pathlib.Path
        Directory to write to.

    header_entries : list
        Header keywords of the identification columns (as in the pipeline).

    pymusepipe_flux : dict, optional
        Passed to ``IO.write_pymusepipe_offset_table`` for the table of all
        exposures (FLUX_SCALE and BACKGROUND from pymusepipe).

    Returns
    -------
    str
        Short description of what was written.
    """

    from astropy.table import Table
    from IO import (
        make_table_entry,
        write_final_results,
        write_pymusepipe_offset_table,
    )
    from geometry import (
        detector_translation_to_sky_translation,
        detector_errors_to_sky_errors,
    )
    from methods import NoAlignment

    wcs = prepared["wcs"]

    dRA, dDEC = detector_translation_to_sky_translation(
        wcs, state["tx"], state["ty"], logger=logger
    )

    if np.isfinite(state["tx_error"]) and np.isfinite(state["ty_error"]):
        dRA_sigma, dDEC_sigma = detector_errors_to_sky_errors(
            wcs, state["tx_error"], state["ty_error"]
        )
    else:
        dRA_sigma = dDEC_sigma = np.nan

    values = dict(
        RA_OFFSET=dRA,
        RA_OFFSET_ERROR=dRA_sigma,
        DEC_OFFSET=dDEC,
        DEC_OFFSET_ERROR=dDEC_sigma,
        ROTANGLE=state["rot"],
        ROTATION_ERROR=state["rot_error"],
    )

    manual_dir.mkdir(parents=True, exist_ok=True)

    # 1) table of this exposure alone (identification columns from the header,
    #    as in the pipeline)
    table_entry = make_table_entry(
        prepared["hdul"],
        dRA,
        dRA_sigma,
        dDEC,
        dDEC_sigma,
        state["rot"],
        state["rot_error"],
        output_table_entries=header_entries,
        logger=logger,
    )

    write_final_results(
        [tuple(value for _, value in table_entry.items())],
        method_name=f"{MANUAL_METHOD_NAME}_{obs_name}",
        path=str(manual_dir),
        logger=logger,
    )

    # 2) the output table of the run with this row replaced; earlier manual
    #    results are kept if the table matches the run
    written = [f"Manual_{obs_name}_alignment_output_table.fits"]

    if row is not None:

        merged_file = manual_dir / f"{MANUAL_METHOD_NAME}_alignment_output_table.fits"

        merged = None
        if merged_file.exists():
            previous = Table.read(merged_file, mask_invalid=False)
            if len(previous) == len(table) and all(
                str(a).strip() == str(b).strip()
                for a, b in zip(previous["DATE_OBS"], table["DATE_OBS"])
            ):
                merged = previous
                if logger:
                    logger.info(f"Adding to the manual results in {merged_file}")
            elif logger:
                logger.warning(
                    f"{merged_file} does not match the output table of the run, "
                    f"starting again from the output table."
                )

        if merged is None:
            merged = table.copy()
            if METHOD_COLUMN not in merged.colnames:
                has_result = [
                    bool(np.all(np.isfinite([merged[c][k] for c in VALUE_COLUMNS])))
                    for k in range(len(merged))
                ]
                merged[METHOD_COLUMN] = np.where(has_result, table_name, "None")

        # room for "Manual+PointSources"
        merged[METHOD_COLUMN] = np.asarray(merged[METHOD_COLUMN], dtype="U24")

        for column, value in values.items():
            merged[column][row] = value
        merged[METHOD_COLUMN][row] = state["method"]

        merged.write(merged_file, format="fits", overwrite=True)

        if logger:
            n_manual = sum(
                str(m).startswith(MANUAL_METHOD_NAME) for m in merged[METHOD_COLUMN]
            )
            logger.info(
                f"Wrote {merged_file} ({n_manual} manual result(s) in the table)"
            )

        write_pymusepipe_offset_table(
            merged,
            str(manual_dir / f"{MANUAL_METHOD_NAME}_pymusepipe_offset_table.fits"),
            pymusepipe_flux=pymusepipe_flux,
            logger=logger,
        )

        written.append(merged_file.name)

    # 3) difference image of the saved result, as in the pipeline QA
    qa = NoAlignment(QA_path=manual_dir)
    qa.name = state["method"]
    qa.plot_difference_image(
        prepared["reference"],
        prepared["exposure"],
        state["tx"],
        state["ty"],
        state["rot"],
        obs_name,
        custom_title=f"{obs_name}: {state['method']} alignment",
    )
    written.append(f"difference_image_{obs_name}_{state['method']}.png")

    if logger:
        logger.info(
            f"Saved the {state['method']} alignment of {obs_name}: tx = "
            f"{state['tx']:.3f}, ty = {state['ty']:.3f}, rot = {state['rot']:.4f} "
            f"(RA_OFFSET {dRA * 3600:.4f} arcsec, DEC_OFFSET {dDEC * 3600:.4f} arcsec) "
            f"to {manual_dir}"
        )

    return ", ".join(written)


# ============================================================================
# Interactive window
# ============================================================================


def _normalise(image):
    """Scale an image to roughly [0, 1] with the ``DISPLAY_PERCENTILES`` (NaN kept)."""
    lo, hi = np.nanpercentile(image, DISPLAY_PERCENTILES)
    return (image - lo) / (hi - lo if hi > lo else 1.0)


class ManualAlignmentWindow:
    """
    Interactive matplotlib window for the manual alignment of one exposure.

    Panels: overlay (red = reference, cyan = shifted exposure, white where
    they agree), difference reference - shifted exposure, and contours of
    both. The three panels share the zoom.

    Keys: arrows move the exposure by the translation step, ',' and '.'
    rotate it by the rotation step, '+' and '-' double / halve both steps.
    The text boxes set the values directly. "Refine" runs the selected method
    from the current values, "Reset" goes back to the initial guess, "Save"
    writes the current values.
    """

    def __init__(
        self,
        reference,
        exposure,
        guess,
        guess_source,
        obs_name,
        refine_fn,
        save_fn,
        methods=("FFT", "OpticalFlow", "PointSources"),
        translation_step=TRANSLATION_STEP,
        rotation_step=ROTATION_STEP,
        logger=None,
    ):
        """
        Set up the window.

        Parameters
        ----------
        reference, exposure : np.2darray
            The prepared frames (``prepare_frames``).

        guess : tuple
            Initial (tx, ty, rot).

        guess_source : str
            Where the guess came from (shown in the window).

        obs_name : str
            Name of the exposure.

        refine_fn : callable
            refine_fn(method_name, (tx, ty, rot), verify) -> dict or None,
            see ``refine_alignment``.

        save_fn : callable
            save_fn(state) -> str, see ``save_alignment``.
        """

        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, RadioButtons, TextBox, CheckButtons
        from methods import NoAlignment

        self.plt = plt
        self.logger = logger
        self.obs_name = obs_name
        self.refine_fn = refine_fn
        self.save_fn = save_fn
        self.guess = tuple(float(v) for v in guess)
        self.guess_source = guess_source
        self.translation_step = float(translation_step)
        self.rotation_step = float(rotation_step)

        # the same rotate-then-shift as the QA plots of the methods
        self._shifter = NoAlignment()

        self.reference = reference
        self.exposure = exposure
        self.reference_display = _normalise(reference)
        self.exposure_display = _normalise(exposure)
        self.exposure_finite = np.isfinite(exposure).astype(float)

        self._set_state(*self.guess, method=MANUAL_METHOD_NAME)
        self.saved = False
        self.message = ""
        self._syncing = False

        # the arrow keys are "back" / "forward" of the toolbar by default
        for key in ("keymap.back", "keymap.forward"):
            plt.rcParams[key] = [
                k for k in plt.rcParams[key] if k not in ("left", "right")
            ]

        self.fig = plt.figure(figsize=(18, 8.5))
        self.fig.canvas.manager.set_window_title(f"Manual alignment: {obs_name}")

        self.ax_overlay = self.fig.add_axes([0.03, 0.30, 0.30, 0.62])
        self.ax_difference = self.fig.add_axes(
            [0.35, 0.30, 0.30, 0.62],
            sharex=self.ax_overlay,
            sharey=self.ax_overlay,
        )
        self.ax_contours = self.fig.add_axes(
            [0.67, 0.30, 0.30, 0.62],
            sharex=self.ax_overlay,
            sharey=self.ax_overlay,
        )

        aligned = self._aligned()

        self.overlay_image = self.ax_overlay.imshow(self._overlay(aligned))
        self.ax_overlay.set_title("Red = reference, cyan = exposure")

        # fixed colour limits (from the initial guess), so changes are visible
        difference = self.reference_display - aligned
        limit = np.nanpercentile(np.abs(difference), 99)
        self.difference_image = self.ax_difference.imshow(
            difference, cmap="gray", vmin=-limit, vmax=limit
        )
        self.ax_difference.set_title("Reference - exposure")

        self.ax_contours.imshow(
            np.zeros_like(self.reference_display), cmap="gray", vmin=0, vmax=1
        )
        self.ax_contours.set_title("Contours: red = reference, blue = exposure")
        self.reference_levels = self._levels(self.reference)
        self.exposure_levels = self._levels(self.exposure)
        if self.reference_levels is not None:
            self.ax_contours.contour(
                self.reference,
                levels=self.reference_levels,
                colors="red",
                linewidths=0.8,
            )
        self.exposure_contours = None

        for ax in (self.ax_overlay, self.ax_difference, self.ax_contours):
            ax.set_xlabel("X Pixel")
            ax.set_ylabel("Y Pixel")

        # controls
        self.text_boxes = {}
        for k, (key, label) in enumerate(
            (("tx", "tx [px]"), ("ty", "ty [px]"), ("rot", "rot [deg]"))
        ):
            ax = self.fig.add_axes([0.06 + 0.12 * k, 0.17, 0.07, 0.045])
            box = TextBox(ax, label + " ", initial="", color="0.15", hovercolor="0.25")
            box.on_submit(lambda text, key=key: self._on_text(key, text))
            self.text_boxes[key] = box

        ax = self.fig.add_axes([0.43, 0.08, 0.10, 0.14])
        ax.set_title("Refine with", fontsize=9)
        self.method_buttons = RadioButtons(
            ax,
            list(methods),
            active=0,
            radio_props=dict(edgecolor="white", facecolor="deepskyblue"),
        )

        ax = self.fig.add_axes([0.54, 0.08, 0.12, 0.06])
        self.verify_button = CheckButtons(
            ax,
            ["verify (reverse)"],
            [True],
            frame_props=dict(edgecolor="white"),
            check_props=dict(facecolor="deepskyblue"),
        )

        self.buttons = {}
        for k, (label, callback) in enumerate(
            (
                ("Refine", self._on_refine),
                ("Reset", self._on_reset),
                ("Save", self._on_save),
            )
        ):
            ax = self.fig.add_axes([0.70 + 0.09 * k, 0.15, 0.08, 0.06])
            button = Button(ax, label, color="0.2", hovercolor="0.35")
            button.on_clicked(callback)
            self.buttons[label] = button

        self.status_text = self.fig.text(0.03, 0.015, "", fontsize=9, va="bottom")

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("close_event", self._on_close)

        self._update()

    # -- state ------------------------------------------------------------

    def _set_state(self, tx, ty, rot, method=MANUAL_METHOD_NAME, errors=None):
        """Set the current values; errors are NaN unless they come from a refinement."""
        errors = errors if errors is not None else (np.nan, np.nan, np.nan)
        self.state = dict(
            tx=float(tx),
            ty=float(ty),
            rot=float(rot),
            tx_error=float(errors[0]),
            ty_error=float(errors[1]),
            rot_error=float(errors[2]),
            method=method,
            roundtrip=np.nan,
        )

    def _change(self, tx=None, ty=None, rot=None):
        """Change the values by hand (the result is no longer a refinement)."""
        self._set_state(
            self.state["tx"] if tx is None else tx,
            self.state["ty"] if ty is None else ty,
            self.state["rot"] if rot is None else rot,
        )
        self.message = ""
        self._update()

    # -- drawing ------------------------------------------------------------

    def _aligned(self):
        """The display exposure, rotated and shifted by the current values."""
        return self._shifter.shift_image(
            self.exposure_display,
            self.state["tx"],
            self.state["ty"],
            self.state["rot"],
        )

    def _overlay(self, aligned):
        """RGB overlay: reference in red, aligned exposure in green and blue."""
        reference = np.clip(np.nan_to_num(self.reference_display), 0, 1)
        aligned = np.clip(np.nan_to_num(aligned), 0, 1)
        return np.dstack([reference, aligned, aligned])

    @staticmethod
    def _levels(frame, n_levels=2, lower_per=80, upper_per=99):
        """Contour levels from percentiles of the frame (higher than the QA contour plots, which trace the background noise of faint frames)."""
        levels = np.unique(
            np.nanpercentile(frame, np.linspace(lower_per, upper_per, n_levels))
        )
        return levels if len(levels) >= 2 else None

    def _correlation(self, aligned):
        """Normalised cross-correlation of the reference and the aligned exposure, over the pixels both cover."""
        from stats import masked_ncc

        covered = (
            self._shifter.shift_image(
                self.exposure_finite,
                self.state["tx"],
                self.state["ty"],
                self.state["rot"],
            )
            > 0.99
        )

        # mode "nearest" repeats the edges of the shifted frame: leave them out
        ny, nx = covered.shape
        margin = int(
            np.ceil(
                max(abs(self.state["tx"]), abs(self.state["ty"]))
                + np.radians(abs(self.state["rot"])) * max(ny, nx) / 2
            )
            + 2
        )
        covered[:margin] = covered[-margin:] = False
        covered[:, :margin] = covered[:, -margin:] = False

        mask = covered & np.isfinite(self.reference_display) & np.isfinite(aligned)

        if mask.sum() < 100:
            return np.nan

        return masked_ncc(self.reference_display, aligned, mask)

    def _update(self):
        """Redraw the panels, the text boxes and the status line."""

        aligned = self._aligned()

        self.overlay_image.set_data(self._overlay(aligned))
        self.difference_image.set_data(self.reference_display - aligned)

        if self.exposure_contours is not None:
            self.exposure_contours.remove()
            self.exposure_contours = None

        if self.exposure_levels is not None:
            aligned_raw = self._shifter.shift_image(
                self.exposure,
                self.state["tx"],
                self.state["ty"],
                self.state["rot"],
            )
            self.exposure_contours = self.ax_contours.contour(
                aligned_raw,
                levels=self.exposure_levels,
                colors="deepskyblue",
                linewidths=0.8,
            )

        self._syncing = True
        for key, box in self.text_boxes.items():
            box.set_val(f"{self.state[key]:.4f}")
        self._syncing = False

        self.fig.suptitle(
            f"{self.obs_name}   tx = {self.state['tx']:.3f} px, ty = "
            f"{self.state['ty']:.3f} px, rot = {self.state['rot']:.4f} deg   "
            f"({self.state['method']}, NCC {self._correlation(aligned):.4f})"
        )

        self.status_text.set_text(
            f"Initial guess: {self.guess_source} ({self.guess[0]:.3f}, "
            f"{self.guess[1]:.3f}, {self.guess[2]:.4f}).   Steps: "
            f"{self.translation_step:g} px, {self.rotation_step:g} deg.   Keys: "
            f"arrows move, ',' '.' rotate, '+' '-' change the steps.\n"
            + (self.message or " ")
        )

        self.fig.canvas.draw_idle()

    def _show_message(self, message):
        """Show a message in the status line and draw it right away (before a long step)."""
        self.message = message
        self._update()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    # -- callbacks ------------------------------------------------------------

    def _on_text(self, key, text):
        """A value was typed into one of the text boxes."""
        if self._syncing:
            return
        try:
            value = float(text)
        except ValueError:
            self.message = f"Not a number: {text!r}"
            self._update()
            return
        if value != self.state[key]:
            self._change(**{key: value})

    def _on_key(self, event):
        """Keyboard controls (ignored while a text box is being edited)."""

        if any(box.capturekeystrokes for box in self.text_boxes.values()):
            return

        t, r = self.translation_step, self.rotation_step

        # the frames are displayed with origin upper: "up" is towards row 0
        moves = {
            "left": dict(tx=self.state["tx"] - t),
            "right": dict(tx=self.state["tx"] + t),
            "up": dict(ty=self.state["ty"] - t),
            "down": dict(ty=self.state["ty"] + t),
            ",": dict(rot=self.state["rot"] - r),
            ".": dict(rot=self.state["rot"] + r),
        }

        if event.key in moves:
            self._change(**moves[event.key])
        elif event.key in ("+", "="):
            self.translation_step *= 2
            self.rotation_step *= 2
            self._update()
        elif event.key == "-":
            self.translation_step /= 2
            self.rotation_step /= 2
            self._update()

    def _on_refine(self, _event):
        """Refine the current values with the selected method."""

        method_name = self.method_buttons.value_selected
        verify = self.verify_button.get_status()[0]

        self._show_message(
            f"Refining with {method_name}"
            + (" (forward and reverse)" if verify else "")
            + " - the window does not respond until it is done ..."
        )

        current = (self.state["tx"], self.state["ty"], self.state["rot"])
        try:
            result = self.refine_fn(method_name, current, verify)
        except Exception as error:
            if self.logger:
                self.logger.exception(f"{method_name} refinement failed")
            result = None
            self.message = f"{method_name} refinement raised {error!r}, see the log."
        else:
            self.message = (
                f"{method_name} refinement failed, values unchanged."
                if result is None
                else ""
            )

        if result is not None:
            self._set_state(
                result["tx"],
                result["ty"],
                result["rot"],
                method=f"{MANUAL_METHOD_NAME}+{method_name}",
                errors=(result["tx_error"], result["ty_error"], result["rot_error"]),
            )
            self.state["roundtrip"] = result["roundtrip"]
            self.message = (
                f"{method_name}: moved by ({result['tx'] - current[0]:+.3f}, "
                f"{result['ty'] - current[1]:+.3f}) px, "
                f"{result['rot'] - current[2]:+.4f} deg; round-trip distance "
                f"{result['roundtrip']:.3f} px"
                + (" (not checked)" if not verify else "")
            )

        self._update()

    def _on_reset(self, _event):
        """Go back to the initial guess."""
        self._change(*self.guess)

    def _on_save(self, _event):
        """Save the current values."""
        self._show_message("Saving ...")
        try:
            written = self.save_fn(dict(self.state))
        except Exception as error:
            if self.logger:
                self.logger.exception("Saving the manual alignment failed")
            self.message = f"Saving failed: {error!r}, see the log."
        else:
            self.saved = True
            self.message = f"Saved ({self.state['method']}): {written}"
        self._update()

    def _on_close(self, _event):
        """Warn if the window is closed without saving."""
        if not self.saved and self.logger:
            self.logger.warning(
                f"Manual alignment window of {self.obs_name} closed without saving."
            )

    def show(self):
        """Open the window (blocks until it is closed)."""
        self.plt.show()


# ============================================================================
# Command line
# ============================================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Manual alignment of one exposure, set up from the output of "
        "a pipeline run."
    )

    parser.add_argument(
        "file",
        type=str,
        help="The exposure to align: a path, or a file name in the input directory.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Output directory of the pipeline run (hybrid or single method). The "
        "initial guess is taken from its output table, and the results are written "
        f"to <output_dir>/{MANUAL_DIR}. Default is the current working directory.",
    )

    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="Configuration file of the run. Needed to prepare the frames exactly "
        "as the pipeline (common frame of the pointing) and for --pymusepipe_flux. "
        "Without it the exposure is aligned in its own frame (the rotation centre "
        "can then differ slightly from the pipeline's).",
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Directory of the exposures. Default: the directory of `file` if it is "
        "a path, else the input directory of the run (run manifest).",
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
        default=None,
        help="Method name of the output table to take the initial guess from "
        f"(<table>_alignment_output_table.fits). Default: the first of "
        f"{', '.join(RESULT_TABLE_NAMES)} that exists.",
    )

    parser.add_argument(
        "--guess",
        type=float,
        nargs=3,
        default=None,
        metavar=("TX", "TY", "ROT"),
        help="Initial guess in pixels and degrees, instead of the output table.",
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        help="No window: take the initial guess (refined with --refine, if given) "
        "and save it.",
    )

    parser.add_argument(
        "--refine",
        type=str,
        default=None,
        choices=("FFT", "OpticalFlow", "PointSources"),
        help="With --headless: refine the initial guess with this method.",
    )

    parser.add_argument(
        "--translation_step",
        type=float,
        default=TRANSLATION_STEP,
        help=f"Initial step of the arrow keys in pixels. Default is {TRANSLATION_STEP}.",
    )

    parser.add_argument(
        "--rotation_step",
        type=float,
        default=ROTATION_STEP,
        help=f"Initial step of the rotation keys in degrees. Default is {ROTATION_STEP}.",
    )

    parser.add_argument(
        "--pymusepipe_flux",
        action="store_true",
        help="Compute FLUX_SCALE and BACKGROUND of the pymusepipe offset table of all "
        "exposures with pymusepipe when saving (slow, needs --config_file). Default: "
        "1 and 0.",
    )

    parser.add_argument(
        "--pymusepipe_filter",
        type=str,
        default="WFI_BB",
        help="Filter for --pymusepipe_flux. Default is 'WFI_BB'.",
    )

    args = parser.parse_args()

    import matplotlib

    if args.headless:
        matplotlib.use("Agg")

    # imports happen here so we can set global backends for matplotlib
    # before importing any other modules that use it

    from astropy.io import fits

    from logger import get_logger, log_run_environment
    from IO import OUTPUT_TABLE_ENTRIES

    manual_dir = pathlib.Path(args.output_dir) / MANUAL_DIR
    manual_dir.mkdir(parents=True, exist_ok=True)

    # the QA plots of the methods during the refinements
    refine_QA_dir = manual_dir / "refine_QA"
    refine_QA_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(manual_dir)
    log_run_environment(logger)

    manifest = find_run_manifest(args.output_dir)

    config = None
    if args.config_file is not None:
        with open(args.config_file, "r") as f:
            config = json.load(f)

    # where the files are
    if args.input_dir is not None:
        input_dir = args.input_dir
    elif os.path.dirname(args.file):
        input_dir = os.path.dirname(args.file)
    else:
        input_dir = manifest.get("input_dir", ".")

    file_path = (
        args.file if os.path.exists(args.file) else os.path.join(input_dir, args.file)
    )
    reference_file = args.ref_file_path or manifest.get("reference_file")

    for description, path in (("exposure", file_path), ("reference", reference_file)):
        if path is None or not os.path.exists(path):
            raise FileNotFoundError(
                f"The {description} {path} does not exist (the paths of the run "
                f"manifest are those of the machine the pipeline ran on). Give it "
                f"with " + ("`file` / --input_dir." if description == "exposure" else "--ref_file_path.")
            )

    obs_name = pathlib.Path(file_path).stem

    logger.info(f"Manual alignment of {file_path}")
    logger.info(f"Reference file: {reference_file}")
    logger.info(f"Output directory of the run: {args.output_dir}")

    table_name, table = find_results_table(
        args.output_dir, table_name=args.table, logger=logger
    )

    # the exposures of the pointing, as in the pipeline
    pointing, index, config_row = (None, None, None)
    if config is not None:
        pointing, index, config_row = find_exposure(config, file_path)
        if pointing is None:
            logger.warning(
                f"{os.path.basename(file_path)} is not in {args.config_file}, "
                f"aligning it in its own frame."
            )
    else:
        logger.warning(
            "No --config_file: aligning the exposure in its own frame, not in the "
            "common frame of its pointing as the pipeline does (the rotation "
            "centre can differ slightly from the pipeline's)."
        )

    if pointing is not None:
        exposures = config[pointing]
        pointing_files = [
            file_path if k == index else os.path.join(input_dir, exposure[2])
            for k, exposure in enumerate(exposures)
        ]
        logger.info(
            f"Pointing {pointing}, exposure {index + 1} of {len(exposures)}: "
            f"preparing the frames of the {len(exposures)} exposures."
        )
    else:
        pointing_files, index = [file_path], 0

    prepared = prepare_frames(
        file_path, pointing_files, index, reference_file, logger=logger
    )

    # row of the exposure in the output table
    date_obs = fits.getheader(file_path, 0).get("DATE-OBS")
    row = find_table_row(table, date_obs, logger=logger)

    if row is None:
        logger.warning(
            f"DATE-OBS {date_obs} of {file_path} is not in the {table_name} output "
            f"table: only the table of this exposure will be written."
        )
    elif config_row is not None and config_row != row:
        logger.warning(
            f"The exposure is row {config_row} of the configuration but row {row} "
            f"of the output table (by DATE_OBS). Was the run made with this "
            f"configuration?"
        )

    # rows of the other exposures of the pointing (for the guess)
    if pointing is not None:
        first_row = config_row - index
        pointing_rows = list(range(first_row, first_row + len(config[pointing])))
        pointing_rows = [r for r in pointing_rows if r < len(table)]
    elif row is not None and int(table["POINTING_OBS"][row]) >= 0:
        pointing_rows = [
            r
            for r in range(len(table))
            if int(table["POINTING_OBS"][r]) == int(table["POINTING_OBS"][row])
        ]
    else:
        pointing_rows = []

    guess, guess_source = initial_guess(
        table,
        table_name,
        row,
        pointing_rows,
        prepared["wcs"],
        guess=args.guess,
        logger=logger,
    )

    logger.info(
        f"Initial guess from {guess_source}: tx = {guess[0]:.3f}, ty = "
        f"{guess[1]:.3f}, rot = {guess[2]:.4f}"
    )

    pymusepipe_flux = None
    if args.pymusepipe_flux:
        if config is None:
            logger.warning(
                "--pymusepipe_flux needs --config_file, writing FLUX_SCALE 1 and "
                "BACKGROUND 0."
            )
        else:
            pymusepipe_flux = dict(
                config_dict=config,
                input_dir=input_dir,
                reference_file=reference_file,
                filter_name=args.pymusepipe_filter,
            )

    header_entries = (
        list(config["headers"].values()) if config is not None else OUTPUT_TABLE_ENTRIES
    )

    def refine(method_name, current, verify):
        """Refinement for the window, see ``refine_alignment``."""
        return refine_alignment(
            method_name,
            prepared["reference"],
            prepared["exposure"],
            prepared["wcs"],
            current,
            obs_name,
            QA_path=refine_QA_dir,
            verify=verify,
            logger=logger,
        )

    def save(state):
        """Saving for the window, see ``save_alignment``."""
        return save_alignment(
            state,
            prepared,
            table,
            table_name,
            row,
            obs_name,
            manual_dir,
            header_entries,
            pymusepipe_flux=pymusepipe_flux,
            logger=logger,
        )

    if args.headless:

        state = dict(
            tx=guess[0],
            ty=guess[1],
            rot=guess[2],
            tx_error=np.nan,
            ty_error=np.nan,
            rot_error=np.nan,
            method=MANUAL_METHOD_NAME,
        )

        if args.refine is not None:
            result = refine(args.refine, guess, True)
            if result is None:
                raise RuntimeError(
                    f"{args.refine} refinement failed, nothing saved (see the log)."
                )
            state.update(
                tx=result["tx"],
                ty=result["ty"],
                rot=result["rot"],
                tx_error=result["tx_error"],
                ty_error=result["ty_error"],
                rot_error=result["rot_error"],
                method=f"{MANUAL_METHOD_NAME}+{args.refine}",
            )

        save(state)

    else:

        window = ManualAlignmentWindow(
            prepared["reference"],
            prepared["exposure"],
            guess,
            guess_source,
            obs_name,
            refine_fn=refine,
            save_fn=save,
            translation_step=args.translation_step,
            rotation_step=args.rotation_step,
            logger=logger,
        )
        window.show()
