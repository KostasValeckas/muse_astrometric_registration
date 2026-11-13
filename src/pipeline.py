from logger import get_logger, format_elapsed, log_run_environment, peak_memory_mb

import time

import itertools
import json
import os
import pathlib
from collections import deque

from methods import NoAlignment

from IO import read_fits_image, extract_image_wcs, make_table_entry, write_final_results
from astropy.io import fits
from geometry import (
    reproject_image_to_target_wcs,
    transformation_matrix,
    compose_transformations,
    matrix_to_parameters,
    inverse_transformation,
    detector_translation_to_sky_translation,
    detector_errors_to_sky_errors,
    sky_translation_to_detector_translation,
    reproject_dithers,
    remove_nan_edges_dithers,
)

from astropy.table import Table

from stats import masked_ncc

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from scipy.ndimage import shift, rotate


# ============================================================================
# Geometry / consistency helpers
# ============================================================================

# Points (measured from the rotation centre) at which two transformations are
# compared in the consistency checks.
CHECK_POINT = (100.0, 100.0)  # the single fixed point used so far
CHECK_ON_FRAME_CORNERS = False  # True: use the 4 frame corners + centre instead.
#                                   This measures the worst-case error over the
#                                   whole frame (stricter on large detectors), so
#                                   the distance threshold may need retuning.


def _image_centre(shape):
    """
    Rotation centre in (x, y) order for an array of `shape` = (ny, nx).
    `transformation_matrix` / `matrix_to_parameters` treat points as [x, y],
    so the centre is given in (x, y) order, not (y, x).

    This is the centre scipy.ndimage.rotate rotates about ((n - 1) / 2 along
    each axis), which is what the alignment methods use to apply rotations.
    """
    return ((shape[1] - 1) / 2.0, (shape[0] - 1) / 2.0)


def _is_usable_alignment(status):
    """
    A dither-to-dither / science entry may be used as a link in a composition
    only if it was measured (1) or derived consistently (0.5). Failed (0) and
    inconsistent (0.7) entries are excluded.
    """
    return status == 1 or status == 0.5


def _check_points(shape):
    """(N, 2) array of test points, relative to the rotation centre."""
    if CHECK_ON_FRAME_CORNERS:
        hx, hy = shape[1] / 2.0, shape[0] / 2.0
        return np.array(
            [[-hx, -hy], [-hx, hy], [hx, -hy], [hx, hy], [0.0, 0.0]], dtype=float
        )
    return np.array([CHECK_POINT], dtype=float)


def _apply_params(tx, ty, rot_deg, pts):
    """Rotate (about the centre, i.e. points are centre-relative) and then
    translate an (N, 2) array of (x, y) points, with the same convention as
    `transformation_matrix`: x -> R(-rot) x + t.

    rot_deg is positive counter-clockwise as the frame is displayed with
    origin upper (numpy array as read by astropy.io.fits), which is the
    scipy.ndimage.rotate convention, i.e. a clockwise R(-rot) in (x, y)."""
    th = np.radians(-rot_deg)
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s], [s, c]])
    return pts @ R.T + np.array([tx, ty])


def _roundtrip_distance(forward, reverse, points):
    """
    Max distance between the test points and reverse(forward(points)).

    forward / reverse are (tx, ty, rot_deg).  For a perfectly consistent pair
    (reverse really is the inverse of forward) this is 0.  Unlike negating the
    parameters, this is correct for any rotation, and the (N, 2) point handling
    avoids the (2,1)+(2,) broadcasting trap.  Returns NaN if anything is
    non-finite (callers treat NaN as inconsistent).
    """
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    back = _apply_params(*reverse, _apply_params(*forward, pts))
    return float(np.max(np.linalg.norm(back - pts, axis=1)))


def _shortest_direct_path(sources, goal, n, is_link):
    """
    Breadth-first search for the shortest path from any of `sources` to `goal`
    over the dithers 0 .. n-1, using the links a -> b for which
    `is_link(a, b)` is True.

    Every link is one direct measurement (or its exact inverse), so this is the
    path built from the fewest direct measurements. A path through a composed
    entry is the same as the chain of direct links it was built from, so
    composed entries never need to be used as links themselves. Ties are broken
    by the lowest dither index (sources and neighbours are visited in
    ascending order).

    Returns the list of dithers [source, ..., goal], or None if there is no path.
    """
    parent = {}
    queue = deque()

    for source in sorted(sources):
        if source == goal:
            return [source]
        parent[source] = None
        queue.append(source)

    while queue:
        a = queue.popleft()
        for b in range(n):
            if b in parent or not is_link(a, b):
                continue

            parent[b] = a

            if b == goal:
                path = [b]
                while parent[path[-1]] is not None:
                    path.append(parent[path[-1]])
                return path[::-1]

            queue.append(b)

    return None


def _science_status(success, composed, composed_from):
    """(label, colour) for the science-vs-reference QA panels."""
    if composed:
        return f"COMPOSED from {composed_from}", "goldenrod"
    if success == 1:
        return "SUCCESS", "green"
    if success == 0.7:
        return "INCONSISTENT", "orangered"
    return "FAILURE", "red"


# for plotting
# status of a dither pair that was never aligned (targeted dither mode:
# only the links needed to recover failed science alignments are computed)
NOT_ATTEMPTED = -1.0

STATUS_COLORS = {
    NOT_ATTEMPTED: "lightgray",
    0: "red",
    1: "green",
    0.5: "yellow",
    0.7: "orange",
}  # not attempted # failed  # success  # composed # inconsistent


cmap = ListedColormap(
    [
        STATUS_COLORS[NOT_ATTEMPTED],
        STATUS_COLORS[0],
        STATUS_COLORS[0.5],
        STATUS_COLORS[0.7],
        STATUS_COLORS[1],
    ]
)
norm = BoundaryNorm([-1.5, 0, 0.5, 0.7, 1, 1.5], cmap.N)


# ============================================================================
# QA-plot layout helpers: nothing below assumes a particular number of dithers.
#
#  * The 4 numeric summary panels (status / x / y / rotation) are 4 *quantities*,
#    so they stay 4 panels whatever the number of dithers; each is an n x n matrix.
#  * Dither thumbnails and the science-vs-reference panels wrap after
#    MAX_PANEL_COLS columns.  With 4 dithers this is the original layout.
#  * Cell-text font size is fitted to the panel width and n; above MAX_ANNOTATED
#    dithers the per-cell text is replaced by a colorbar / colour legend.
# ============================================================================
MAX_PANEL_COLS = (
    5  # panels per row before wrapping (4 dithers + reference = old layout)
)
N_SUMMARY_PANELS = 4  # status / x offsets / y offsets / rotation
MAX_ANNOTATED = (
    12  # above this many dithers: no per-cell text (colorbar / legend instead)
)


def _science_grid_shape(n_dithers, max_cols=MAX_PANEL_COLS):
    """(n_rows, n_cols) for n_dithers dither panels + 1 reference panel.
    Every panel is two axes (image above, text below), so rows come in pairs."""
    n_items = n_dithers + 1
    n_cols = min(n_items, max_cols)
    n_blocks = int(np.ceil(n_items / n_cols))
    return 2 * n_blocks, n_cols


def _dither_grid_shape(n_dithers, n_cols=N_SUMMARY_PANELS):
    """(n_rows, n_cols): thumbnail rows (wrapped at n_cols) + one summary row."""
    return int(np.ceil(n_dithers / n_cols)) + 1, n_cols


def _combined_grid_shape(n_dithers):
    """Science block(s) on top, one numeric-summary row underneath."""
    sci_rows, sci_cols = _science_grid_shape(n_dithers)
    return sci_rows + 1, max(sci_cols, N_SUMMARY_PANELS)


def _display_limits(image, lo=5, hi=95):
    """Percentile stretch that works for masked arrays and plain arrays with NaNs."""
    vals = np.ma.masked_invalid(image).compressed()
    if vals.size == 0:
        return None, None
    v0, v1 = np.percentile(vals, [lo, hi])
    return v0, v1


def _dither_tick_labels(n):
    return [f"Dither {i + 1}" if n <= 6 else f"D{i + 1}" for i in range(n)]


def _fit_fontsize(ax, n, texts, max_fs=12.0, min_fs=5.0):
    """Largest font (<= max_fs) such that the longest cell text fits one cell of
    an n x n matrix drawn on `ax` (width taken from the axes' current size)."""
    w_in = ax.get_position().width * ax.figure.get_figwidth()
    cell_pt = w_in / max(n, 1) * 72.0
    longest = max(len(t) for t in texts)
    return float(np.clip(cell_pt / (0.65 * longest), min_fs, max_fs))


def _draw_dither_matrix(ax, matrix, title, n, text_fn, status=False, **imshow_kwargs):
    """n x n dither-to-dither matrix.  Up to MAX_ANNOTATED dithers every cell is
    annotated; above that the text would overlap, so numeric matrices get a
    colorbar and the status matrix gets a colour legend instead."""
    im = ax.imshow(matrix, interpolation="nearest", **imshow_kwargs)
    ax.set_title(title)
    ax.set_xlabel("Dither Index")
    ax.set_ylabel("Dither Index")

    step = max(1, int(np.ceil(n / 12)))  # thin the ticks for large n
    ticks = list(range(0, n, step))
    labels = _dither_tick_labels(n)
    for setter, labsetter in (
        (ax.set_xticks, ax.set_xticklabels),
        (ax.set_yticks, ax.set_yticklabels),
    ):
        setter(ticks)
        labsetter([labels[t] for t in ticks])
    if n > 4:
        ax.tick_params(axis="x", labelrotation=45)
    if n > MAX_ANNOTATED:
        ax.tick_params(labelsize=8)

    if n <= MAX_ANNOTATED:
        texts = [[text_fn(matrix[i, j]) for j in range(n)] for i in range(n)]
        fs = _fit_fontsize(ax, n, [t for row in texts for t in row])
        for i in range(n):
            for j in range(n):
                ax.text(
                    j,
                    i,
                    texts[i][j],
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=fs,
                )
    elif status:
        names = {
            0: "failed",
            0.5: "composed",
            0.7: "inconsistent",
            1: "ok",
            NOT_ATTEMPTED: "not attempted",
        }
        ax.legend(
            handles=[
                Patch(facecolor=STATUS_COLORS[k], label=names[k])
                for k in (1, 0.5, 0.7, 0, NOT_ATTEMPTED)
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.32),
            ncol=2,
            fontsize=8,
        )
    else:
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _format_parameters(obj):
    """
    The scalar settings of an alignment method (its public attributes that are
    numbers, strings, booleans, None or short tuples of these), as
    "name = value" pairs for the log.
    """
    scalar = (bool, int, float, str, type(None), np.integer, np.floating)
    items = []
    for name, value in sorted(vars(obj).items()):
        if name.startswith("_") or name in ("logger", "FAILED_RETURN", "QA_path"):
            continue
        if isinstance(value, scalar) or (
            isinstance(value, tuple)
            and len(value) <= 6
            and all(isinstance(v, scalar) for v in value)
        ):
            items.append(f"{name} = {value!r}")
    return ", ".join(items)


# ============================================================================
# Output directory guard
# ============================================================================

RUN_MANIFEST_NAME = "run_manifest.json"


def check_run_manifest(output_dir, input_dir, reference_file):
    """
    Guard against writing a run into an output directory that holds the output
    of a different galaxy (QA plots and dither tables of the old run would be
    overwritten or mixed with the new ones).

    The first run writes a small manifest (reference file and input directory)
    to `output_dir`. A later run into the same directory is refused if its
    reference file has a different name. A different input directory with the
    same reference file only gives a warning (the data may have been moved).

    Parameters
    ----------
    output_dir : str or pathlib.Path
        The output directory of the run.

    input_dir : str
        The input directory of the run.

    reference_file : str
        The reference file of the run.

    Returns
    -------
    warning : str or None
        A warning message to be logged, or None.

    Raises
    ------
    ValueError
        If the output directory holds the output of a run with a different
        reference file.
    """

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / RUN_MANIFEST_NAME

    current = {
        "reference_file": os.path.abspath(str(reference_file)),
        "input_dir": os.path.abspath(str(input_dir)),
    }

    warning = None
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                previous = json.load(f)
        except (OSError, ValueError):
            previous = None

        if previous is None:
            warning = (
                f"Could not read the run manifest {manifest_path}, it will be "
                f"overwritten."
            )
        else:
            previous_reference = os.path.basename(previous.get("reference_file", ""))
            if previous_reference != os.path.basename(current["reference_file"]):
                raise ValueError(
                    f"The output directory {output_dir} holds the output of a run "
                    f"with reference file {previous.get('reference_file')}, but "
                    f"this run uses {current['reference_file']}. Use a different "
                    f"output directory (or remove {manifest_path} if the old "
                    f"output is not needed)."
                )
            if previous.get("input_dir") != current["input_dir"]:
                warning = (
                    f"The output directory {output_dir} was used before with the "
                    f"same reference file but input directory "
                    f"{previous.get('input_dir')} (now {current['input_dir']})."
                )

    with open(manifest_path, "w") as f:
        json.dump(current, f, indent=2)

    return warning


class AlignmentPipeline:
    """
    Class to run the alignment pipeline.
    """

    def __init__(
        self,
        config_dict,
        input_dir,
        reference_file,
        output_dir,
        aligment_method=NoAlignment,
        distance_threshold=0.10,
        write_inconsistent=False,
        skip_rotation_estimation=False,
        scan_for_angles=False,
        initial_guess_sources=None,
        dither_mode="full",
    ):
        """
        Initialize the AlignmentPipeline.

        Args:
            config_dict (dict): Configuration dictionary.
            input_dir (str): Path to the input directory.
            reference_file (str): Path to the reference file.
            output_dir (str): Path to the output directory.
            aligment_method (callable): The alignment method to use.
            distance_threshold (float): The threshold (pixels) for the round-trip
                distance between a forward and a reverse alignment to consider
                them consistent. Default is 0.10.
            write_inconsistent (bool): If a science alignment is still flagged
                inconsistent (0.7) after recovery was attempted, write its direct
                measurement to the output table (True) or write None values
                (False, default). An inconsistent alignment is not a result, so
                by default it is written as null; True is only for debugging.
            skip_rotation_estimation (bool): Passed to every alignment method.
                Takes precedence over `scan_for_angles`.
            scan_for_angles (bool): Passed to every alignment method. If True,
                use the shared angle scan. If False (default), use each
                method's own direct rotation estimator.
            initial_guess_sources (list): Optional list of (results_dir, method_name)
                tuples, in priority order, pointing to the output of earlier pipeline
                runs on the same data. For every alignment, the first source that
                holds a usable result is passed to the method as initial guess.
                Default None (no initial guesses).
            dither_mode (str): "full" (default) aligns every dither pair before
                the science alignment. "targeted" aligns the science frames first
                and then only the dither pairs needed to recover the science
                alignments that are not clean, searching for the shortest
                composition one path length at a time. Pairs that are never
                aligned get status NOT_ATTEMPTED (-1) in the dither tables.
        """

        if dither_mode not in ("full", "targeted"):
            raise ValueError(
                f"dither_mode must be 'full' or 'targeted', got {dither_mode!r}"
            )

        # refuse to mix the output of different galaxies in one directory,
        # before the log file is created there
        manifest_warning = check_run_manifest(output_dir, input_dir, reference_file)

        self.logger = get_logger(output_dir)

        if manifest_warning is not None:
            self.logger.warning(manifest_warning)

        self.config_dict = config_dict
        self.input_dir = input_dir
        self.reference_file = reference_file
        self.output_dir = output_dir

        self.output_table_entries = config_dict["headers"]

        self.method = aligment_method
        self.distance_threshold = distance_threshold
        self.write_inconsistent = write_inconsistent
        self.skip_rotation_estimation = skip_rotation_estimation
        self.scan_for_angles = scan_for_angles
        self.dither_mode = dither_mode

        self.initial_guess_sources = (
            list(initial_guess_sources) if initial_guess_sources is not None else []
        )

        # cache for the tables read from disk for the initial guesses
        self._guess_cache = {}

        # which direction of the science alignment the reported result comes
        # from, per (pointing, dither): "forward", "reverse" or None (failed)
        self.science_directions = {}

        # difference image that corresponds to every reported science result,
        # per (pointing, dither): path, or None if no result is reported
        self.reported_science_images = {}

    # ------------------------------------------------------------------
    # Initial guesses from earlier pipeline runs (read from disk)
    # ------------------------------------------------------------------

    def _load_dither_guess_tables(self, results_dir, method_name, pointing):
        """
        Read the dither tables written by `align_dithers` of an earlier run.

        Returns (x_table, y_table, rotation_table, success_table), or None if
        the files do not exist.
        """

        key = ("dither", str(results_dir), method_name, pointing)

        if key in self._guess_cache:
            return self._guess_cache[key]

        tables = []

        for prefix in ("x", "y", "rotation", "success"):
            filename = (
                pathlib.Path(results_dir)
                / f"{prefix}_dither_table_{str(pointing)}_{method_name}.csv"
            )

            if not filename.exists():
                self.logger.warning(
                    f"No dither table {filename} found, no {method_name} dither "
                    f"guesses for pointing {pointing}."
                )
                self._guess_cache[key] = None
                return None

            tables.append(np.atleast_2d(np.loadtxt(filename, delimiter=",")))

        self._guess_cache[key] = tuple(tables)

        return self._guess_cache[key]

    def _load_science_guess_table(self, results_dir, method_name):
        """
        Read the final output table written by `run` of an earlier run.

        Returns the astropy Table, or None if the file does not exist or does
        not match the configuration.
        """

        key = ("science", str(results_dir), method_name)

        if key in self._guess_cache:
            return self._guess_cache[key]

        filename = (
            pathlib.Path(results_dir) / f"{method_name}_alignment_output_table.fits"
        )

        table = None

        if not filename.exists():
            self.logger.warning(
                f"No output table {filename} found, no {method_name} science guesses."
            )

        else:
            table = Table.read(filename, mask_invalid=False)

            # rows are written in configuration order, one per exposure
            n_expected = sum(
                len(exposures)
                for pointing, exposures in self.config_dict.items()
                if pointing != "headers"
            )

            if len(table) != n_expected:
                self.logger.warning(
                    f"Output table {filename} has {len(table)} rows, expected "
                    f"{n_expected}. Not using it for initial guesses."
                )
                table = None

        self._guess_cache[key] = table

        return table

    def _science_row_index(self, pointing, i):
        """
        Row index of dither `i` of `pointing` in the output table (rows are
        written in configuration order).
        """

        index = 0

        for config_pointing, exposures in self.config_dict.items():
            if config_pointing == "headers":
                continue

            if config_pointing == pointing:
                return index + i

            index += len(exposures)

        return None

    def _get_initial_guess(self, kind, pointing, i, j=None, shape=None, wcs=None):
        """
        Initial guess for one alignment, from the first source in
        `initial_guess_sources` that holds a usable result.

        Parameters
        ----------
        kind : str
            "dither" (dither i -> dither j), "science_forward"
            (reference -> dither i) or "science_reverse" (dither i -> reference).

        pointing : str
            Pointing name.

        i, j : int
            Dither indices (j only for "dither").

        shape : tuple
            Frame shape, needed to invert the transformation for
            "science_reverse".

        wcs : astropy.wcs.WCS
            WCS of dither i, needed to convert sky offsets back to pixels for
            the science guesses.

        Returns
        -------
        (tx, ty), rot, source : the guess and the name of the method it came
        from, or (None, None), None, None if no guess is available.
        """

        no_guess = (None, None), None, None

        for results_dir, method_name in self.initial_guess_sources:

            if kind == "dither":

                tables = self._load_dither_guess_tables(
                    results_dir, method_name, pointing
                )

                if tables is None:
                    continue

                x_table, y_table, rotation_table, success_table = tables

                if not _is_usable_alignment(success_table[i, j]):
                    continue

                tx, ty, rot = x_table[i, j], y_table[i, j], rotation_table[i, j]

            else:

                table = self._load_science_guess_table(results_dir, method_name)

                if table is None:
                    continue

                row = table[self._science_row_index(pointing, i)]

                dRA, dDEC, rot = row["RA_OFFSET"], row["DEC_OFFSET"], row["ROTANGLE"]

                if not np.all(np.isfinite([dRA, dDEC, rot])):
                    continue

                # the output table is always in the reference -> dither convention
                tx, ty = sky_translation_to_detector_translation(wcs, dRA, dDEC)

                if kind == "science_reverse":
                    # the inverse of a rotation about the centre is not (-tx, -ty, -rot)
                    centre = _image_centre(shape)
                    T = transformation_matrix(tx, ty, rot, centre)
                    tx, ty, rot = matrix_to_parameters(inverse_transformation(T), centre)

            if not np.all(np.isfinite([tx, ty, rot])):
                continue

            self.logger.info(
                f"Initial guess for {kind} alignment (pointing {pointing}, dither {i+1}"
                + (f" to dither {j+1}" if j is not None else "")
                + f") from {method_name}: tx = {tx:.3f}, ty = {ty:.3f}, rot = {rot:.4f}"
            )

            return (float(tx), float(ty)), float(rot), method_name

        if self.initial_guess_sources:
            self.logger.warning(
                f"No initial guess available for {kind} alignment (pointing {pointing}, "
                f"dither {i+1}" + (f" to dither {j+1}" if j is not None else "") + ")."
            )

        return no_guess

    def _cross_ref_dither_table(
        self, sucess_table, table, threshold_frac=0.1, minimum_value=0.1
    ):
        """
        LEGACY (not called by the pipeline; superseded by
        `_cross_ref_dither_table_displacement`).

        Helper method to `align_dithers`. Opposite values
        (for example x_offset 1 -> 4 and x_offset 4 -> 1) should be the same up
        to threshold. If not, mark the pair inconsistent in the success table.
        """

        # generic handling for all unique dither pairs (assumes square table)
        n = table.shape[0]
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((i, j))

        for a, b in pairs:
            val_ab = np.abs(table[a, b])
            val_ba = np.abs(table[b, a])

            if val_ab < minimum_value and val_ba < minimum_value:
                self.logger.info(
                    f"Values too small to check consistency between dither {a+1} and dither {b+1} ({val_ab} vs. {val_ba}). Skipping consistency check for this pair..."
                )
                continue

            if (
                (abs(val_ab - val_ba) / max(abs(val_ab), abs(val_ba))) > threshold_frac
                and np.isfinite(val_ab)
                and np.isfinite(val_ba)
            ):

                self.logger.warning(
                    f"Alignment between dither {a+1} and dither {b+1} is inconsistent ({val_ab} vs. {val_ba}). Marking as inconsistent."
                )
                sucess_table[a, b] = 0.7
                sucess_table[b, a] = 0.7

        return sucess_table, table

    def _cross_ref_dither_table_displacement(
        self,
        sucess_table,
        x_table,
        y_table,
        rotation_table,
        shape,
        displacement_threshold=0.15,
    ):
        """
        Consistency check between the two directions of every dither pair.

        For a pair (a, b) the forward alignment a -> b is applied to the test
        point(s) and then the reverse alignment b -> a; a consistent pair brings
        the point(s) back to where they started (the round trip is the
        identity).  This is correct for any rotation, unlike comparing against
        the negated parameters.

        Only pairs where BOTH directions are usable (status 1 or 0.5) and finite
        are checked: a failed entry may hold finite junk and must not be allowed
        to downgrade a good link to 0.7.
        """

        n = sucess_table.shape[0]
        points = _check_points(shape)

        for a in range(n):
            for b in range(a + 1, n):

                if not (
                    _is_usable_alignment(sucess_table[a, b])
                    and _is_usable_alignment(sucess_table[b, a])
                ):
                    continue

                fwd = (x_table[a, b], y_table[a, b], rotation_table[a, b])
                rev = (x_table[b, a], y_table[b, a], rotation_table[b, a])

                if not np.all(np.isfinite(fwd + rev)):
                    continue

                displacement = _roundtrip_distance(fwd, rev, points)

                if displacement > displacement_threshold:
                    self.logger.warning(
                        f"Alignment between dither {a+1} and dither {b+1} is inconsistent "
                        f"(round-trip displacement: {displacement:.3f}). Marking as inconsistent."
                    )
                    sucess_table[a, b] = 0.7
                    sucess_table[b, a] = 0.7

        return sucess_table

    def _plot_dither_alignment(
        self,
        dither_set,
        n_dithers,
        succes_table,
        x_return_table,
        y_return_table,
        rotation_return_table,
        obs_title,
        save_path=None,
        title=None,
        save=True,
        show=False,
        axes=None,
        summary_only=False,
    ):
        """
        Dither-to-dither QA: thumbnails (wrapped, any number of dithers) plus the
        4-panel numeric summary (status / x / y / rotation).

        axes : None, 2D array of Axes, or 1D array of >= 4 Axes
            None            -> create a standalone figure sized for n_dithers.
            2D (rows, >=4)  -> LAST row = the 4 summary panels; all rows above it
                               (row-major) = dither thumbnails.  Use
                               _dither_grid_shape(n) for the required shape.
            1D / single row -> summary panels only (no thumbnails), e.g. when the
                               images are already shown elsewhere.
        summary_only : with axes=None, build a 1 x 4 summary-only figure.

        When `axes` is given the caller owns the figure: save/show/title are ignored.
        """
        owns_figure = axes is None

        if owns_figure:
            if summary_only:
                n_rows, n_cols = 1, N_SUMMARY_PANELS
            else:
                n_rows, n_cols = _dither_grid_shape(n_dithers)
            fig, axes = plt.subplots(
                n_rows, n_cols, figsize=(5.5 * n_cols, 6 * n_rows), squeeze=False
            )
        else:
            axes = np.asarray(axes, dtype=object)

        if axes.ndim == 2:
            fig = axes[0, 0].figure
            image_axes = list(axes[:-1].ravel())  # every row but the last
            summary_axes = axes[-1, :N_SUMMARY_PANELS]
        else:
            fig = axes[0].figure
            image_axes = []
            summary_axes = axes[:N_SUMMARY_PANELS]

        if len(summary_axes) < N_SUMMARY_PANELS:
            raise ValueError(
                f"Need {N_SUMMARY_PANELS} summary axes, got {len(summary_axes)}."
            )

        # ---- dither thumbnails ------------------------------------------------
        if image_axes and n_dithers > len(image_axes):
            self.logger.warning(
                f"Only {len(image_axes)} thumbnail axes for {n_dithers} dithers; "
                f"showing the first {len(image_axes)}."
            )
        for k, ax in enumerate(image_axes):
            if k >= n_dithers:
                ax.axis("off")  # unused cell in the grid
                continue
            vmin, vmax = _display_limits(dither_set[k])
            ax.imshow(
                dither_set[k],
                cmap="gray",
                origin="lower",
                interpolation="nearest",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_title(f"Dither {k + 1}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")

        # ---- numeric summary --------------------------------------------------
        long_labels = n_dithers <= 8

        def status_text(v):
            if v == NOT_ATTEMPTED:
                return "NOT RUN" if long_labels else "-"
            if v == 0:
                return "FAILED" if long_labels else "FAIL"
            if v == 0.7:
                return "INCONSISTENT" if long_labels else "INCON"
            if v == 0.5:
                return "COMPOSED" if long_labels else "COMP"
            return "OK"

        def number(v):
            return f"{v:.3f}"

        _draw_dither_matrix(
            summary_axes[0],
            succes_table,
            "Alignment Status Between Dithers",
            n_dithers,
            status_text,
            status=True,
            cmap=cmap,
            norm=norm,
        )
        _draw_dither_matrix(
            summary_axes[1],
            x_return_table,
            "X Offsets Between Dithers",
            n_dithers,
            number,
            cmap="Set1",
        )
        _draw_dither_matrix(
            summary_axes[2],
            y_return_table,
            "Y Offsets Between Dithers",
            n_dithers,
            number,
            cmap="Set1",
        )
        _draw_dither_matrix(
            summary_axes[3],
            rotation_return_table,
            "Rotation Angles Between Dithers",
            n_dithers,
            number,
            cmap="Set1",
        )

        # ---- layout / save / show: only when we own the figure ----------------
        if owns_figure:
            fig.suptitle(title or f"Dither Alignment QA for {obs_title}", fontsize=16)
            fig.tight_layout()
            if save:
                if save_path is None:
                    save_path = (
                        pathlib.Path(self.output_dir) / "aligment_QA" / "dithers"
                    )
                save_path = pathlib.Path(save_path)
                save_path.mkdir(parents=True, exist_ok=True)
                fig.savefig(save_path / f"{obs_title}_dither_alignment.png")
            if show:
                plt.show()
            else:
                plt.close(fig)

        return fig, axes

    @staticmethod
    def _is_direct_dither_link(success_table, a, b):
        """
        True if a -> b is a direct measurement (status 1), or the exact inverse
        of one (b -> a has status 1, i.e. a mirrored alignment).
        """
        return success_table[a, b] == 1 or success_table[b, a] == 1

    @staticmethod
    def _direct_dither_link(
        a,
        b,
        success_table,
        x_table,
        y_table,
        rotation_table,
        x_error_table,
        y_error_table,
        rotation_error_table,
        centre,
    ):
        """
        Transformation matrix and variances (x, y, rotation) of the direct link
        a -> b: the measurement itself if a -> b has status 1, otherwise the
        inverse of b -> a (with the same errors).
        """
        if success_table[a, b] == 1:
            first, second = a, b
        else:
            first, second = b, a

        T = transformation_matrix(
            x_table[first, second],
            y_table[first, second],
            rotation_table[first, second],
            centre,
        )

        if first != a:
            # the inverse of a rotation about the centre is NOT (-tx, -ty, -rot)
            # unless rot == 0, so invert the matrix
            T = inverse_transformation(T)

        var = (
            np.array(
                [
                    x_error_table[first, second],
                    y_error_table[first, second],
                    rotation_error_table[first, second],
                ],
                dtype=float,
            )
            ** 2
        )

        return T, var

    @staticmethod
    def _empty_dither_tables(n_dithers, status=0.0):
        """
        Empty dither tables, in the order returned by `align_dithers`:
        (x, x_error, y, y_error, rotation, rotation_error, success).
        Values are NaN, the success table is filled with `status`.
        """
        tables = [np.full((n_dithers, n_dithers), np.nan) for _ in range(6)]
        tables.append(np.full((n_dithers, n_dithers), status, dtype=float))

        return tuple(tables)

    def _align_dither_pair(self, i, j, dither_set, wcs_set, method, obs_name, tables):
        """
        Align dither i -> dither j (run(dither_i, dither_j)) and store the
        result in entry [i, j] of `tables` (as given by `_empty_dither_tables`).

        Returns True if the alignment succeeded.
        """

        (
            x_return_table,
            x_error_table,
            y_return_table,
            y_error_table,
            rotation_return_table,
            rotation_error_table,
            succes_table,
        ) = tables

        # Same WCS convention as the science alignment: the first image
        # is the "reference", so reference_wcs belongs to the first image
        # and input_wcs to the second.  (All WCSs are identical after
        # reproject_dithers, so this only matters for correctness.)

        # initial guess from earlier runs (if any sources are given)
        guess_translation, guess_rotation, _ = self._get_initial_guess(
            "dither", obs_name, i, j=j
        )

        (
            (tx, ty),
            rotation_angle,
            (tx_error, ty_error),
            rotation_error,
            success,
        ) = method.run(
            dither_set[i],
            dither_set[j],
            f"{obs_name}_dither_{i+1}_to_dither_{j+1}",
            input_wcs=wcs_set[j],
            reference_wcs=wcs_set[i],
            initial_guess_translation=guess_translation,
            initial_guess_rotation=guess_rotation,
            skip_rotation_estimation=self.skip_rotation_estimation,
            scan_for_angles=self.scan_for_angles,
        )

        x_return_table[i, j] = tx
        y_return_table[i, j] = ty
        rotation_return_table[i, j] = rotation_angle

        x_error_table[i, j] = tx_error
        y_error_table[i, j] = ty_error
        rotation_error_table[i, j] = rotation_error

        succes_table[i, j] = success

        return succes_table[i, j] == 1

    def _save_dither_tables(self, obs_name, tables):
        """
        Write the dither tables (as given by `_empty_dither_tables`) to csv in
        the output directory.
        """

        prefixes = (
            "x",
            "x_error",
            "y",
            "y_error",
            "rotation",
            "rotation_error",
            "success",
        )
        headers = (
            "x_offset_table",
            "x_error_table",
            "y_offset_table",
            "y_error_table",
            "rotation_offset_table",
            "rotation_error_table",
            "success_table",
        )

        for prefix, header, table in zip(prefixes, headers, tables):
            filename = (
                f"{prefix}_dither_table_{str(obs_name)}_{self.aligner.name}.csv"
            )
            np.savetxt(
                pathlib.Path(self.output_dir) / filename,
                table,
                delimiter=",",
                header=header,
            )

    def _guess_link_score(self, pointing, a, b):
        """
        How promising the dither link a <-> b is, from the dither tables of the
        initial guess sources: 2 if any source measured it directly (status 1 in
        either direction), 1 if any source has a composed value, 0 otherwise.
        Only used to order the attempts in the targeted dither mode.
        """

        score = 0

        for results_dir, method_name in self.initial_guess_sources:
            tables = self._load_dither_guess_tables(results_dir, method_name, pointing)

            if tables is None:
                continue

            success_table = tables[3]

            if success_table.shape[0] <= max(a, b):
                continue

            statuses = (success_table[a, b], success_table[b, a])

            if 1 in statuses:
                return 2

            if 0.5 in statuses:
                score = 1

        return score

    def _verified_dither_link(
        self, a, b, dither_set, wcs_set, method, obs_name, tables, shape
    ):
        """
        Align the dither pair a <-> b in both directions (a -> b, then b -> a)
        and check that the two are consistent, i.e. that the round trip
        a -> b -> a brings the test points back to within `distance_threshold`
        (the same check as `_cross_ref_dither_table_displacement` and the
        science alignment).

        The link is usable (True) only if both directions succeed AND they are
        consistent; both entries then keep status 1. Otherwise:
        - a -> b fails: b -> a is not run (the link can never be verified),
        - only one direction succeeds: that entry is set to 0.7, since it
          cannot be verified (as for the science alignment),
        - both succeed but disagree: both entries are set to 0.7.
        So a status 1 entry in a targeted table is always a verified link, and
        `_recover_science_alignments` only composes through verified links.
        """
        succes_table = tables[6]

        if not self._align_dither_pair(
            a, b, dither_set, wcs_set, method, obs_name, tables
        ):
            self.logger.info(
                f"Dither {a+1} -> dither {b+1} failed, link {a+1} <-> {b+1} "
                f"cannot be verified (reverse not run)."
            )
            return False

        if not self._align_dither_pair(
            b, a, dither_set, wcs_set, method, obs_name, tables
        ):
            self.logger.warning(
                f"Dither {b+1} -> dither {a+1} failed, dither {a+1} -> dither "
                f"{b+1} cannot be verified. Marking as inconsistent."
            )
            succes_table[a, b] = 0.7
            return False

        x_table, _, y_table, _, rotation_table, _, _ = tables

        distance = _roundtrip_distance(
            (x_table[a, b], y_table[a, b], rotation_table[a, b]),
            (x_table[b, a], y_table[b, a], rotation_table[b, a]),
            _check_points(shape),
        )

        # (written so that a NaN distance also counts as inconsistent)
        if not distance <= self.distance_threshold:
            self.logger.warning(
                f"Alignment between dither {a+1} and dither {b+1} is inconsistent "
                f"(round-trip displacement: {distance:.3f}). Marking as inconsistent."
            )
            succes_table[a, b] = 0.7
            succes_table[b, a] = 0.7
            return False

        self.logger.info(
            f"Dither link {a+1} <-> {b+1} verified (round-trip displacement: "
            f"{distance:.3f})."
        )

        return True

    def align_dithers_targeted(
        self, dither_set, wcs_set, method, science_table, obs_name=None
    ):
        """
        Align only the dither pairs needed to recover the science alignments
        that are not clean (status != 1), instead of all pairs.

        The anchors are the dithers with a clean science alignment. For every
        non-clean dither (the goal) the candidate paths

            anchor -> m_1 -> ... -> goal

        (no dither visited twice) are tried one at a time, shortest first, until
        one works. A path works if every link on it is verified: aligned in
        both directions and consistent, see `_verified_dither_link`. Every link
        is aligned at most once (the result is cached), so a link that failed
        rules out every path through it, and a verified link is reused by the
        other paths and the other goals. E.g. for goal 3 and anchor 1:

            3 - 1;  then 3 - 2 - 1;  then 3 - 4 - 1;  ...;  then 3 - 2 - 4 - 1 ...

        Within one path length the paths are ordered by the number of links
        that still have to be aligned (cheapest first), then by
        `_guess_link_score` (links the initial guess sources measured first),
        then by dither index.

        Since only verified links are left with status 1, the shortest path
        found here is also the one `_recover_science_alignments` composes.

        Returns the tables in the same order as `align_dithers`; pairs that were
        not aligned have status NOT_ATTEMPTED.
        """

        n_dithers = len(dither_set)

        tables = self._empty_dither_tables(n_dithers, status=NOT_ATTEMPTED)

        min_height = min([dither.shape[0] for dither in dither_set])
        min_width = min([dither.shape[1] for dither in dither_set])
        shape = (min_height, min_width)

        anchors = [k for k in range(n_dithers) if science_table[k][6] == 1]
        pending = [k for k in range(n_dithers) if science_table[k][6] != 1]

        if not pending:
            self.logger.info(
                f"Pointing {obs_name}: all science alignments are clean, "
                f"no dither alignments needed."
            )

        elif not anchors:
            self.logger.warning(
                f"Pointing {obs_name}: no clean science alignment to compose from, "
                f"no dither alignments done."
            )

        else:
            self.logger.info(
                f"Pointing {obs_name}: targeted dither alignment for dither(s) "
                f"{', '.join(str(k + 1) for k in pending)} from clean dither(s) "
                f"{', '.join(str(k + 1) for k in anchors)}."
            )

            # (min, max) -> True / False, every link is aligned at most once
            links = {}

            def key(a, b):
                return (min(a, b), max(a, b))

            def link(a, b):
                if key(a, b) not in links:
                    links[key(a, b)] = self._verified_dither_link(
                        a, b, dither_set, wcs_set, method, obs_name, tables, shape
                    )
                return links[key(a, b)]

            def path_links(path):
                return list(zip(path[:-1], path[1:]))

            def path_order(path):
                pairs = path_links(path)
                untested = sum(key(a, b) not in links for a, b in pairs)
                score = sum(self._guess_link_score(obs_name, a, b) for a, b in pairs)
                return (untested, -score, path)

            for goal in pending:

                # possible intermediate dithers: a path through another anchor
                # contains a shorter path from that anchor, tried before
                others = [
                    k for k in range(n_dithers) if k != goal and k not in anchors
                ]
                found = None

                # number of intermediate dithers on the path
                for n_mid in range(0, len(others) + 1):

                    paths = [
                        [anchor, *mids, goal]
                        for anchor in anchors
                        for mids in itertools.permutations(others, n_mid)
                    ]

                    if not paths:
                        break

                    while paths and found is None:

                        # drop paths through a link already known to fail
                        paths = [
                            p
                            for p in paths
                            if all(links.get(key(a, b), True) for a, b in path_links(p))
                        ]

                        if not paths:
                            break

                        path = min(paths, key=path_order)
                        paths.remove(path)

                        if len(path) > 2:
                            self.logger.info(
                                f"Pointing {obs_name}: trying dither path "
                                + " -> ".join(str(k + 1) for k in path)
                                + f" for dither {goal+1}."
                            )

                        # links are tried in order and stop at the first failure
                        if all(link(a, b) for a, b in path_links(path)):
                            found = path

                    if found is not None:
                        break

                if found is not None:
                    self.logger.info(
                        f"Pointing {obs_name}: dither {goal+1} can be composed "
                        f"with {len(found) - 1} verified dither link(s): "
                        + " -> ".join(str(k + 1) for k in found)
                        + "."
                    )
                else:
                    self.logger.warning(
                        f"Pointing {obs_name}: no verified dither path to dither "
                        f"{goal+1}."
                    )

            self.logger.info(
                f"Pointing {obs_name}: {int(np.sum(tables[6] != NOT_ATTEMPTED))} "
                f"dither alignment(s) run ({len(links)} link(s) tested, "
                f"{sum(links.values())} verified), instead of "
                f"{n_dithers * n_dithers} for the full table."
            )

        self._save_dither_tables(obs_name, tables)

        return tables

    def align_dithers(self, dither_set, wcs_set, method, obs_name=None):
        """
        Aligning all dithers relative to each other (all combinations).

        This gives more precise measurements and detects any corrupt frames.

        Direction convention: table[a, b] comes from run(dither_a, dither_b),
        i.e. it is the alignment a -> b.
        """

        n_dithers = len(dither_set)

        # initialize tables to store the results
        tables = self._empty_dither_tables(n_dithers, status=0.0)
        (
            x_return_table,
            x_error_table,
            y_return_table,
            y_error_table,
            rotation_return_table,
            rotation_error_table,
            succes_table,
        ) = tables

        min_height = min([dither.shape[0] for dither in dither_set])
        min_width = min([dither.shape[1] for dither in dither_set])
        shape = (min_height, min_width)
        centre = _image_centre(shape)

        # first aligment, all pairs of dithers are aligned to each other,
        # and the tables are populated

        for i in range(n_dithers):
            for j in range(n_dithers):
                self._align_dither_pair(
                    i, j, dither_set, wcs_set, method, obs_name, tables
                )

        # check for consistency back and forth between all pairs of dithers,
        # and flag any inconsistent pairs

        self.logger.info("Checking dither pairs for forward/reverse consistency ...")

        succes_table = self._cross_ref_dither_table_displacement(
            succes_table,
            x_return_table,
            y_return_table,
            rotation_return_table,
            shape,
            displacement_threshold=self.distance_threshold,
        )
        self.logger.info("Consistency check complete.")

        # Now we try to map out any missed paths: every pair that is not a clean
        # direct measurement is replaced by the shortest path through the direct
        # measurements (a mirrored direct measurement is a path of length one)

        def is_link(a, b):
            return self._is_direct_dither_link(succes_table, a, b)

        for i in range(n_dithers):
            for j in range(n_dithers):
                if i == j or succes_table[i, j] == 1:
                    continue

                path = _shortest_direct_path([i], j, n_dithers, is_link)

                if path is None:
                    self.logger.warning(
                        f"No path found from dither {i+1} to dither {j+1}."
                    )
                    continue

                T = np.eye(3)
                # TODO: check if quadrature is the correct way to propagate errors for composed transformations
                var = np.zeros(3)

                for a, b in zip(path[:-1], path[1:]):
                    T_link, var_link = self._direct_dither_link(
                        a,
                        b,
                        succes_table,
                        x_return_table,
                        y_return_table,
                        rotation_return_table,
                        x_error_table,
                        y_error_table,
                        rotation_error_table,
                        centre,
                    )
                    T = compose_transformations(T, T_link)
                    var = var + var_link

                (
                    x_return_table[i, j],
                    y_return_table[i, j],
                    rotation_return_table[i, j],
                ) = matrix_to_parameters(T, centre)

                (
                    x_error_table[i, j],
                    y_error_table[i, j],
                    rotation_error_table[i, j],
                ) = np.sqrt(var)

                succes_table[i, j] = 0.5

                intermediates = path[1:-1]

                if not intermediates:
                    # one link that is not i -> j itself: the mirrored j -> i
                    name = f"{obs_name}_dither_{i+1}_to_dither_{j+1}_mirrored"
                    title = f"Mirrored Alignment from Dither {i+1} to Dither {j+1}"
                    self.logger.info(
                        f"Using the inverse of dither {j+1} -> dither {i+1} for "
                        f"dither {i+1} -> dither {j+1}."
                    )
                else:
                    through = ", ".join(str(k + 1) for k in intermediates)
                    name = (
                        f"{obs_name}_dither_{i+1}_to_dither_{j+1}_composed_via_"
                        + "_".join(str(k + 1) for k in intermediates)
                    )
                    title = (
                        f"Composed Alignment from Dither {i+1} to Dither {j+1} "
                        f"through Dither(s) {through}"
                    )
                    self.logger.info(
                        f"Found a path from dither {i+1} to dither {j+1} through "
                        f"dither(s) {through} ({len(path) - 1} direct measurements)."
                    )

                method.plot_difference_image(
                    dither_set[i],
                    dither_set[j],
                    x_return_table[i, j],
                    y_return_table[i, j],
                    rotation_return_table[i, j],
                    name,
                    show=False,
                    custom_title=title,
                )

        # check for consistency again (now also covers mirrored / composed entries)

        self.logger.info(
            "Checking for consistency after extrapolating missing paths ..."
        )

        succes_table = self._cross_ref_dither_table_displacement(
            succes_table,
            x_return_table,
            y_return_table,
            rotation_return_table,
            shape,
            displacement_threshold=self.distance_threshold,
        )

        self.logger.info("Consistency check complete.")

        self._save_dither_tables(obs_name, tables)

        return (
            x_return_table,
            x_error_table,
            y_return_table,
            y_error_table,
            rotation_return_table,
            rotation_error_table,
            succes_table,
        )

    def _plot_science_alignment(
        self,
        pointing,
        dither_frames,
        dither_frame_arrays,
        reprojected_reference_frames,
        science_tables,
        QA_dir_science,
        axes=None,
        save=True,
        show=False,
    ):
        """
        Science-vs-reference QA for any number of dithers.

        One panel per dither plus one for the reference; each panel is an image
        axis with a text axis underneath, laid out row-major and wrapped after
        MAX_PANEL_COLS columns.

        axes : None or 2D array of Axes with an even number of rows
            Required shape: _science_grid_shape(len(dither_frames)).  With axes
            given, the caller owns the figure and `save` is ignored.

        #TODO a hard copy and carries too much info in the parameters, needs refactor
        """
        n_dithers = len(dither_frames)
        n_items = n_dithers + 1  # + reference panel
        owns_figure = axes is None

        if owns_figure:
            n_rows, n_cols = _science_grid_shape(n_dithers)
            fig, axes = plt.subplots(
                n_rows, n_cols, figsize=(4.4 * n_cols, 5 * n_rows), squeeze=False
            )
            fig.suptitle(
                f"Pointing {pointing} alignment against reference frame", fontsize=16
            )
        else:
            axes = np.asarray(axes, dtype=object)
            fig = axes[0, 0].figure

        n_cols = axes.shape[1]
        capacity = (axes.shape[0] // 2) * n_cols
        if capacity < n_items:
            raise ValueError(
                f"Axes grid {axes.shape} fits {capacity} panels but {n_items} "
                f"(dithers + reference) are needed; see _science_grid_shape()."
            )

        def panel_axes(k):
            block, col = divmod(k, n_cols)
            return axes[2 * block, col], axes[2 * block + 1, col]

        # ---- one panel per dither ---------------------------------------------
        for i in range(n_dithers):
            ax_img, ax = panel_axes(i)

            image = dither_frame_arrays[i]
            vmin, vmax = _display_limits(image)
            ax_img.imshow(
                image,
                cmap="gray",
                origin="lower",
                interpolation="nearest",
                vmin=vmin,
                vmax=vmax,
            )
            ax_img.set_title(f"Dither {i + 1}")
            ax_img.set_xlabel("x")
            ax_img.set_ylabel("y")
            ax_img.tick_params(labelbottom=False, labelleft=False)

            ax.axis("off")

            (
                tx,
                ty,
                rotation_angle,
                tx_error,
                ty_error,
                rotation_error,
                success,
                composed,
                composed_from,
            ) = science_tables[pointing][i]

            status_text, status_color = _science_status(
                success, composed, composed_from
            )

            if success is None or tx is None or ty is None or rotation_angle is None:
                ax.text(
                    0.05,
                    0.5,
                    f"Dither {i + 1}\nNo result",
                    ha="left",
                    va="center",
                    fontsize=11,
                    color="white",
                )
                ax.text(
                    0.05,
                    0.18,
                    f"Status: {status_text}",
                    ha="left",
                    va="center",
                    fontsize=10,
                    color=status_color,
                    fontweight="bold",
                )
                continue

            main_text = (
                f"Dither {i + 1}\n"
                f"tx = {tx:.3f} px\n"
                f"ty = {ty:.3f} px\n"
                f"rot = {rotation_angle:.3f} deg\n"
                f"tx_err = {tx_error:.3f}\n"
                f"ty_err = {ty_error:.3f}\n"
                f"rot_err = {rotation_error:.3f}"
            )
            ax.text(
                0.05,
                0.95,
                main_text,
                ha="left",
                va="top",
                fontsize=9,
                color="white",
                linespacing=1.4,
            )
            ax.text(
                0.05,
                0.18,
                f"Status: {status_text}",
                ha="left",
                va="center",
                fontsize=10,
                color=status_color,
                fontweight="bold",
            )
            ax.axvspan(0.0, 1.0, ymin=0.0, ymax=1.0, facecolor="white", alpha=0.0)
            ax.set_facecolor("white")
            for spine in ax.spines.values():
                spine.set_visible(False)

        # ---- reference panel (the item after the last dither) -----------------
        # reprojected_reference_frames is filled per pointing in run(); accept the
        # (pointing, k) key form too so either convention works.
        ref_key = (
            pointing if pointing in reprojected_reference_frames else (pointing, 0)
        )
        reference_frame_pointing = reprojected_reference_frames[ref_key]
        v_med = np.nanmedian(reference_frame_pointing)
        v_std = np.nanstd(reference_frame_pointing)

        ref_ax, summary_ax = panel_axes(n_dithers)
        ref_ax.imshow(
            reference_frame_pointing,
            cmap="gray",
            origin="lower",
            interpolation="nearest",
            vmin=v_med - 3 * v_std,
            vmax=v_med + 3 * v_std,
        )
        ref_ax.set_title("Reference")
        ref_ax.set_xlabel("x")
        ref_ax.set_ylabel("y")
        ref_ax.tick_params(labelbottom=False, labelleft=False)

        def _short_status(s):
            if s == 1:
                return "OK"
            if s == 0.5:
                return "COMP"
            if s == 0.7:
                return "INCON"
            return "FAIL"

        summary_ax.axis("off")
        summary_ax.text(
            0.05,
            0.9,
            f"Reference alignment\nPointing {pointing}\n"
            f"{n_dithers} dithers\n"
            f"Reference file:\n{self.reference_file}",
            ha="left",
            va="top",
            fontsize=10,
            linespacing=1.5,
        )
        summary_ax.text(
            0.05,
            0.45,
            "Status summary:\n"
            + "\n".join(
                f"D{i + 1}: {_short_status(science_tables[pointing][i][6])}"
                for i in range(n_dithers)
            ),
            ha="left",
            va="top",
            fontsize=9 if n_dithers <= 8 else 6,
            linespacing=1.4,
        )

        # ---- hide unused cells at the end of the grid -------------------------
        for k in range(n_items, capacity):
            for unused in panel_axes(k):
                unused.axis("off")

        if owns_figure:
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            if save:
                fig.savefig(
                    QA_dir_science / f"{pointing}_dither_to_reference_QA.png", dpi=200
                )
            if show:
                plt.show()
            else:
                plt.close(fig)

        return fig, axes

    def _plot_combined_report(
        self,
        pointing,
        dither_frames,
        dither_frame_arrays,
        reprojected_reference_frames,
        science_tables,
        QA_dir_science,
        QA_dir_combined,
        success_table,
        x_table,
        y_table,
        rotation_table,
        save=True,
        show=False,
    ):
        """
        One-page QA report: science-vs-reference block(s) on top, the dither-to-
        dither numeric summary underneath.  Grid size follows the dither count.
        """
        n_dithers = len(dither_frame_arrays)
        n_rows, n_cols = _combined_grid_shape(n_dithers)
        sci_rows = n_rows - 1

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(4.4 * n_cols, 5.33 * n_rows), squeeze=False
        )
        fig.suptitle(f"Pointing {pointing} alignment QA report", fontsize=16)

        self._plot_science_alignment(
            pointing,
            dither_frames,
            dither_frame_arrays,
            reprojected_reference_frames,
            science_tables,
            QA_dir_science,
            axes=axes[:sci_rows, :],
        )

        # a single row of axes -> summary panels only (thumbnails are already
        # shown in the science block above)
        self._plot_dither_alignment(
            dither_frame_arrays,
            n_dithers,
            success_table,
            x_table,
            y_table,
            rotation_table,
            obs_title=pointing + "_after_compositions_",
            axes=axes[-1, :N_SUMMARY_PANELS],
        )
        for ax in axes[-1, N_SUMMARY_PANELS:]:
            ax.axis("off")

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        if save:
            pathlib.Path(QA_dir_combined).mkdir(parents=True, exist_ok=True)
            fig.savefig(
                QA_dir_combined / f"{pointing}_combined_QA_{self.aligner.name}.png",
                dpi=200,
                bbox_inches="tight",
            )
        if show:
            plt.show()
        else:
            plt.close(fig)  # run() never closed these -> figure leak per pointing

        return fig, axes

    def _science_forward_reverse(
        self,
        pointing,
        i,
        reference_frame,
        dither_frame,
        reference_wcs,
        dither_wcs,
    ):
        """
        Align dither `i` to the reference in both directions and check that the
        two results are consistent.

        Direction convention: run(reference, dither) gives ref -> dither, and
        run(dither, reference) gives dither -> ref, so the second should be the
        inverse of the first.

        Returns the science-table tuple
        (tx, ty, rot, tx_err, ty_err, rot_err, status, composed, composed_from)
        with status 1 (consistent), 0.7 (cannot be verified / inconsistent) or
        0 (both directions failed).
        """

        # initial guesses from earlier runs (if any sources are given)
        guess_translation, guess_rotation, _ = self._get_initial_guess(
            "science_forward", pointing, i, shape=dither_frame.shape, wcs=dither_wcs
        )
        guess_translation_rev, guess_rotation_rev, _ = self._get_initial_guess(
            "science_reverse", pointing, i, shape=dither_frame.shape, wcs=dither_wcs
        )

        (
            (tx, ty),
            rotation_angle,
            (tx_error, ty_error),
            rotation_error,
            success,
        ) = self.aligner.run(
            reference_frame,
            dither_frame,
            f"{pointing}_dither_{i+1}_to_reference",
            input_wcs=dither_wcs,
            reference_wcs=reference_wcs,
            initial_guess_translation=guess_translation,
            initial_guess_rotation=guess_rotation,
            skip_rotation_estimation=self.skip_rotation_estimation,
            scan_for_angles=self.scan_for_angles,
        )

        (
            (tx_rev, ty_rev),
            rotation_angle_rev,
            (tx_error_rev, ty_error_rev),
            rotation_error_rev,
            success_rev,
        ) = self.aligner.run(
            dither_frame,
            reference_frame,
            f"{pointing}_reference_to_dither_{i+1}",
            input_wcs=reference_wcs,
            reference_wcs=dither_wcs,
            initial_guess_translation=guess_translation_rev,
            initial_guess_rotation=guess_rotation_rev,
            skip_rotation_estimation=self.skip_rotation_estimation,
            scan_for_angles=self.scan_for_angles,
        )

        fwd_ok = success == 1
        rev_ok = success_rev == 1
        shape = dither_frame.shape
        centre = _image_centre(shape)

        if fwd_ok and rev_ok:
            distance = _roundtrip_distance(
                (tx, ty, rotation_angle),
                (tx_rev, ty_rev, rotation_angle_rev),
                _check_points(shape),
            )
            direction = "forward"
            if distance <= self.distance_threshold:
                status = 1
            else:
                # (written so that a NaN distance also counts as inconsistent)
                self.logger.warning(
                    f"Alignment for pointing {pointing}, dither {i+1} is inconsistent. "
                    f"Round-trip distance: {distance:.3f}. Marking as inconsistent."
                )
                status = 0.7

        elif fwd_ok:
            # forward worked, reverse did not: cannot verify
            self.logger.warning(
                f"Pointing {pointing}, dither {i+1}: reverse alignment failed, "
                f"forward alignment cannot be verified. Marking as inconsistent."
            )
            direction = "forward"
            status = 0.7

        elif rev_ok:
            # reverse worked, forward did not: use the true inverse of the reverse
            self.logger.warning(
                f"Pointing {pointing}, dither {i+1}: forward alignment failed, "
                f"using the inverse of the reverse alignment. Marking as inconsistent."
            )
            T_rev = transformation_matrix(tx_rev, ty_rev, rotation_angle_rev, centre)
            tx, ty, rotation_angle = matrix_to_parameters(
                inverse_transformation(T_rev), centre
            )
            tx_error, ty_error, rotation_error = (
                tx_error_rev,
                ty_error_rev,
                rotation_error_rev,
            )
            direction = "reverse"
            status = 0.7

        else:
            direction = None
            status = 0.0

        self.science_directions[(pointing, i)] = direction

        # the last two entries book-keep whether this was a composed alignment
        # and what it was composed from
        return (
            tx,
            ty,
            rotation_angle,
            tx_error,
            ty_error,
            rotation_error,
            status,
            False,
            None,
        )

    def _recover_science_alignments(
        self,
        pointing,
        n,
        dither_frame_arrays,
        reprojected_reference_frames,
        science_tables,
        success_table,
        x_table,
        y_table,
        rotation_table,
        x_error_table,
        y_error_table,
        rotation_error_table,
    ):
        """
        Replace every science alignment that is not a clean direct measurement
        (status 0 failed, 0.7 inconsistent) by the shortest composition

            ref -> k (direct science, status 1)  ->  ... ->  i (direct dither links)

        Path length is the number of direct measurements: the science link plus
        the dither links, where a dither link is a direct dither measurement
        (status 1) or its exact inverse (mirrored). A path through a recovered or
        composed entry is the same as the chain of direct links it was built
        from, so only direct links are needed and there is no limit on the
        number of intermediate dithers. Ties are broken by the lowest dither index.

        Direction convention:
            science_tables[pointing][i] : ref -> i
            dither tables [a, b]        : a -> b
        compose_transformations(A, B) applies A first, then B.
        """
        st = science_tables[pointing]
        pending = [i for i in range(n) if st[i][6] != 1]

        for i in pending:
            self.logger.warning(
                f"Found non-clean case at pointing {pointing}, dither {i+1} "
                f"(status {st[i][6]}). Attempting to recover..."
            )

        # dithers with a direct (clean) science alignment are the starting points
        anchors = [k for k in range(n) if st[k][6] == 1]

        def is_link(a, b):
            return self._is_direct_dither_link(success_table, a, b)

        for i in list(pending):
            centre = _image_centre(dither_frame_arrays[i].shape)

            path = _shortest_direct_path(anchors, i, n, is_link)

            if path is None:
                continue

            k = path[0]
            tx_k, ty_k, rot_k, xe_k, ye_k, re_k = st[k][:6]

            T = transformation_matrix(tx_k, ty_k, rot_k, centre)
            # TODO: check if quadrature is the correct way to propagate errors for composed transformations
            var = np.array([xe_k, ye_k, re_k], dtype=float) ** 2

            for a, b in zip(path[:-1], path[1:]):
                T_link, var_link = self._direct_dither_link(
                    a,
                    b,
                    success_table,
                    x_table,
                    y_table,
                    rotation_table,
                    x_error_table,
                    y_error_table,
                    rotation_error_table,
                    centre,
                )
                T = compose_transformations(T, T_link)
                var = var + var_link

            tx_c, ty_c, rot_c = matrix_to_parameters(T, centre)
            via = f"dither {k + 1}" + "".join(
                f" via dither {j + 1}" for j in path[1:-1]
            )

            st[i] = (tx_c, ty_c, rot_c, *np.sqrt(var), 0.5, True, via)

            self.aligner.plot_difference_image(
                reprojected_reference_frames[pointing],
                dither_frame_arrays[i],
                tx_c,
                ty_c,
                rot_c,
                f"{pointing}_dither_{i+1}_to_reference_composed_via_{via.replace(' ', '_')}",
                show=False,
                custom_title=f"Composed Alignment from Dither {i+1} to reference through {via}",
            )

            self.logger.info(
                f"Recovered alignment for pointing {pointing}, dither {i+1} using {via} "
                f"({len(path)} direct measurements)."
            )
            pending.remove(i)

        for i in pending:
            self.logger.warning(
                f"After attempting recovery, the alignment for pointing {pointing}, "
                f"dither {i+1} is still not clean (status {st[i][6]})."
            )

    def _failed_table_row(self, filename):
        """
        Output table row (header entries and None values) for an exposure of a
        pointing that could not be processed. The header entries are read from
        the file if possible, so the row still identifies the exposure (and the
        hybrid run can match it with the other methods); otherwise they are
        None as well.
        """
        try:
            with fits.open(filename) as hdul:
                entry = make_table_entry(
                    hdul,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    output_table_entries=self.output_table_entries.values(),
                    logger=self.logger,
                )
            return tuple(entry.values())
        except Exception:
            self.logger.exception(
                f"Could not read the header of {filename} either, writing an "
                f"empty row."
            )
            return tuple([None] * (len(self.output_table_entries) + 6))

    def _log_run_summary(self, output_table):
        """
        Last lines of a run before the total time: how many exposures have a
        result, and anything that went wrong unexpectedly (alignments or
        pointings that raised an error), so a long run can be checked from the
        end of its log.
        """
        n_headers = len(self.output_table_entries)
        n_ok = sum(
            1
            for row in output_table
            if row[n_headers] is not None and np.isfinite(row[n_headers])
        )

        self.logger.info(
            f"Summary ({self.aligner.name}): {n_ok} of {len(output_table)} "
            f"exposures aligned, {len(output_table) - n_ok} without a result. "
            f"Peak memory {peak_memory_mb():.0f} MB."
        )

        n_crashed = getattr(self.aligner, "n_crashed_runs", 0)
        if n_crashed:
            self.logger.warning(
                f"Summary ({self.aligner.name}): {n_crashed} alignment(s) raised an "
                f"unexpected error and were set as failed (search the log for "
                f"'Unexpected error')."
            )

        if self.crashed_pointings:
            self.logger.warning(
                f"Summary ({self.aligner.name}): pointing(s) "
                f"{', '.join(str(p) for p in self.crashed_pointings)} raised an "
                f"unexpected error and were written as failed (search the log for "
                f"'Unexpected error')."
            )

    def run(self, skip_dithers = False):
        """
        Run the alignment pipeline.

        For now simple. The work is done in `_run`; this wrapper only logs the
        total execution time as the last line of the log. If the run crashes or
        is interrupted, it logs "ABORTED" with the traceback instead of
        "finished" (so the log shows how far it got and how long it took), and
        re-raises.

        skip_dithers: bool
            For debugging, dev stuff, don't touch.
        """
        start_time = time.perf_counter()
        try:
            self._run(skip_dithers=skip_dithers)
        except BaseException as e:
            # a crash (or KeyboardInterrupt / a killed job) must not look like a
            # finished run in the log - log the traceback and re-raise
            self.logger.exception(
                f"Alignment pipeline ({self.method.__name__}) ABORTED "
                f"({type(e).__name__}) after "
                f"{format_elapsed(time.perf_counter() - start_time)}."
            )
            raise
        self.logger.info(
            f"Alignment pipeline ({self.method.__name__}) finished, total "
            f"execution time: {format_elapsed(time.perf_counter() - start_time)}"
        )

    def _run(self, skip_dithers = False):
        """
        The alignment pipeline itself, see `run`.
        """

        self.logger.info("Starting alignment pipeline...")

        # code version, host, packages, command line: to reproduce and debug
        log_run_environment(self.logger)

        self.logger.info(f"Reference file: {self.reference_file}")
        self.logger.info(f"Input directory: {self.input_dir}")
        self.logger.info(f"Output directory: {self.output_dir}")

        n_exposures = sum(
            len(exposures)
            for pointing, exposures in self.config_dict.items()
            if pointing != "headers"
        )
        self.logger.info(
            f"Configuration: {len(self.config_dict) - 1} pointings, "
            f"{n_exposures} exposures"
        )
        self.logger.info(
            f"Pipeline settings: method {self.method.__name__}, distance_threshold "
            f"{self.distance_threshold}, write_inconsistent {self.write_inconsistent}, "
            f"skip_rotation_estimation {self.skip_rotation_estimation}, "
            f"scan_for_angles {self.scan_for_angles}, dither_mode {self.dither_mode}, "
            f"initial guesses from {self.initial_guess_sources or 'none'}"
        )

        # pointings that raised an unexpected error (written as failed)
        self.crashed_pointings = []

        # output containers
        output_table = []
        dither_tables = {}
        science_tables = {}
        reprojected_reference_frames = {}

        # Create the QA directory for the dithers
        QA_dir_dithers = pathlib.Path(self.output_dir) / "aligment_QA" / "dithers"
        QA_dir_dithers.mkdir(parents=True, exist_ok=True)

        # and create the QA directory for the science frames
        QA_dir_science = pathlib.Path(self.output_dir) / "aligment_QA" / "science"
        QA_dir_science.mkdir(parents=True, exist_ok=True)

        # Create the QA directory for combined reports
        QA_dir_combined = pathlib.Path(self.output_dir) / "aligment_QA" / "combined"
        QA_dir_combined.mkdir(parents=True, exist_ok=True)

        # make an aligner with dither QA path
        self.aligner = self.method(QA_path=QA_dir_dithers, logger=self.logger)

        self.logger.info(
            f"{self.aligner.name} parameters: " + _format_parameters(self.aligner)
        )

        # prepare the reference frame and its WCS
        reference_WCS = extract_image_wcs(self.reference_file, logger=self.logger)
        reference_frame, _ = read_fits_image(self.reference_file, logger=self.logger)

        # this is the actual aligment loop

        self.logger.info("Starting alignment loop...")
        for pointing, exposures in self.config_dict.items():

            # this entry is only used for bookeeping
            if pointing == "headers":
                continue

            # rows of this pointing start here (removed again if the pointing
            # crashes, see below)
            n_rows_before = len(output_table)
            pointing_start_time = time.perf_counter()

            # an unexpected error in one pointing (I/O, reprojection, recovery,
            # QA report) must not end the run: it is logged, the exposures of
            # the pointing are written as failed, and the next pointing is run
            try:

                science_tables[pointing] = {}

                self.logger.info(f"Processing pointing: {pointing}")

                dither_frame_arrays = []
                dither_frame_wcs = []
                dither_hduls = []

                dither_frames = [
                    pathlib.Path(self.input_dir) / exposure[2] for exposure in exposures
                ]

                # I/O for dithers
                for dither_frame in dither_frames:
                    self.logger.info(f"Reading dither frame: {dither_frame}")

                    input_WCS = extract_image_wcs(
                        dither_frame, logger=self.logger, wcs_extension=1
                    )

                    input_frame, hdul = read_fits_image(
                        dither_frame, logger=self.logger, data_extension=1
                    )

                    dither_frame_arrays.append(input_frame)
                    dither_hduls.append(hdul)
                    dither_frame_wcs.append(input_WCS)

                # just for good measure, make sure QA path is reset to dithers:

                self.aligner.QA_path = QA_dir_dithers

                # bring dithers to the same WCS and shape using reprojection
                dither_frame_arrays, dither_frame_wcs = reproject_dithers(
                    dither_frame_arrays,
                    dither_frame_wcs,
                    logger=self.logger,
                    method="interpolation",
                )

                # remove nan edge

                # TODO we keep this and use it for reprojection in a bit of a hack

                original_shape = dither_frame_arrays[0].shape

                dither_frame_arrays, edge_cut_value = remove_nan_edges_dithers(
                    dither_frame_arrays, logger=self.logger
                )

                if not skip_dithers and self.dither_mode == "full":

                    # ... and run the dither aligment
                    (
                        x_table,
                        x_error_table,
                        y_table,
                        y_error_table,
                        rotation_table,
                        rotation_error_table,
                        success_table,
                    ) = self.align_dithers(
                        dither_frame_arrays, dither_frame_wcs, self.aligner, obs_name=pointing
                    )

                # targeted mode: the dither tables are filled after the science
                # alignment; skipped dithers: empty tables (no links)

                else:

                    (
                        x_table,
                        x_error_table,
                        y_table,
                        y_error_table,
                        rotation_table,
                        rotation_error_table,
                        success_table,
                    ) = self._empty_dither_tables(
                        len(dither_frame_arrays), status=NOT_ATTEMPTED
                    )

                # now science aligment

                # set the QA path for the science alignment to the science QA directory
                self.aligner.QA_path = QA_dir_science

                # all the WCS are now the same, so we only need to reproject the reference frame once per dither

                reprojected_reference_frame, _ = reproject_image_to_target_wcs(
                    reference_frame,
                    reference_WCS,
                    dither_frame_wcs[0],
                    shape_out=original_shape,
                )

                reference_WCS_reprojected = dither_frame_wcs[
                    0
                ]  # update reference WCS to match the dither WCS

                # also crop the same edges as the dither frames to ensure consistency
                reprojected_reference_frame = reprojected_reference_frame[
                    edge_cut_value : reprojected_reference_frame.shape[0] - edge_cut_value,
                    edge_cut_value : reprojected_reference_frame.shape[1] - edge_cut_value,
                ]

                # keep this for plotting later
                reprojected_reference_frames[pointing] = reprojected_reference_frame

                # 1) align every dither to the reference both ways and check consistency
                for i in range(len(dither_frame_arrays)):

                    self.logger.info(
                        f"Aligning dither frame {dither_frames[i]} to reference frame {self.reference_file}"
                    )

                    science_tables[pointing][i] = self._science_forward_reverse(
                        pointing,
                        i,
                        reprojected_reference_frame,
                        dither_frame_arrays[i],
                        reference_WCS_reprojected,
                        dither_frame_wcs[i],
                    )

                # targeted mode: align only the dither links needed to recover the
                # science alignments that are not clean
                if not skip_dithers and self.dither_mode == "targeted":

                    self.aligner.QA_path = QA_dir_dithers

                    (
                        x_table,
                        x_error_table,
                        y_table,
                        y_error_table,
                        rotation_table,
                        rotation_error_table,
                        success_table,
                    ) = self.align_dithers_targeted(
                        dither_frame_arrays,
                        dither_frame_wcs,
                        self.aligner,
                        science_tables[pointing],
                        obs_name=pointing,
                    )

                    self.aligner.QA_path = QA_dir_science

                dither_tables[pointing] = (x_table, y_table, rotation_table, success_table)

                # 2) extrapolate: recover failed / inconsistent entries by composing
                #    transformations through other dithers
                self._recover_science_alignments(
                    pointing,
                    len(dither_frames),
                    dither_frame_arrays,
                    reprojected_reference_frames,
                    science_tables,
                    success_table,
                    x_table,
                    y_table,
                    rotation_table,
                    x_error_table,
                    y_error_table,
                    rotation_error_table,
                )

                # one-page QA report; the grid size follows the number of dithers
                self._plot_combined_report(
                    pointing,
                    dither_frames,
                    dither_frame_arrays,
                    reprojected_reference_frames,
                    science_tables,
                    QA_dir_science,
                    QA_dir_combined,
                    success_table,
                    x_table,
                    y_table,
                    rotation_table,
                )

                # now make the output tuples and add to the output table

                for i in range(len(dither_frames)):
                    (
                        tx,
                        ty,
                        rotation_angle,
                        tx_error,
                        ty_error,
                        rotation_error,
                        success,
                        composed,
                        composed_from,
                    ) = science_tables[pointing][i]

                    # 1 and 0.5 are always written; a still-inconsistent (0.7) entry
                    # is written only if write_inconsistent is set; 0 is never written
                    writable = success in (1, 0.5) or (
                        self.write_inconsistent and success == 0.7
                    )
                    has_values = all(
                        v is not None and np.isfinite(v)
                        for v in (
                            tx,
                            ty,
                            rotation_angle,
                            tx_error,
                            ty_error,
                            rotation_error,
                        )
                    )

                    if not (writable and has_values):
                        self.logger.warning(
                            f"Alignment for pointing {pointing}, dither {i+1} failed "
                            f"(status {success}). Writing None values."
                        )

                        dRA = dRA_sigma = dDEC = dDEC_sigma = rotation_angle = (
                            rotation_error
                        ) = None

                        self.reported_science_images[(pointing, i)] = None

                    else:
                        if success == 0.7:
                            self.logger.warning(
                                f"Pointing {pointing}, dither {i+1}: writing an inconsistent "
                                f"alignment (no recovery possible)."
                            )

                        # convert offsets and offset errors from detector pixels to sky coordinates (degrees)

                        dRA, dDEC = detector_translation_to_sky_translation(
                            dither_frame_wcs[i], tx, ty, logger=self.logger
                        )

                        # errors are propagated through the WCS Jacobian, not
                        # converted like a displacement (see the function)
                        dRA_sigma, dDEC_sigma = detector_errors_to_sky_errors(
                            dither_frame_wcs[i], tx_error, ty_error
                        )

                        # book-keep the difference image of the reported result
                        if composed:
                            image_obs_name = (
                                f"{pointing}_dither_{i+1}_to_reference_composed_via_"
                                f"{composed_from.replace(' ', '_')}"
                            )
                        elif self.science_directions.get((pointing, i)) == "reverse":
                            image_obs_name = f"{pointing}_reference_to_dither_{i+1}"
                        else:
                            image_obs_name = f"{pointing}_dither_{i+1}_to_reference"

                        self.reported_science_images[(pointing, i)] = (
                            QA_dir_science
                            / f"difference_image_{image_obs_name}_{self.aligner.name}.png"
                        )

                    table_entry = make_table_entry(
                        dither_hduls[i],
                        dRA,
                        dRA_sigma,
                        dDEC,
                        dDEC_sigma,
                        rotation_angle,
                        rotation_error,
                        output_table_entries=self.output_table_entries.values(),
                        logger=self.logger,
                    )

                    # unpack the values and append
                    output_table.append(tuple(value for _, value in table_entry.items()))

            except Exception:
                self.logger.exception(
                    f"Unexpected error while processing pointing {pointing} - "
                    f"writing its {len(exposures)} exposures as failed and "
                    f"continuing with the next pointing."
                )
                self.crashed_pointings.append(pointing)

                del output_table[n_rows_before:]
                for i, exposure in enumerate(exposures):
                    self.reported_science_images[(pointing, i)] = None
                    output_table.append(
                        self._failed_table_row(
                            pathlib.Path(self.input_dir) / exposure[2]
                        )
                    )

            self.logger.info(
                f"Pointing {pointing} done in "
                f"{format_elapsed(time.perf_counter() - pointing_start_time)}, "
                f"peak memory so far {peak_memory_mb():.0f} MB"
            )


        # write the final results to a FITS table
        write_final_results(
            output_table, self.aligner.name, self.output_dir, logger=self.logger
        )

        self._log_run_summary(output_table)
