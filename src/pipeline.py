from logger import get_logger
 
import pathlib
 
from methods import  NoAlignment
 
from IO import read_fits_image, extract_image_wcs, make_table_entry, write_final_results
from geometry import (
    reproject_image_to_target_wcs,
    transformation_matrix,
    compose_transformations,
    matrix_to_parameters,
    inverse_transformation,
    detector_translation_to_sky_translation,
    reproject_dithers,
    compare_distance
)

import numpy as np
import matplotlib.pyplot as plt
 
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
 
def _image_centre(shape):
    """
    Rotation centre in (x, y) order for an array of `shape` = (ny, nx).
    `transformation_matrix` / `matrix_to_parameters` treat points as [x, y],
    so the centre must be (nx // 2, ny // 2), not (ny // 2, nx // 2).
    (Identical for square frames.)
    """
    return (shape[1] // 2, shape[0] // 2)
 
def _is_usable_alignment(status):
    """
    A dither-to-dither entry may be used as a link in a composition only if it
    was measured (1) or derived consistently (0.5). Failed (0) and
    inconsistent (0.7) entries are excluded.
    """
    return status == 1 or status == 0.5
 
# for plotting
STATUS_COLORS = {
    0: "red",
    1: "green",
    0.5: "yellow",
    0.7: "orange",
}  # failed  # success  # composed # inconsistent
 
 
 
cmap = ListedColormap(
    [STATUS_COLORS[0], STATUS_COLORS[0.5], STATUS_COLORS[0.7], STATUS_COLORS[1]]
)
norm = BoundaryNorm([0, 0.5, 0.7, 1, 1.5], cmap.N)
 
 
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
MAX_PANEL_COLS = 5       # panels per row before wrapping (4 dithers + reference = old layout)
N_SUMMARY_PANELS = 4     # status / x offsets / y offsets / rotation
MAX_ANNOTATED = 12       # above this many dithers: no per-cell text (colorbar / legend instead)
 
 
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
 
    step = max(1, int(np.ceil(n / 12)))               # thin the ticks for large n
    ticks = list(range(0, n, step))
    labels = _dither_tick_labels(n)
    for setter, labsetter in ((ax.set_xticks, ax.set_xticklabels),
                              (ax.set_yticks, ax.set_yticklabels)):
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
                ax.text(j, i, texts[i][j], ha="center", va="center",
                        color="black", fontsize=fs)
    elif status:
        names = {0: "failed", 0.5: "composed", 0.7: "inconsistent", 1: "ok"}
        ax.legend(
            handles=[Patch(facecolor=STATUS_COLORS[k], label=names[k]) for k in (1, 0.5, 0.7, 0)],
            loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2, fontsize=8,
        )
    else:
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
 
 
class AlignmentPipeline:
    """
    Class to run the alignment pipeline.
    """
 
    def __init__(self, config_dict, input_dir, reference_file, output_dir, aligment_method = NoAlignment, distance_threshold = 0.10):
        """
        Initialize the AlignmentPipeline.
 
        Args:
            config_dict (dict): Configuration dictionary.
            input_dir (str): Path to the input directory.
            reference_file (str): Path to the reference file.
            output_dir (str): Path to the output directory.
            aligment_method (callable): The alignment method to use.
            distance_threshold (float): The threshold for the distance between cross-alignments to consider them consistent. Default is 0.10.
        """
 
        self.logger = get_logger(output_dir)
 
        self.config_dict = config_dict
        self.input_dir = input_dir
        self.reference_file = reference_file
        self.output_dir = output_dir
 
        self.output_table_entries = config_dict["headers"]
 
        self.method = aligment_method
        self.distance_threshold = distance_threshold    
    def _cross_ref_dither_table(
        self, sucess_table, table, threshold_frac=0.1, minimum_value=0.1
    ):
        """
        Helper method to `align_dithers`. Opposite values
        (for example x_offset 1 -> 4 and x_offset 4 -> 1) should be the same up
        to threshold. If now, replace with nans and and adjust sucess table.
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
                    f"Alignment between dither {a+1} and dither {b+1} is inconsistent ({val_ab} vs. {val_ba}). Replacing with NaN values."
                )
                sucess_table[a, b] = 0.7
                sucess_table[b, a] = 0.7
 
        return sucess_table, table
    
    def _cross_ref_dither_table_displacement(
        self, sucess_table, x_table, y_table, rotation_table, displacement_threshold=0.15, pixel_origin = (100,100)
    ):
        """
        Same as `_cross_ref_dither_table` but checks the displacement between the two dithers instead of the individual x, y and rotation values.
        """

        # generic handling for all unique dither pairs (assumes square table)
        n = sucess_table.shape[0]
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((i, j))

        for a, b in pairs:

            a_x = x_table[a, b]
            a_y = y_table[a, b]
            a_rot = rotation_table[a, b]

            # negate the values for the opposite direction
            b_x = -x_table[b, a]
            b_y = -y_table[b, a]
            b_rot = -rotation_table[b, a]

            # now, for a point (100,100) from origin, first rotate and then 
            # displace. Calculate the euclidean distance between the two points. 
            # If it is larger than the threshold, then the alignment is inconsistent.

            point = np.array([[pixel_origin[0]], [pixel_origin[1]]])

            displacement = compare_distance(a_x, a_y, a_rot, b_x, b_y, b_rot, point_from_center= point)

            if displacement > displacement_threshold:
                self.logger.warning(
                    f"Alignment between dither {a+1} and dither {b+1} is inconsistent (displacement: {displacement}). Replacing with NaN values."
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
            image_axes = list(axes[:-1].ravel())          # every row but the last
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
                ax.axis("off")                            # unused cell in the grid
                continue
            vmin, vmax = _display_limits(dither_set[k])
            ax.imshow(
                dither_set[k], cmap="gray", origin="lower",
                interpolation="nearest", vmin=vmin, vmax=vmax,
            )
            ax.set_title(f"Dither {k + 1}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
 
        # ---- numeric summary --------------------------------------------------
        long_labels = n_dithers <= 8
 
        def status_text(v):
            if v == 0:
                return "FAILED" if long_labels else "FAIL"
            if v == 0.7:
                return "INCONSISTENT" if long_labels else "INCON"
            if v == 0.5:
                return "COMPOSED" if long_labels else "COMP"
            return "OK"
 
        def number(v):
            return f"{v:.3f}"
 
        _draw_dither_matrix(summary_axes[0], succes_table,
                            "Alignment Status Between Dithers", n_dithers,
                            status_text, status=True, cmap=cmap, norm=norm)
        _draw_dither_matrix(summary_axes[1], x_return_table,
                            "X Offsets Between Dithers", n_dithers,
                            number, cmap="Set1")
        _draw_dither_matrix(summary_axes[2], y_return_table,
                            "Y Offsets Between Dithers", n_dithers,
                            number, cmap="Set1")
        _draw_dither_matrix(summary_axes[3], rotation_return_table,
                            "Rotation Angles Between Dithers", n_dithers,
                            number, cmap="Set1")
 
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
 
    def align_dithers(self, dither_set, wcs_set, method, obs_name=None):
        """
        Aligning all dithers relative to each other (all combinations).
 
        This gives more precise measurements and detects any corrupt frames.
        """
 
        n_dithers = len(dither_set)
 
        # initialize tables to store the results
        x_return_table = np.full((n_dithers, n_dithers), np.nan)
        x_error_table = np.full((n_dithers, n_dithers), np.nan)
        y_return_table = np.full((n_dithers, n_dithers), np.nan)
        y_error_table = np.full((n_dithers, n_dithers), np.nan)
        rotation_return_table = np.full((n_dithers, n_dithers), np.nan)
        rotation_error_table = np.full((n_dithers, n_dithers), np.nan)
        succes_table = np.full((n_dithers, n_dithers), 0, dtype=float)
 
        min_height = min([dither.shape[0] for dither in dither_set])
        min_width = min([dither.shape[1] for dither in dither_set])
 
        # first aligment, all pairs of dithers are aligned to each other, 
        # and the tables are populated
 
        for i, dither_1 in enumerate(dither_set):
 
            for j, dither_2 in enumerate(dither_set):
 
                (
                    (tx, ty),
                    rotation_angle,
                    (tx_error, ty_error),
                    rotation_error,
                    success,
                ) = method.run(
                    dither_1,
                    dither_2,
                    f"{obs_name}_dither_{i+1}_to_dither_{j+1}",
                    input_wcs = wcs_set[i],
                    reference_wcs = wcs_set[j],
                )
 
                x_return_table[i, j] = tx
                y_return_table[i, j] = ty
                rotation_return_table[i, j] = rotation_angle
 
                x_error_table[i, j] = tx_error
                y_error_table[i, j] = ty_error
                rotation_error_table[i, j] = rotation_error
 
                succes_table[i, j] = success
 
        # check for consistency back and forth between all pairs of dithers, 
        # and replace any inconsistent values with nans
 
        self.logger.info("Checking for consistency in x alignment ...")
 
        succes_table = self._cross_ref_dither_table_displacement(
            succes_table, x_return_table, y_return_table, rotation_return_table, displacement_threshold=self.distance_threshold
        )
        self.logger.info("Consistency check complete.")
 
        # Now we try to map out any missed paths
        for i in range(n_dithers):
            for j in range(n_dithers):
                if i == j:
                    continue
 
                # this bool is used to break out of the loop if a path is found
                success = False
                if not succes_table[i, j] == 1:
                    # direct path does not exist, look for intermediate frames
 
                    # first check if opposite pair exists - if so, replace with negative of that:
                    if succes_table[j, i] == 1:
                        # The inverse of a rotation about the centre is NOT
                        # (-tx, -ty, -rot) unless rot == 0, so invert the matrix.
                        centre = _image_centre((min_height, min_width))
 
                        T_j_i = transformation_matrix(
                            x_return_table[j, i],
                            y_return_table[j, i],
                            rotation_return_table[j, i],
                            centre,
                        )
 
                        (
                            x_return_table[i, j],
                            y_return_table[i, j],
                            rotation_return_table[i, j],
                        ) = matrix_to_parameters(inverse_transformation(T_j_i), centre)
 
                        x_error_table[i, j] = x_error_table[j, i]
                        y_error_table[i, j] = y_error_table[j, i]
                        rotation_error_table[i, j] = rotation_error_table[j, i]
                        succes_table[i, j] = 0.5
                        success = True
 
                        # the QA plot:
                        method.plot_difference_image(
                            dither_set[i],
                            dither_set[j],
                            x_return_table[i, j],
                            y_return_table[i, j],
                            rotation_return_table[i, j],
                            f"{obs_name}_dither_{i+1}_to_dither_{j+1}_mirrored",
                            show=False,
                            custom_title=f"Mirrored Alignment from Dither {i+1} to Dither {j+1}",
                        )
 
                    # this is the loop for all dithers. If opposite pair was found,
                    # the success bool will break it.
                    for k in range(n_dithers):
                        if success:
                            break
                        
                        # skip duplicates
                        if k == i or k == j:
                            continue
 
                        # one intermediate frame search
                        if succes_table[i, k] == 1 and succes_table[k, j] == 1:
                            self.logger.info(
                                f"Found a path from dither {i+1} to dither {j+1} through dither {k+1}."
                            )
                            T_i_k = transformation_matrix(
                                x_return_table[i, k],
                                y_return_table[i, k],
                                rotation_return_table[i, k],
                                _image_centre((min_height, min_width)),
                            )
                            T_k_j = transformation_matrix(
                                x_return_table[k, j],
                                y_return_table[k, j],
                                rotation_return_table[k, j],
                                _image_centre((min_height, min_width)),
                            )
                            T_i_j_composed = compose_transformations(T_i_k, T_k_j)
                            tx_composed, ty_composed, rotation_angle_composed = (
                                matrix_to_parameters(
                                    T_i_j_composed,
                                    _image_centre((min_height, min_width)),
                                )
                            )
                            x_return_table[i, j] = tx_composed
                            y_return_table[i, j] = ty_composed
                            rotation_return_table[i, j] = rotation_angle_composed
 
                            # TODO: check if this is the correct way to propagate errors for composed transformations
 
                            tx_error = np.sqrt(
                                x_error_table[i, k] ** 2 + x_error_table[k, j] ** 2
                            )
                            ty_error = np.sqrt(
                                y_error_table[i, k] ** 2 + y_error_table[k, j] ** 2
                            )
                            rotation_error = np.sqrt(
                                rotation_error_table[i, k] ** 2
                                + rotation_error_table[k, j] ** 2
                            )
 
                            x_error_table[i, j] = tx_error
                            y_error_table[i, j] = ty_error
                            rotation_error_table[i, j] = rotation_error
 
                            succes_table[i, j] = 0.5
 
                            success = True
 
                            method.plot_difference_image(
                                dither_set[i],
                                dither_set[j],
                                tx_composed,
                                ty_composed,
                                rotation_angle_composed,
                                f"{obs_name}_dither_{i+1}_to_dither_{j+1}_composed",
                                show=False,
                                custom_title=f"Composed Alignment from Dither {i+1} to Dither {j+1} through Dither {k+1}",
                            )
                            break
 
                    # two intermediate frames search
                    if not success:
                        for k in range(n_dithers):
                            if k == i or k == j:
                                continue
                            for l in range(n_dithers):
                                if l == i or l == j or l == k:
                                    continue
 
                                if (
                                    succes_table[i, k] == 1
                                    and succes_table[k, l] == 1
                                    and succes_table[l, j] == 1
                                ):
                                    self.logger.info(
                                        f"Found a path from dither {i+1} to dither {j+1} through dithers {k+1} and {l+1}."
                                    )
                                    T_i_k = transformation_matrix(
                                        x_return_table[i, k],
                                        y_return_table[i, k],
                                        rotation_return_table[i, k],
                                        _image_centre((min_height, min_width)),
                                    )
                                    T_k_l = transformation_matrix(
                                        x_return_table[k, l],
                                        y_return_table[k, l],
                                        rotation_return_table[k, l],
                                        _image_centre((min_height, min_width)),
                                    )
                                    T_l_j = transformation_matrix(
                                        x_return_table[l, j],
                                        y_return_table[l, j],
                                        rotation_return_table[l, j],
                                        _image_centre((min_height, min_width)),
                                    )
                                    T_i_j_composed = compose_transformations(
                                        compose_transformations(T_i_k, T_k_l), T_l_j
                                    )
                                    (
                                        tx_composed,
                                        ty_composed,
                                        rotation_angle_composed,
                                    ) = matrix_to_parameters(
                                        T_i_j_composed,
                                        _image_centre((min_height, min_width)),
                                    )
                                    x_return_table[i, j] = tx_composed
                                    y_return_table[i, j] = ty_composed
                                    rotation_return_table[i, j] = (
                                        rotation_angle_composed
                                    )
 
                                    #TODO: check if this is the correct way to propagate errors for composed transformations
 
                                    tx_error = np.sqrt(
                                        x_error_table[i, k] ** 2
                                        + x_error_table[k, l] ** 2
                                        + x_error_table[l, j] ** 2
                                    )
 
                                    ty_error = np.sqrt(
                                        y_error_table[i, k] ** 2
                                        + y_error_table[k, l] ** 2
                                        + y_error_table[l, j] ** 2
                                    )
 
                                    rotation_error = np.sqrt(
                                        rotation_error_table[i, k] ** 2
                                        + rotation_error_table[k, l] ** 2
                                        + rotation_error_table[l, j] ** 2
                                    )
 
                                    x_error_table[i, j] = tx_error
                                    y_error_table[i, j] = ty_error
                                    rotation_error_table[i, j] = rotation_error
                                    succes_table[i, j] = 0.5
 
                                    success = True
 
                                    method.plot_difference_image(
                                        dither_set[i],
                                        dither_set[j],
                                        tx_composed,
                                        ty_composed,
                                        rotation_angle_composed,
                                        f"{obs_name}_dither_{i+1}_to_dither_{j+1}_composed_2_intermediates",
                                        show=False,
                                        custom_title=f"Composed Alignment from Dither {i+1} to Dither {j+1} through Dithers {k+1} and {l+1}",
                                    )
 
                                    break
                            if success:
                                break
 
                    if not success:
                        self.logger.warning(
                            f"No path found from dither {i+1} to dither {j+1}."
                        )
 
        # check for consistency again
 
        self.logger.info(
            "Checking for consistency after extrapolating missing paths ..."
        )
 
        self.logger.info("Checking for consistency in x alignment ...")
 
        succes_table = self._cross_ref_dither_table_displacement(
            succes_table, x_return_table, y_return_table, rotation_return_table, displacement_threshold=self.distance_threshold
        )
 
        self.logger.info("Consistency check complete.")
 
        return (
            x_return_table,
            x_error_table,
            y_return_table,
            y_error_table,
            rotation_return_table,
            rotation_error_table,
            succes_table
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
        n_items = n_dithers + 1                           # + reference panel
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
                image, cmap="gray", origin="lower", interpolation="nearest",
                vmin=vmin, vmax=vmax,
            )
            ax_img.set_title(f"Dither {i + 1}")
            ax_img.set_xlabel("x")
            ax_img.set_ylabel("y")
            ax_img.tick_params(labelbottom=False, labelleft=False)
 
            ax.axis("off")
 
            (
                tx, ty, rotation_angle,
                tx_error, ty_error, rotation_error,
                success, composed, composed_from,
            ) = science_tables[pointing][i]
 
            status_text = (
                "SUCCESS"
                if (bool(success) and not bool(composed))
                else (
                    f"COMPOSED from dither {composed_from}"
                    if bool(composed)
                    else "FAILURE"
                )
            )
            status_color = (
                "green"
                if (bool(success) and not bool(composed))
                else "orange" if bool(composed) else "red"
            )
 
            if success is None or tx is None or ty is None or rotation_angle is None:
                ax.text(0.05, 0.5, f"Dither {i + 1}\nNo result",
                        ha="left", va="center", fontsize=11, color="white")
                ax.text(0.05, 0.18, f"Status: {status_text}",
                        ha="left", va="center", fontsize=10,
                        color=status_color, fontweight="bold")
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
            ax.text(0.05, 0.95, main_text, ha="left", va="top", fontsize=9,
                    color="white", linespacing=1.4)
            ax.text(0.05, 0.18, f"Status: {status_text}", ha="left", va="center",
                    fontsize=10, color=status_color, fontweight="bold")
            ax.axvspan(0.0, 1.0, ymin=0.0, ymax=1.0, facecolor="white", alpha=0.0)
            ax.set_facecolor("white")
            for spine in ax.spines.values():
                spine.set_visible(False)
 
        # ---- reference panel (the item after the last dither) -----------------
        # reprojected_reference_frames is filled per pointing in run(); accept the
        # (pointing, k) key form too so either convention works.
        ref_key = pointing if pointing in reprojected_reference_frames else (pointing, 0)
        reference_frame_pointing = reprojected_reference_frames[ref_key]
        v_med = np.nanmedian(reference_frame_pointing)
        v_std = np.nanstd(reference_frame_pointing)
 
        ref_ax, summary_ax = panel_axes(n_dithers)
        ref_ax.imshow(
            reference_frame_pointing, cmap="gray", origin="lower",
            interpolation="nearest", vmin=v_med - 3 * v_std, vmax=v_med + 3 * v_std,
        )
        ref_ax.set_title("Reference")
        ref_ax.set_xlabel("x")
        ref_ax.set_ylabel("y")
        ref_ax.tick_params(labelbottom=False, labelleft=False)
 
        summary_ax.axis("off")
        summary_ax.text(
            0.05, 0.9,
            f"Reference alignment\nPointing {pointing}\n"
            f"{n_dithers} dithers\n"
            f"Reference file:\n{self.reference_file}",
            ha="left", va="top", fontsize=10, linespacing=1.5,
        )
        summary_ax.text(
            0.05, 0.45,
            "Status summary:\n"
            + "\n".join(
                f"D{i + 1}: {'OK' if bool(science_tables[pointing][i][6]) else 'FAIL'}"
                for i in range(n_dithers)
            ),
            ha="left", va="top", fontsize=9 if n_dithers <= 8 else 6,
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
            pointing, dither_frames, dither_frame_arrays,
            reprojected_reference_frames, science_tables, QA_dir_science,
            axes=axes[:sci_rows, :],
        )
 
        # a single row of axes -> summary panels only (thumbnails are already
        # shown in the science block above)
        self._plot_dither_alignment(
            dither_frame_arrays, n_dithers, success_table,
            x_table, y_table, rotation_table,
            obs_title=pointing + "_after_compositions_",
            axes=axes[-1, :N_SUMMARY_PANELS],
        )
        for ax in axes[-1, N_SUMMARY_PANELS:]:
            ax.axis("off")
 
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        if save:
            pathlib.Path(QA_dir_combined).mkdir(parents=True, exist_ok=True)
            fig.savefig(QA_dir_combined / f"{pointing}_combined_QA.png", dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)          # run() never closed these -> figure leak per pointing
 
        return fig, axes
 
    def run(self):
        """
        Run the alignment pipeline.
 
        For now simple.
        """
 
        self.logger.info("Starting alignment pipeline...")
 
        self.logger.info(f"Reference file: {self.reference_file}")
 
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
        self.aligner = self.method(
            QA_path=QA_dir_dithers, logger=self.logger
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
                dither_frame_arrays, dither_frame_wcs, logger=self.logger, method="interpolation"
            )
 
            # ... and run the dither aligment
            (
                x_table,
                x_error_table,
                y_table,
                y_error_table,
                rotation_table,
                rotation_error_table,
                success_table
            ) = self.align_dithers(
                dither_frame_arrays,
                dither_frame_wcs,
                self.aligner,
                obs_name=pointing,
            )
 
            dither_tables[pointing] = (x_table, y_table, rotation_table, success_table)
 
            # now science aligment
 
 
            # set the QA path for the science alignment to the science QA directory
            self.aligner.QA_path = QA_dir_science
 
            # all the WCS are now the same, so we only need to reproject the reference frame once per dither
 
            reprojected_reference_frame, _ = reproject_image_to_target_wcs(
                reference_frame,
                reference_WCS,
                dither_frame_wcs[0],
                shape_out=dither_frame_arrays[0].shape,
            )

            reference_WCS_reprojected = dither_frame_wcs[0]  # update reference WCS to match the dither WCS
 
            # keep this for plotting later
            reprojected_reference_frames[pointing] = (
                reprojected_reference_frame
            )
 
            for i, dither_frame in enumerate(dither_frame_arrays):
 
                self.logger.info(
                    f"Aligning dither frame {dither_frame} to reference frame {self.reference_file}"
                )
 
                (
                    (tx, ty),
                    rotation_angle,
                    (tx_error, ty_error),
                    rotation_error,
                    success,
                ) = self.aligner.run(
                    reprojected_reference_frame,
                    dither_frame_arrays[i],
                    f"{pointing}_dither_{i+1}_to_reference",
                    input_wcs=dither_frame_wcs[i],
                    reference_wcs=reference_WCS_reprojected
                )

                # run it in reverse, to check for consistency
                (
                    (tx_rev, ty_rev),
                    rotation_angle_rev,
                    (tx_error_rev, ty_error_rev),
                    rotation_error_rev,
                    success_rev,
                ) = self.aligner.run(
                    dither_frame_arrays[i],
                    reprojected_reference_frame,
                    f"{pointing}_reference_to_dither_{i+1}",
                    input_wcs=reference_WCS_reprojected,
                    reference_wcs=dither_frame_wcs[i]
                )

                # check for consistency

                distance = compare_distance(
                    tx, ty, rotation_angle,
                    -tx_rev, -ty_rev, -rotation_angle_rev
                )

                if distance > self.distance_threshold:
                    self.logger.warning(
                        f"Alignment for pointing {pointing}, dither {i+1} is inconsistent. Distance: {distance:.3f}. Marking as inconsistent."
                    )
                    success = 0.7
 
                # put the results in the output table
                science_tables[pointing][i] = (
                    tx,
                    ty,
                    rotation_angle,
                    tx_error,
                    ty_error,
                    rotation_error,
                    success,
                    # the last two entries are for bookeeping whether this was a composed alignment 
                    # and which dither it was composed from
                    False,
                    i,
                )
 
            # TODO insert reverse aligment here for consistency check 
 
            # Now we check for failed alignments and attempt to recover them by
            # composing transformations through other dithers
            for i in range(len(dither_frames)):
                (
                    tx,
                    ty,
                    rotation_angle,
                    tx_error,
                    ty_error,
                    rotation_error,
                    success,
                    _,
                    _,
                ) = science_tables[pointing][i]
 
                if not (success==1):
                    self.logger.warning(
                        f"Found failed case at pointing {pointing}, dither {i+1}. Attempting to recover..."
                    )
 
                    # Direction convention (see geometry.py):
                    #   science table : run(reference, dither_i) -> ref -> i
                    #   dither tables : run(dither_a, dither_b)  -> a -> b, stored at [a, b]
                    # To get ref -> i through an already-aligned dither j we chain
                    #   ref -> j   (science_tables[j])   then   j -> i   (dither table [j, i]).
                    # compose_transformations(A, B) applies A first, then B.
                    centre = _image_centre(dither_frame_arrays[i].shape)
                    n_dith = len(dither_frames)
                    recovered = False
 
                    # one intermediate frame: ref -> j -> i
                    for j in range(n_dith):
                        if j == i:
                            continue
                        if not _is_usable_alignment(success_table[j, i]):
                            continue
                        if not science_tables[pointing][j][6]:
                            continue
 
                        (
                            tx_j,
                            ty_j,
                            rotation_angle_j,
                            tx_error_science_j,
                            ty_error_science_j,
                            rotation_error_science_j,
                            _,
                            _,
                            _,
                        ) = science_tables[pointing][j]
 
                        S_ref_j = transformation_matrix(
                            tx_j, ty_j, rotation_angle_j, centre
                        )
                        D_j_i = transformation_matrix(
                            x_table[j, i], y_table[j, i], rotation_table[j, i], centre
                        )
                        T_ref_i = compose_transformations(S_ref_j, D_j_i)
                        tx_composed, ty_composed, rotation_angle_composed = (
                            matrix_to_parameters(T_ref_i, centre)
                        )
 
                        # TODO: check if this is the correct way to propagate errors for composed transformations
                        tx_error = np.sqrt(
                            tx_error_science_j**2 + x_error_table[j, i] ** 2
                        )
                        ty_error = np.sqrt(
                            ty_error_science_j**2 + y_error_table[j, i] ** 2
                        )
                        rotation_error = np.sqrt(
                            rotation_error_science_j**2
                            + rotation_error_table[j, i] ** 2
                        )
 
                        science_tables[pointing][i] = (
                            tx_composed,
                            ty_composed,
                            rotation_angle_composed,
                            tx_error,
                            ty_error,
                            rotation_error,
                            True,
                            True,
                            j + 1,
                        )
 
                        self.logger.info(
                            f"Recovered alignment for pointing {pointing}, dither {i+1} using dither {j+1}."
                        )
 
                        self.aligner.plot_difference_image(
                            reprojected_reference_frames[pointing],
                            dither_frame_arrays[i],
                            tx_composed,
                            ty_composed,
                            rotation_angle_composed,
                            f"{pointing}_dither_{i+1}_to_reference_composed_via_dither_{j+1}",
                            show=False,
                            custom_title=f"Composed Alignment from Dither {i+1} to reference through dither {j+1}",
                        )
 
                        recovered = True
                        break
 
                    # two intermediate frames: ref -> k -> j -> i
                    if not recovered:
                        for k in range(n_dith):
                            if k == i or not science_tables[pointing][k][6]:
                                continue
                            for j in range(n_dith):
                                if j == i or j == k:
                                    continue
                                if not (
                                    _is_usable_alignment(success_table[k, j])
                                    and _is_usable_alignment(success_table[j, i])
                                ):
                                    continue
 
                                (
                                    tx_k,
                                    ty_k,
                                    rotation_angle_k,
                                    tx_error_science_k,
                                    ty_error_science_k,
                                    rotation_error_science_k,
                                    _,
                                    _,
                                    _,
                                ) = science_tables[pointing][k]
 
                                S_ref_k = transformation_matrix(
                                    tx_k, ty_k, rotation_angle_k, centre
                                )
                                D_k_j = transformation_matrix(
                                    x_table[k, j],
                                    y_table[k, j],
                                    rotation_table[k, j],
                                    centre,
                                )
                                D_j_i = transformation_matrix(
                                    x_table[j, i],
                                    y_table[j, i],
                                    rotation_table[j, i],
                                    centre,
                                )
                                T_ref_i = compose_transformations(
                                    compose_transformations(S_ref_k, D_k_j), D_j_i
                                )
                                tx_composed, ty_composed, rotation_angle_composed = (
                                    matrix_to_parameters(T_ref_i, centre)
                                )
 
                                tx_error = np.sqrt(
                                    tx_error_science_k**2
                                    + x_error_table[k, j] ** 2
                                    + x_error_table[j, i] ** 2
                                )
                                ty_error = np.sqrt(
                                    ty_error_science_k**2
                                    + y_error_table[k, j] ** 2
                                    + y_error_table[j, i] ** 2
                                )
                                rotation_error = np.sqrt(
                                    rotation_error_science_k**2
                                    + rotation_error_table[k, j] ** 2
                                    + rotation_error_table[j, i] ** 2
                                )
 
                                science_tables[pointing][i] = (
                                    tx_composed,
                                    ty_composed,
                                    rotation_angle_composed,
                                    tx_error,
                                    ty_error,
                                    rotation_error,
                                    True,
                                    True,
                                    f"dither {k + 1} via dither {j + 1}",
                                )
 
                                self.logger.info(
                                    f"Recovered alignment for pointing {pointing}, dither {i+1} using dither {k+1} via dither {j+1}."
                                )
 
                                self.point_sources_aligner.plot_difference_image(
                                    reprojected_reference_frames[pointing],
                                    dither_frame_arrays[i],
                                    tx_composed,
                                    ty_composed,
                                    rotation_angle_composed,
                                    f"{pointing}_dither_{i+1}_to_reference_composed_via_dither_{k+1}_and_dither_{j+1}",
                                    show=False,
                                    custom_title=f"Composed Alignment from Dither {i+1} to reference through dither {k+1} and dither {j+1}",
                                )
 
                                recovered = True
                                break
                            if recovered:
                                break
 
                    if not science_tables[pointing][i][6]:
                        self.logger.warning(
                            f"After attempting recovery, the alignment for pointing {pointing}, dither {i+1} is still failed."
                        )
                    else:
                        self.logger.info(
                            f"Successfully recovered alignment for pointing {pointing}, dither {i+1}."
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
                    _,
                    _,
                ) = science_tables[pointing][i]
 
                if not success:
                    self.logger.warning(
                        f"Alignment for pointing {pointing}, dither {i+1} failed. Writing None values."
                    )
 
                    dRA = dRA_sigma = dDEC = dDEC_sigma = rotation_angle = (
                        rotation_error
                    ) = None
 
                else:
 
                    # convert offsets and offset errors from detector pixels to sky coordinates (degrees)
 
                    dRA, dDEC = detector_translation_to_sky_translation(
                        dither_frame_wcs[i], tx, ty, logger=self.logger
                    )
 
                    dRA_sigma, dDEC_sigma = detector_translation_to_sky_translation(
                        dither_frame_wcs[i], tx_error, ty_error
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
 
        # write the final results to a FITS table
        write_final_results(output_table, self.output_dir, logger=self.logger)