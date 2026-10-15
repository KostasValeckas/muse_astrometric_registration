from logger import get_logger

import pathlib

from methods import PointSources, NoAlignment

from IO import read_fits_image, extract_image_wcs, make_table_entry, write_final_results
from geometry import (
    reproject_image_to_target_wcs,
    transformation_matrix,
    compose_transformations,
    matrix_to_parameters,
    inverse_transformation,
    detector_translation_to_sky_translation,
    reproject_dithers
)
from filters import remove_nans, remove_bright_sources_simple, mask_zero_as_bpm
import numpy as np
import matplotlib.pyplot as plt

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

# TODO: for now, hardcoded to 4 dithers pr. pointing

# for plotting
STATUS_COLORS = {
    0: "red",
    1: "green",
    0.5: "yellow",
    0.7: "orange",
}  # failed  # success  # composed # inconsistent

# create custom color bar for the status plot
from matplotlib.colors import ListedColormap, BoundaryNorm

cmap = ListedColormap(
    [STATUS_COLORS[0], STATUS_COLORS[0.5], STATUS_COLORS[0.7], STATUS_COLORS[1]]
)
norm = BoundaryNorm([0, 0.5, 0.7, 1, 1.5], cmap.N)

class AlignmentPipeline:
    """
    Class to run the alignment pipeline.
    """

    def __init__(self, config_dict, input_dir, reference_file, output_dir):

        self.logger = get_logger(output_dir)

        self.config_dict = config_dict
        self.input_dir = input_dir
        self.reference_file = reference_file
        self.output_dir = output_dir

        self.output_table_entries = config_dict["headers"]

    def _cross_ref_dither_table(
        self, sucess_table, table, threshold_frac=0.1, minimum_value=0.1
    ):
        """
        Helper method to `align_dithers`. Opposite values
        (for example x_offset 1 -> 4 and x_offset 4 -> 1) should be the same up
        to threshold. If now, replace with nans and and adjust sucess table.

        Assumes 4 dithers.
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

    # -----------------------------------------------------------------
    # CHANGED: added `axes` param, in two flavours:
    #   - a 2x4 array-like of Axes -> full panel: dither image thumbnails
    #     (row 0) + the 4-panel numeric summary (row 1).
    #   - a 1D array-like of 4 Axes -> SUMMARY ROW ONLY (status / x / y /
    #     rotation). Use this when the dither images are already shown
    #     elsewhere (e.g. in the science-alignment panel), so we don't
    #     render the same thumbnails twice.
    #   - None (default) -> creates its own new 2x4 figure, unchanged
    #     standalone behaviour.
    # Figure-level calls (suptitle / tight_layout / savefig / show) only
    # happen when this method owns the figure -- never when it's just
    # drawing onto axes handed to it by a caller.
    # -----------------------------------------------------------------
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
        Helper plotting medthod to `align_dithers`.

        Parameters
        ----------
        axes : None, 2x4 array-like of Axes, or 1D array-like of 4 Axes
            None (default): create a standalone 2x4 figure (original
            behaviour) -- image thumbnails on row 0, numeric summary on
            row 1.
            2x4 array-like: draw the full panel onto those axes instead
            of creating a new figure.
            1D array-like of length 4: draw ONLY the numeric summary row
            (status / x offsets / y offsets / rotation) -- no image
            thumbnails. Useful when the dither images are already shown
            elsewhere in a combined report.
        summary_only : bool
            When `axes=None` and this is True, the standalone figure is
            created as 1x4 (summary row only) instead of 2x4.

        When `axes` is provided, `save`/`show`/`title` are ignored -- the
        caller owns the figure and is responsible for saving/showing it.
        """

        owns_figure = axes is None

        if owns_figure:
            if summary_only:
                fig, axes = plt.subplots(1, 4, figsize=(22, 6))
                image_axes = None
                summary_axes = axes
            else:
                fig, axes = plt.subplots(2, 4, figsize=(22, 12))
                image_axes = axes[0]
                summary_axes = axes[1]
        else:
            axes = np.asarray(axes, dtype=object)
            if axes.ndim == 2:
                fig = axes[0, 0].figure
                image_axes = axes[0]
                summary_axes = axes[1]
            else:
                fig = axes[0].figure
                image_axes = None
                summary_axes = axes

        # ---------------------------------------------------------
        # Dither images (skipped entirely in summary-only mode)
        # ---------------------------------------------------------
        if image_axes is not None:
            for i in range(min(n_dithers, 4)):
                ax = image_axes[i]
                _ = ax.imshow(
                    dither_set[i],
                    cmap="gray",
                    origin="lower",
                    interpolation="nearest",
                    vmin=np.percentile(dither_set[i].compressed(), 5),
                    vmax=np.percentile(dither_set[i].compressed(), 95),
                )
                ax.set_title(f"Dither {i + 1}")
                ax.set_xlabel("x")
                ax.set_ylabel("y")

        # ---------------------------------------------------------
        # 1. Alignment Status
        # ---------------------------------------------------------
        ax = summary_axes[0]

        _ = ax.imshow(
            succes_table,
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
        )

        ax.set_title("Alignment Status Between Dithers")
        ax.set_xlabel("Dither Index")
        ax.set_ylabel("Dither Index")

        ax.set_xticks(range(n_dithers))
        ax.set_yticks(range(n_dithers))
        ax.set_xticklabels([f"Dither {i+1}" for i in range(n_dithers)])
        ax.set_yticklabels([f"Dither {i+1}" for i in range(n_dithers)])

        for i in range(n_dithers):
            for j in range(n_dithers):
                ax.text(
                    j,
                    i,
                    (
                        "FAILED"
                        if succes_table[i, j] == 0
                        else (
                            "INCONSISTENT"
                            if succes_table[i, j] == 0.7
                            else "COMPOSED" if succes_table[i, j] == 0.5 else "OK"
                        )
                    ),
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=12,
                )

        # ---------------------------------------------------------
        # 2. X Offsets
        # ---------------------------------------------------------
        ax = summary_axes[1]

        _ = ax.imshow(
            x_return_table,
            cmap="Set1",
            interpolation="nearest",
        )

        ax.set_title("X Offsets Between Dithers")
        ax.set_xlabel("Dither Index")
        ax.set_ylabel("Dither Index")

        ax.set_xticks(range(n_dithers))
        ax.set_yticks(range(n_dithers))
        ax.set_xticklabels([f"Dither {i+1}" for i in range(n_dithers)])
        ax.set_yticklabels([f"Dither {i+1}" for i in range(n_dithers)])

        for i in range(n_dithers):
            for j in range(n_dithers):
                ax.text(
                    j,
                    i,
                    f"{x_return_table[i, j]:.3f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=12,
                )

        # ---------------------------------------------------------
        # 3. Y Offsets
        # ---------------------------------------------------------
        ax = summary_axes[2]

        _ = ax.imshow(
            y_return_table,
            cmap="Set1",
            interpolation="nearest",
        )

        ax.set_title("Y Offsets Between Dithers")
        ax.set_xlabel("Dither Index")
        ax.set_ylabel("Dither Index")

        ax.set_xticks(range(n_dithers))
        ax.set_yticks(range(n_dithers))
        ax.set_xticklabels([f"Dither {i+1}" for i in range(n_dithers)])
        ax.set_yticklabels([f"Dither {i+1}" for i in range(n_dithers)])

        for i in range(n_dithers):
            for j in range(n_dithers):
                ax.text(
                    j,
                    i,
                    f"{y_return_table[i, j]:.3f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=12,
                )

        # ---------------------------------------------------------
        # 4. Rotation Angles
        # ---------------------------------------------------------
        ax = summary_axes[3]

        _ = ax.imshow(
            rotation_return_table,
            cmap="Set1",
            interpolation="nearest",
        )

        ax.set_title("Rotation Angles Between Dithers")
        ax.set_xlabel("Dither Index")
        ax.set_ylabel("Dither Index")

        ax.set_xticks(range(n_dithers))
        ax.set_yticks(range(n_dithers))
        ax.set_xticklabels([f"Dither {i+1}" for i in range(n_dithers)])
        ax.set_yticklabels([f"Dither {i+1}" for i in range(n_dithers)])

        for i in range(n_dithers):
            for j in range(n_dithers):
                ax.text(
                    j,
                    i,
                    f"{rotation_return_table[i, j]:.3f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=12,
                )

        # ---------------------------------------------------------
        # Layout / save / show -- only when we own the figure.
        # A sub-panel embedded in someone else's combined figure
        # should never call tight_layout/savefig/show on it.
        # ---------------------------------------------------------
        if owns_figure:
            if title:
                plt.suptitle(title, fontsize=16)
            else:
                plt.suptitle(f"Dither Alignment QA for {obs_title}", fontsize=16)

            plt.tight_layout()
            if save:
                if save_path is None:
                    save_path = (
                        pathlib.Path(self.output_dir) / "aligment_QA" / "dithers"
                    )
                    save_path.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path / f"{obs_title}_dither_alignment.png")
            if show:
                plt.show()
            else:
                plt.close()

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

        succes_table, x_return_table = self._cross_ref_dither_table(
            succes_table, x_return_table, threshold_frac=0.1
        )

        self.logger.info("Checking for consistency in y alignment ...")

        succes_table, y_return_table = self._cross_ref_dither_table(
            succes_table, y_return_table, threshold_frac=0.1
        )

        self.logger.info("Checking for consistency in rotation alignment ...")

        succes_table, rotation_return_table = self._cross_ref_dither_table(
            succes_table, rotation_return_table, threshold_frac=0.1
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

        succes_table, x_return_table = self._cross_ref_dither_table(
            succes_table, x_return_table, threshold_frac=0.1
        )

        self.logger.info("Checking for consistency in y alignment ...")

        succes_table, y_return_table = self._cross_ref_dither_table(
            succes_table, y_return_table, threshold_frac=0.1
        )

        self.logger.info("Checking for consistency in rotation alignment ...")

        succes_table, rotation_return_table = self._cross_ref_dither_table(
            succes_table, rotation_return_table, threshold_frac=0.1
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

    # -----------------------------------------------------------------
    # CHANGED: added `axes` param, same pattern as above. Expects a
    # 2x5 array-like of Axes when provided (dither+reference row, and
    # the text-summary row underneath).
    # -----------------------------------------------------------------
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
        Helper method to plot the alignment of the science frames against the reference frame.

        If `axes` is provided (a 2x5 array-like of Axes), draws directly onto
        those axes instead of creating a new figure; in that case `save` is
        ignored since the caller owns the figure.

        #TODO a hard copy and carries too much info in the parameters, needs refactor
        """

        owns_figure = axes is None

        if owns_figure:
            fig, axes = plt.subplots(2, 5, figsize=(22, 10))
            fig.suptitle(
                f"Pointing {pointing} alignment against reference frame", fontsize=16
            )
        else:
            fig = axes[0, 0].figure

        # top row: dither image panels + reference image panel
        for i in range(len(dither_frames)):
            ax = axes[0, i]
            image = dither_frame_arrays[i]
            vmin = np.percentile(image.compressed(), 5)
            vmax = np.percentile(image.compressed(), 95)
            ax.imshow(
                image,
                cmap="gray",
                origin="lower",
                interpolation="nearest",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_title(f"Dither {i + 1}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.tick_params(labelbottom=False, labelleft=False)

        # take first reprojected reference frame for display (they should all be the same more or less)
        reference_frame_pointing = reprojected_reference_frames[(pointing, 0)]
        v_min_ref = np.nanmedian(reference_frame_pointing) - 3 * np.nanstd(
            reference_frame_pointing
        )
        v_max_ref = np.nanmedian(reference_frame_pointing) + 3 * np.nanstd(
            reference_frame_pointing
        )
        ref_ax = axes[0, -1]
        ref_ax.imshow(
            reference_frame_pointing,
            cmap="gray",
            origin="lower",
            interpolation="nearest",
            vmin=v_min_ref,
            vmax=v_max_ref,
        )
        ref_ax.set_title("Reference")
        ref_ax.set_xlabel("x")
        ref_ax.set_ylabel("y")
        ref_ax.tick_params(labelbottom=False, labelleft=False)

        # bottom row: summary of the measured offsets and success state for each dither
        for i in range(len(dither_frames)):
            ax = axes[1, i]
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

        summary_ax = axes[1, -1]
        summary_ax.axis("off")
        summary_ax.text(
            0.05,
            0.9,
            f"Reference alignment\nPointing {pointing}\n"
            f"{len(dither_frames)} dithers\n"
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
                f"D{i + 1}: {'OK' if bool(science_tables[pointing][i][6]) else 'FAIL'}"
                for i in range(len(dither_frames))
            ),
            ha="left",
            va="top",
            fontsize=9,
            linespacing=1.4,
        )

        if owns_figure:
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            if save:
                fig.savefig(
                    QA_dir_science / f"{pointing}_dither_to_reference_QA.png",
                    dpi=200,
                )
            if show:
                plt.show()
            else:
                plt.close(fig)

        return fig, axes

    def run(self):
        """
        Run the alignment pipeline.

        For now simple.
        """

        self.logger.info("Starting alignment pipeline...")

        self.logger.info(f"Reference file: {self.reference_file}")

        output_table = []

        self.logger.info(f"Aligning dithers...")

        # Create the QA directory for the dithers
        QA_dir_dithers = pathlib.Path(self.output_dir) / "aligment_QA" / "dithers"
        QA_dir_dithers.mkdir(parents=True, exist_ok=True)

        # and create the QA directory for the science frames
        QA_dir_science = pathlib.Path(self.output_dir) / "aligment_QA" / "science"
        QA_dir_science.mkdir(parents=True, exist_ok=True)

        # Create the QA directory for combined reports
        QA_dir_combined = pathlib.Path(self.output_dir) / "aligment_QA" / "combined"
        QA_dir_combined.mkdir(parents=True, exist_ok=True)

        self.point_sources_aligner = PointSources(
            QA_path=QA_dir_dithers, logger=self.logger
        )

        dither_tables = {}
        science_tables = {}

        reference_WCS = extract_image_wcs(self.reference_file, logger=self.logger)
        reference_frame, _ = read_fits_image(self.reference_file, logger=self.logger)
        reference_frame = remove_nans(reference_frame, fill_value=0, logger=self.logger)

        reprojected_reference_frames = {}

        for pointing, exposures in self.config_dict.items():

            # this entry is only used for bookeeping
            if pointing == "headers":
                continue

            if len(exposures) < 4:
                self.logger.warning(
                    f"Pointing {pointing} has less than 4 dithers. Skipping."
                )
                continue

            if len(exposures) > 4:
                # take the first 4 dithers only
                self.logger.warning(
                    f"Pointing {pointing} has more than 4 dithers. Taking the first 4 only."
                )
                exposures = exposures[:4]

            science_tables[pointing] = {}

            self.logger.info(f"Processing pointing: {pointing}")

            dither_frame_arrays = []
            dither_frame_wcs = []
            dither_hduls = []

            dither_frames = [
                pathlib.Path(self.input_dir) / exposure[2] for exposure in exposures
            ]

            for dither_frame in dither_frames:
                self.logger.info(f"Reading dither frame: {dither_frame}")

                input_WCS = extract_image_wcs(
                    dither_frame, logger=self.logger, wcs_extension=1
                )

                input_frame, hdul = read_fits_image(
                    dither_frame, logger=self.logger, data_extension=1
                )

                input_frame = remove_nans(input_frame, fill_value=0, logger=self.logger)

                dither_frame_arrays.append(input_frame)
                dither_hduls.append(hdul)
                dither_frame_wcs.append(input_WCS)

            self.point_sources_aligner.QA_path = QA_dir_dithers

                    # bring dithers to the same WCS and shape using reprojection
            dither_set, wcs_set = reproject_dithers(
                dither_set, wcs_set, logger=self.logger, method="interpolation"
            )


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
                self.point_sources_aligner,
                obs_name=pointing,
            )

            dither_tables[pointing] = (x_table, y_table, rotation_table, success_table)

            dither_frame_arrays = dither_set
            dither_frame_wcs = wcs_set

            self.point_sources_aligner.QA_path = QA_dir_science

            for i, dither_frame in enumerate(dither_frame_arrays):

                self.logger.info(
                    f"Aligning dither frame {dither_frame} to reference frame {self.reference_file}"
                )

                reprojected_reference_frame, _ = reproject_image_to_target_wcs(
                    reference_frame,
                    reference_WCS,
                    dither_frame_wcs[i],
                    shape_out=dither_frame_arrays[i].shape,
                )

                reprojected_reference_frame = remove_nans(
                    reprojected_reference_frame, fill_value=0, logger=self.logger
                )

                reprojected_reference_frame, bright_source_mask = (
                    remove_bright_sources_simple(
                        reprojected_reference_frame,
                        logger=self.logger,
                        mask=bright_source_mask,
                    )
                )

                # replace 0's as bad pixels
                reprojected_reference_frame, _ = mask_zero_as_bpm(
                    reprojected_reference_frame, logger=self.logger
                )

                reprojected_reference_frames[(pointing, i)] = (
                    reprojected_reference_frame
                )

                (
                    (tx, ty),
                    rotation_angle,
                    (tx_error, ty_error),
                    rotation_error,
                    success,
                ) = self.point_sources_aligner.run(
                    reprojected_reference_frame,
                    dither_frame_arrays[i],
                    f"{pointing}_dither_{i+1}_to_reference",
                )

                science_tables[pointing][i] = (
                    tx,
                    ty,
                    rotation_angle,
                    tx_error,
                    ty_error,
                    rotation_error,
                    success,
                    False,
                    i,
                )

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

                        self.point_sources_aligner.plot_difference_image(
                            reprojected_reference_frames[(pointing, i)],
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
                                    reprojected_reference_frames[(pointing, i)],
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

            # -----------------------------------------------------------------
            # CHANGED: build ONE figure up front and hand each helper the
            # slice of axes it should draw on, instead of building three
            # separate figures and trying to transplant their axes
            # afterwards (which silently produced an empty figure, since
            # Axes objects can't be moved to another Figure just by
            # reassigning array elements, and the source figures were
            # already closed via plt.close() inside the helpers anyway).
            # -----------------------------------------------------------------
            # The dither images from `align_dithers` are the same frames
            # shown in the science-alignment panel above (they've already
            # been reprojected onto a common WCS), so we don't repeat the
            # thumbnails -- only the numeric summary row (status / x / y /
            # rotation) from `_plot_dither_alignment` is needed. That means
            # the combined figure only needs 3 rows of 5:
            #   rows 0-1 -> science-vs-reference report (2x5)
            #   row 2     -> dither-to-dither numeric summary (4 panels,
            #                last column spare)
            fig_combined, axes_combined = plt.subplots(3, 5, figsize=(22, 16))
            fig_combined.suptitle(
                f"Pointing {pointing} alignment QA report", fontsize=16
            )

            # rows 0-1: science-vs-reference summary (dithers + reference,
            # then the per-dither text summary)
            self._plot_science_alignment(
                pointing,
                dither_frames,
                dither_frame_arrays,
                reprojected_reference_frames,
                science_tables,
                QA_dir_science,
                axes=axes_combined[0:2, :],
            )

            # row 2, cols 0-3: dither-to-dither numeric summary ONLY
            # (a 1D array of 4 Axes triggers summary-only mode -- no
            # duplicate dither image thumbnails)
            self._plot_dither_alignment(
                dither_set,
                len(dither_set),
                success_table,
                x_table,
                y_table,
                rotation_table,
                obs_title=pointing + "_after_compositions_",
                axes=axes_combined[2, 0:4],
            )

            # spare cell in the last column of the summary row
            axes_combined[2, -1].axis("off")

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            fig_combined.savefig(
                QA_dir_combined / f"{pointing}_combined_QA.png", dpi=200
            )

            # now make the output tuples and add to the table

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
