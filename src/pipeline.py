from logger import get_logger

import argparse
import pathlib

from methods import PointSources

from IO import read_fits_image, extract_image_wcs
from geometry import (
    reproject_image_to_target_wcs,
    transformation_matrix,
    compose_transformations,
    matrix_to_parameters,
)
from filters import remove_nans
import numpy as np
import matplotlib.pyplot as plt

from enum import Enum

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

    def _cross_ref_dither_table(self, sucess_table, table, threshold_frac=0.1, minimum_value = 0.1):
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
                self.logger.info(f"Values too small to check consistency between dither {a+1} and dither {b+1} ({val_ab} vs. {val_ba}). Skipping consistency check for this pair...")
                continue




            if (abs(val_ab - val_ba) / max(abs(val_ab), abs(val_ba))) > threshold_frac and val_ab != np.nan and val_ba != np.nan:

                self.logger.warning(
                    f"Alignment between dither {a+1} and dither {b+1} is inconsistent ({val_ab} vs. {val_ba}). Replacing with NaN values."
                )
                sucess_table[a, b] = 0.7
                sucess_table[b, a] = 0.7

        return sucess_table, table

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
    ):
        """
        Helper plotting medthod to `align_dithers`.
        """

        _, axes = plt.subplots(2, 4, figsize=(22, 12))

        # ---------------------------------------------------------
        # Dither images
        # ---------------------------------------------------------
        for i in range(min(n_dithers, 4)):
            ax = axes[0, i]
            _ = ax.imshow(
                dither_set[i],
                cmap="gray",
                origin="lower",
                interpolation="nearest",
                vmin=np.percentile(dither_set[i], 5),
                vmax=np.percentile(dither_set[i], 95),
            )
            ax.set_title(f"Dither {i + 1}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")

        # ---------------------------------------------------------
        # 1. Alignment Status
        # ---------------------------------------------------------
        ax = axes[1, 0]

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
        ax = axes[1, 1]

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
        ax = axes[1, 2]

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
        ax = axes[1, 3]

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
        # Layout
        # ---------------------------------------------------------
        if title:
            plt.suptitle(title, fontsize=16)
        else:
            plt.suptitle(f"Dither Alignment QA for {obs_title}", fontsize=16)

        plt.tight_layout()
        if save:
            if save_path is None:
                save_path = pathlib.Path(self.output_dir) / "aligment_QA" / "dithers"
                save_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path / f"{obs_title}_dither_alignment.png")
        if show:
            plt.show()
        else:
            plt.close()

    def align_dithers(self, dither_set, wcs_set, method, obs_name=None):
        """
        Aligning all dithers relative to each other (all combinations).

        This gives more precise measurements and detects any corrupt frames.

        Parameters
        ----------

        dither_set : list
            List of dither frames to align.

        wcs_set : list
            List of WCS objects corresponding to the dither frames.

        method: AlignmentMethod object

        Returns
        -------

        x_table : np.ndarray
            Table of x offsets for each dither frame.

            ```
            +--------+-------+-------+-------+-------+
            | dither | dit_1 | dit_2 | dit_3 | dit_4 |
            +--------+-------+-------+-------+-------+
            | dit_1  | x_1_1 | x_1_2 | x_1_3 | x_1_4 |
            | dit_2  | x_2_1 | x_2_2 | x_2_3 | x_2_4 |
            | dit_3  | x_3_1 | x_3_2 | x_3_3 | x_3_4 |
            | dit_4  | x_4_1 | x_4_2 | x_4_3 | x_4_4 |
            +--------+-------+-------+-------+-------+
            ```

        y_table : np.ndarray
            Table of y offsets for each dither frame.

            ```
            +--------+-------+-------+-------+-------+
            | dither | dit_1 | dit_2 | dit_3 | dit_4 |
            +--------+-------+-------+-------+-------+
            | dit_1  | y_1_1 | y_1_2 | y_1_3 | y_1_4 |
            | dit_2  | y_2_1 | y_2_2 | y_2_3 | y_2_4 |
            | dit_3  | y_3_1 | y_3_2 | y_3_3 | y_3_4 |
            | dit_4  | y_4_1 | y_4_2 | y_4_3 | y_4_4 |
            +--------+-------+-------+-------+-------+
            ```

        rotation_table : np.ndarray
            Table of rotation angles for each dither frame.

            ```
            +--------+-------+-------+-------+-------+
            | dither | dit_1 | dit_2 | dit_3 | dit_4 |
            +--------+-------+-------+-------+-------+
            | dit_1  | r_1_1 | r_1_2 | r_1_3 | r_1_4 |
            | dit_2  | r_2_1 | r_2_2 | r_2_3 | r_2_4 |
            | dit_3  | r_3_1 | r_3_2 | r_3_3 | r_3_4 |
            | dit_4  | r_4_1 | r_4_2 | r_4_3 | r_4_4 |
            +--------+-------+-------+-------+-------+
            ```

        """

        n_dithers = len(dither_set)

        x_return_table = np.full((n_dithers, n_dithers), np.nan)
        y_return_table = np.full((n_dithers, n_dithers), np.nan)
        rotation_return_table = np.full((n_dithers, n_dithers), np.nan)
        succes_table = np.full((n_dithers, n_dithers), 0, dtype=float)

        min_height = min([dither.shape[0] for dither in dither_set])
        min_width = min([dither.shape[1] for dither in dither_set])

        # use reprojection to crop all dithers to the same size and WCS
        target_wcs = wcs_set[0]

        for i, dither in enumerate(dither_set):
            reprojected_dither, _ = reproject_image_to_target_wcs(
                dither, wcs_set[i], target_wcs, shape_out=(min_height, min_width)
            )

            # remove the NaN values from the reprojected dither:
            reprojected_dither = remove_nans(
                reprojected_dither, fill_value=0, logger=self.logger
            )

            dither_set[i] = reprojected_dither

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

                succes_table[i, j] = success

        # check for consistency

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

        self._plot_dither_alignment(
            dither_set,
            n_dithers,
            succes_table,
            x_return_table,
            y_return_table,
            rotation_return_table,
            obs_title=obs_name + "_prior_compositions_",
            save_path=method.QA_path,
        )

        # Now we try to map out any missed paths
        # Algorithm:
        # Loop through every frame:
        # First, see if direct path exists. If yes, continue to next frame.
        # If not, loop through all other frames and see if a path exists through them.
        # Start by looking through the shortest paths first (i.e. 1 intermediate frame, then 2, etc.)

        for i in range(n_dithers):
            for j in range(n_dithers):
                if i == j:
                    continue

                success = False
                if not succes_table[i, j] == 1:
                    # direct path does not exist, look for intermediate frames

                    # first check if opposite pair exists - if so, replace with negative of that:

                    if succes_table[j, i] == 1:
                        x_return_table[i, j] = -x_return_table[j, i]
                        y_return_table[i, j] = -y_return_table[j, i]
                        rotation_return_table[i, j] = -rotation_return_table[j, i]
                        succes_table[i, j] = 0.5

                        # plot QA plot:

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

                    for k in range(n_dithers):
                        if k == i or k == j:
                            continue

                        # one intermediate frame search
                        if succes_table[i, k] == 1 and succes_table[k, j] == 1:
                            # found a path through k
                            self.logger.info(
                                f"Found a path from dither {i+1} to dither {j+1} through dither {k+1}."
                            )
                            # compose the transformations
                            T_i_k = transformation_matrix(
                                x_return_table[i, k],
                                y_return_table[i, k],
                                rotation_return_table[i, k],
                                (min_height // 2, min_width // 2),
                            )
                            T_k_j = transformation_matrix(
                                x_return_table[k, j],
                                y_return_table[k, j],
                                rotation_return_table[k, j],
                                (min_height // 2, min_width // 2),
                            )
                            T_i_j_composed = compose_transformations(T_i_k, T_k_j)
                            tx_composed, ty_composed, rotation_angle_composed = (
                                matrix_to_parameters(
                                    T_i_j_composed, (min_height // 2, min_width // 2)
                                )
                            )
                            x_return_table[i, j] = tx_composed
                            y_return_table[i, j] = ty_composed
                            rotation_return_table[i, j] = rotation_angle_composed
                            succes_table[i, j] = 0.5

                            success = True

                            # the QA plot:
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
                                    # found a path through k and l
                                    self.logger.info(
                                        f"Found a path from dither {i+1} to dither {j+1} through dithers {k+1} and {l+1}."
                                    )
                                    # compose the transformations
                                    T_i_k = transformation_matrix(
                                        x_return_table[i, k],
                                        y_return_table[i, k],
                                        rotation_return_table[i, k],
                                        (min_height // 2, min_width // 2),
                                    )
                                    T_k_l = transformation_matrix(
                                        x_return_table[k, l],
                                        y_return_table[k, l],
                                        rotation_return_table[k, l],
                                        (min_height // 2, min_width // 2),
                                    )
                                    T_l_j = transformation_matrix(
                                        x_return_table[l, j],
                                        y_return_table[l, j],
                                        rotation_return_table[l, j],
                                        (min_height // 2, min_width // 2),
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
                                        (min_height // 2, min_width // 2),
                                    )
                                    x_return_table[i, j] = tx_composed
                                    y_return_table[i, j] = ty_composed
                                    rotation_return_table[i, j] = (
                                        rotation_angle_composed
                                    )
                                    succes_table[i, j] = 0.5

                                    plt.imshow(succes_table)
                                    plt.show()

                                    success = True

                                    # the QA plot:
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

        # check for consistency

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

        self._plot_dither_alignment(
            dither_set,
            n_dithers,
            succes_table,
            x_return_table,
            y_return_table,
            rotation_return_table,
            obs_title=obs_name + "_after_compositions_",
            save_path=method.QA_path,
        )

        return x_return_table, y_return_table, rotation_return_table

    def run(self):
        """
        Run the alignment pipeline.

        For now simple.
        """

        # output_table_format:
        # file, ra_offset, dec_offset, rotation_angle, ra_error, dec_error, rotation_error

        output_table = []

        # TODO: for now simple to start testing

        # algorithm: align first the dither groups to the first image, and then
        # to external referrence

        self.logger.info("Starting alignment pipeline...")

        self.logger.info(f"Reference file: {self.reference_file}")

        self.logger.info(f"Aligning dithers...")

        dithers_align_table = []

        # Create the QA directory for the dithers
        QA_dir_dithers = pathlib.Path(self.output_dir) / "aligment_QA" / "dithers"
        # Create the QA directory for the dithers
        QA_dir_dithers.mkdir(
            parents=True, exist_ok=True
        )  # Create the QA directory for the dithers

        # Create the QA directory for the dithers
        self.point_sources_aligner = PointSources(
            QA_path=QA_dir_dithers, logger=self.logger
        )

        for pointing, exposures in self.config_dict.items():
            self.logger.info(f"Processing pointing: {pointing}")

            dither_frame_arrays = []
            dither_frame_wcs = []

            # read in the dither frames for this pointing
            dither_frames = [
                pathlib.Path(self.input_dir) / exposure[2] for exposure in exposures
            ]

            for dither_frame in dither_frames:
                self.logger.info(f"Reading dither frame: {dither_frame}")

                # get the input image WCS and data
                input_WCS = extract_image_wcs(
                    dither_frame, logger=self.logger, wcs_extension=1
                )

                input_frame = read_fits_image(
                    dither_frame, logger=self.logger, data_extension=1
                )

                # remove the NaN values from the input frame:
                input_frame = remove_nans(input_frame, fill_value=0, logger=self.logger)

                dither_frame_arrays.append(input_frame)
                dither_frame_wcs.append(input_WCS)

            x_table, y_table, rotation_table = self.align_dithers(
                dither_frame_arrays,
                dither_frame_wcs,
                self.point_sources_aligner,
                obs_name=pointing,
            )

        exit()

        # get the reference image WCS and data
        reference_WCS = extract_image_wcs(self.reference_file, logger=self.logger)

        reference_frame = read_fits_image(self.reference_file, logger=self.logger)

        QA_dir_science = pathlib.Path(self.output_dir) / "aligment_QA" / "science"
        QA_dir_science.mkdir(parents=True, exist_ok=True)

        self.point_sources_aligner = PointSources(
            QA_path=QA_dir_science, logger=self.logger
        )

        input_files = [
            pathlib.Path(self.input_dir) / exposure[2]
            for pointing, exposures in self.config_dict.items()
            for exposure in exposures
        ]

        for input_file in input_files:
            self.logger.info(f"Processing input file: {input_file}")

            # get the input image WCS and data
            input_WCS = extract_image_wcs(
                input_file, logger=self.logger, wcs_extension=1
            )

            input_frame = read_fits_image(
                input_file, logger=self.logger, data_extension=1
            )

            reprojected_reference_frame, _ = reproject_image_to_target_wcs(
                reference_frame, reference_WCS, input_WCS, logger=self.logger
            )

            # remove the NaN values from the reprojected reference frame and input:

            reprojected_reference_frame = remove_nans(
                reprojected_reference_frame, fill_value=0, logger=self.logger
            )

            input_frame = remove_nans(input_frame, fill_value=0, logger=self.logger)

            # run the alignment

            # for the filename, remove any directory path and just keep the filename
            input_filename = pathlib.Path(input_file).name

            (tx, ty), rotation_angle, (tx_error, ty_error), rotation_error, success = (
                self.point_sources_aligner.run(
                    reprojected_reference_frame, input_frame, input_filename
                )
            )

            return_tuple = (
                input_filename,
                tx,
                ty,
                rotation_angle,
                tx_error,
                ty_error,
                rotation_error,
            )

            output_table.append(return_tuple)

        self.logger.info("Alignment pipeline completed.")
        self.logger.info("Output table:")
        self.logger.info(
            "\nfile, ra_offset, dec_offset, rotation_angle, ra_error, dec_error, rotation_error"
        )
        for row in output_table:
            self.logger.info(", ".join(map(str, row)))
