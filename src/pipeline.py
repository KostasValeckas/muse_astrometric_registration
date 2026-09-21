from logger import get_logger

import argparse
import pathlib

from methods import PointSources

from IO import read_fits_image, extract_image_wcs
from geometry import reproject_image_to_target_wcs, transformation_matrix, compose_transformations, matrix_to_parameters
from filters import remove_nans
import numpy as np
import matplotlib.pyplot as plt





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


    def align_dithers(self, dither_set, wcs_set, method, obs_name = None):
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
        succes_tabe = np.full((n_dithers, n_dithers), False)

        min_height = min([dither.shape[0] for dither in dither_set])
        min_width = min([dither.shape[1] for dither in dither_set])

        for i, dither_1 in enumerate(dither_set):

            for j, dither_2 in enumerate(dither_set):

                wcs_1 = wcs_set[i]
                wcs_2 = wcs_set[j]

                # reproject dither 1 to dither 2 WCS
                reprojected_dither_1, _ = reproject_image_to_target_wcs(
                    dither_1, wcs_1, wcs_2, logger=self.logger, shape_out=(min_height, min_width)
                )

                reprojected_dither_2, _ = reproject_image_to_target_wcs(
                    dither_2, wcs_2, wcs_2, logger=self.logger, shape_out=(min_height, min_width)
                )

                

                # remove NaN values from the reprojected dither 1 and dither 2
                reprojected_dither_1 = remove_nans(
                    reprojected_dither_1, fill_value=0, logger=self.logger
                )

                reprojected_dither_2 = remove_nans(
                    reprojected_dither_2, fill_value=0, logger=self.logger
                )


                # just testing

                print("center reprojected: ", reprojected_dither_1.shape[0]//2, reprojected_dither_1.shape[1]//2)
                print("center dither 2: ", dither_2.shape[0]//2, dither_2.shape[1]//2)

                print("center reprojected dither 2: ", reprojected_dither_2.shape[0]//2, reprojected_dither_2.shape[1]//2)

                (tx, ty), rotation_angle, (tx_error, ty_error), rotation_error, success = method.run(
                    reprojected_dither_1, reprojected_dither_2, f"{obs_name}_dither_{i+1}_to_dither_{j+1}"
                )  


                x_return_table[i, j] = tx
                y_return_table[i, j] = ty
                rotation_return_table[i, j] = rotation_angle

                succes_tabe[i, j] = success

        # For not just testing: construct the path from image 1 to 4 using 1 - 2, 2 - 3, 3 - 4 and check if the total offset is consistent with the direct 1 - 4 offset.

        center = (min_height // 2, min_width // 2)

        T_1_4 = transformation_matrix(x_return_table[0, 3], y_return_table[0, 3], rotation_return_table[0, 3], center)
        T_1_2 = transformation_matrix(x_return_table[0, 1], y_return_table[0, 1], rotation_return_table[0, 1], center)
        T_2_3 = transformation_matrix(x_return_table[1, 2], y_return_table[1, 2], rotation_return_table[1, 2], center)
        T_3_4 = transformation_matrix(x_return_table[2, 3], y_return_table[2, 3], rotation_return_table[2, 3], center)

        T_1_3_composed = compose_transformations(T_1_2, T_2_3)

        T_1_4_composed = compose_transformations(T_1_3_composed, T_3_4)

        tx_composed, ty_composed, rotation_angle_composed = matrix_to_parameters(T_1_4_composed, center)

        tx_direct, ty_direct, rotation_angle_direct = matrix_to_parameters(T_1_4, center)

        print("Direct 1-4: ", tx_direct, ty_direct, rotation_angle_direct)
        print("Composed 1-4: ", tx_composed, ty_composed, rotation_angle_composed)

        exit()

        plt.figure(figsize=(10, 8))
        plt.imshow(succes_tabe, cmap='Greens', interpolation='nearest')
        plt.colorbar(label='Alignment Success')
        plt.title('Alignment Success Between Dithers')
        plt.xlabel('Dither Index')
        plt.ylabel('Dither Index')
        plt.xticks(range(n_dithers), [f'Dither {i+1}' for i in range(n_dithers)])
        plt.yticks(range(n_dithers), [f'Dither {i+1}' for i in range(n_dithers)])
        for i in range(n_dithers):
            for j in range(n_dithers):
                plt.text(j, i, f'{succes_tabe[i, j]}', ha='center', va='center', color='black', fontsize=8)
        plt.show()

        plt.figure(figsize=(10, 8))
        plt.imshow(x_return_table, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='X Offset')
        plt.title('X Offsets Between Dithers')
        plt.xlabel('Dither Index')
        plt.ylabel('Dither Index')
        plt.xticks(range(n_dithers), [f'Dither {i+1}' for i in range(n_dithers)])
        plt.yticks(range(n_dithers), [f'Dither {i+1}' for i in range(n_dithers)])
        for i in range(n_dithers):
            for j in range(n_dithers):
                plt.text(j, i, f'{x_return_table[i, j]:.3f}', ha='center', va='center', color='white', fontsize=8)
        plt.show()

        plt.figure(figsize=(10, 8))
        plt.imshow(y_return_table, cmap='plasma', interpolation='nearest')
        plt.colorbar(label='Y Offset')
        plt.title('Y Offsets Between Dithers')
        plt.xlabel('Dither Index')
        plt.ylabel('Dither Index')
        plt.xticks(range(n_dithers), [f'Dither {i+1}' for i in range(n_dithers)])
        plt.yticks(range(n_dithers), [f'Dither {i+1}' for i in range(n_dithers)])
        for i in range(n_dithers):
            for j in range(n_dithers):
                plt.text(j, i, f'{y_return_table[i, j]:.3f}', ha='center', va='center', color='white', fontsize=8)
        plt.show()

        plt.figure(figsize=(10, 8))
        plt.imshow(rotation_return_table, cmap='cividis', interpolation='nearest')
        plt.colorbar(label='Rotation Angle')
        plt.title('Rotation Angles Between Dithers')
        plt.xlabel('Dither Index')
        plt.ylabel('Dither Index')
        plt.xticks(range(n_dithers), [f'Dither {i+1}' for i in range(n_dithers)])
        plt.yticks(range(n_dithers), [f'Dither {i+1}' for i in range(n_dithers)])
        for i in range(n_dithers):
            for j in range(n_dithers):
                plt.text(j, i, f'{rotation_return_table[i, j]:.3f}', ha='center', va='center', color='white', fontsize=8)
        plt.show()

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
        QA_dir_dithers.mkdir(parents=True, exist_ok=True)# Create the QA directory for the dithers
        
        # Create the QA directory for the dithers
        self.point_sources_aligner = PointSources(QA_path=QA_dir_dithers, logger=self.logger)


        for pointing, exposures in self.config_dict.items():
            self.logger.info(f"Processing pointing: {pointing}")

            dither_frame_arrays = []
            dither_frame_wcs = []

            # read in the dither frames for this pointing
            dither_frames = [pathlib.Path(self.input_dir) / exposure[2] for exposure in exposures]

            for dither_frame in dither_frames:
                self.logger.info(f"Reading dither frame: {dither_frame}")

                # get the input image WCS and data
                input_WCS = extract_image_wcs(dither_frame, logger=self.logger, wcs_extension=1)

                input_frame = read_fits_image(dither_frame, logger=self.logger, data_extension=1)

                # remove the NaN values from the input frame:
                input_frame = remove_nans(
                    input_frame, fill_value=0, logger=self.logger
                )

                dither_frame_arrays.append(input_frame)
                dither_frame_wcs.append(input_WCS)

            x_table, y_table, rotation_table = self.align_dithers(dither_frame_arrays, dither_frame_wcs, self.point_sources_aligner, obs_name=pointing)


        exit()

            
            


        # get the reference image WCS and data
        reference_WCS = extract_image_wcs(self.reference_file, logger=self.logger)

        reference_frame = read_fits_image(self.reference_file, logger=self.logger)

        QA_dir_science = pathlib.Path(self.output_dir) / "aligment_QA" / "science"
        QA_dir_science.mkdir(parents=True, exist_ok=True)

        self.point_sources_aligner = PointSources(QA_path=QA_dir_science, logger=self.logger)


        input_files = [pathlib.Path(self.input_dir) / exposure[2] for pointing, exposures in self.config_dict.items() for exposure in exposures]


        for input_file in input_files:
            self.logger.info(f"Processing input file: {input_file}")

            # get the input image WCS and data
            input_WCS = extract_image_wcs(input_file, logger=self.logger, wcs_extension=1)

            input_frame = read_fits_image(input_file, logger=self.logger, data_extension=1)

            reprojected_reference_frame, _ = reproject_image_to_target_wcs(
                reference_frame, reference_WCS, input_WCS, logger=self.logger
            )

            # remove the NaN values from the reprojected reference frame and input:

            reprojected_reference_frame = remove_nans(
                reprojected_reference_frame, fill_value=0, logger=self.logger
            )

            input_frame = remove_nans(
                input_frame, fill_value=0, logger=self.logger
            )

            # run the alignment

            # for the filename, remove any directory path and just keep the filename
            input_filename = pathlib.Path(input_file).name

            (tx, ty), rotation_angle, (tx_error, ty_error), rotation_error, success = self.point_sources_aligner.run(
                reprojected_reference_frame, input_frame, input_filename
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


            


            

            