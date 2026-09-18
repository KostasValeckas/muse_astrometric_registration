from logger import get_logger

import argparse
import pathlib

from methods import PointSources

from IO import read_fits_image, extract_image_wcs
from geometry import reproject_image_to_target_wcs
from filters import remove_nans



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

            # first frame will be the reference for the dithers

            reference_file = pathlib.Path(self.input_dir) / exposures[0][2]
            self.logger.info(f"Reference file for this pointing: {reference_file}")

            # get the reference image WCS and data
            reference_WCS = extract_image_wcs(reference_file, logger=self.logger, wcs_extension=1)

            reference_frame = read_fits_image(reference_file, logger=self.logger, data_extension=1)



            for i, exposure in enumerate(exposures):

                if i == 0:
                    self.logger.info(f"Skipping reference exposure: {exposure[2]}")
                    continue

                input_file = pathlib.Path(self.input_dir) / exposure[2]
                self.logger.info(f"Processing exposure: {input_file}")

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
                    success,
                    tx,
                    ty,
                    rotation_angle,
                    tx_error,
                    ty_error,
                    rotation_error,
                )

                dithers_align_table.append(return_tuple)


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


            


            

            