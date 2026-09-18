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

    def __init__(self, input_files, reference_file, output_dir):

        self.logger = get_logger(output_dir)

        self.input_files = input_files
        self.reference_file = reference_file
        self.output_dir = output_dir

        self.point_sources_aligner = PointSources(logger=self.logger)

    def run(self, config_file, ref_file_path, output_dir):
        """
        Run the alignment pipeline.

        For now simple.
        """


        # output_table_format:
        # file, ra_offset, dec_offset, rotation_angle, ra_error, dec_error, rotation_error

        output_table = []

        # TODO: for now simple to start testing

        # get the reference image WCS and data
        reference_WCS = extract_image_wcs(ref_file_path, logger=self.logger)

        reference_frame = read_fits_image(ref_file_path, logger=self.logger)


        for input_file in self.input_files:
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


            


            

            