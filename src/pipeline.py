from logger import get_logger

import argparse
import pathlib

from methods import PointSources

from io import read_fits_image, extact_image_wcs
from geometry import reproject_image_to_target_wcs


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
        reference_WCS = extact_image_wcs(ref_file_path, logger=self.logger)

        reference_frame = read_fits_image(ref_file_path, logger=self.logger)


        for input_file in self.input_files:
            self.logger.info(f"Processing input file: {input_file}")

            # get the input image WCS and data
            input_WCS = extact_image_wcs(input_file, logger=self.logger, wcs_extension=1)

            input_frame = read_fits_image(input_file, logger=self.logger, data_extension=1)

            reprojected_input_frame = reproject_image_to_target_wcs(
                input_frame, input_WCS, reference_WCS, logger=self.logger
            )


            

            