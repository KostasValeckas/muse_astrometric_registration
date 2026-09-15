from abc import ABC, abstractmethod
import os
import astroalign as aa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import os

class AligmentMethod(ABC):
    """
    Abstract base class for all alignment methods.

    This class defines the interface that all alignment methods must implement.
    """

    def __init__(self, name, version, QA_path= None , logger=None):
        """
        Initialize the alignment method.

        Parameters
        ----------
        name : str
            Name of the alignment method.

        version :
            Version of the alignment method.

        QA_path : str or Path
            Path to save Quality Assurance (QA) plots.

        logger: Optional
            Logger object for logging messages.
        """

        self.name = name
        self.version = version
        self.logger = logger

        if QA_path is None:
            self.QA_path = os.path.join(os.getcwd(), "aligment_QA")
            # Create the QA directory if it doesn't exist
            if not os.path.exists(self.QA_path):
                os.makedirs(self.QA_path)
        else:
            self.QA_path = QA_path

        # this is the return value when the alignment fails
        self.FAILED_RETURN = (None, None), None, (None, None), None, False

    def check_input(self, reference_frame, input_frame):
        """
        Check if the reference and input frames are compatible for alignment.

        Checks for shape, type and nan_values.

        Parameters
        ----------
        reference_frame : np.2darray
            The reference image to which the target image will be aligned.
            Must be the same shape and scale as the input_frame.

        input_frame : np.2darray
            The target image that needs to be aligned to the reference image.

        Raises
        ------
        bool
            True if input is valid, False otherwise.
        """

        if reference_frame.shape != input_frame.shape:
            if self.logger:
                self.logger.error("Shape mismatch between reference and input frames.")

            return False

        if not isinstance(reference_frame, type(input_frame)):
            if self.logger:
                self.logger.error("Type mismatch between reference and input frames.")

            return False

        if np.isnan(reference_frame).any() or np.isnan(input_frame).any():
            if self.logger:
                self.logger.error("NaN values found in reference or input frames.")

            return False

        if self.logger:
            self.logger.info("Input frames are valid for alignment.")

        return True

    @abstractmethod
    def run(
        self,
        reference_frame,
        input_frame,
        obs_name,
        initial_guess_translation=(None, None),
        initial_guess_rotation=None,
        save_QA=True,
        show_QA=False,
        **kwargs
    ):
        """
        Align the target image to the reference image.

        Parameters
        ----------

        reference_frame : np.2darray
            The reference image to which the target image will be aligned.
            Must be the same shape and scale as the input_frame.

        input_frame : np.2darray
            The target image that needs to be aligned to the reference image.

        obs_name: str
            Name of the observation, used for saving QA plots and logging.

        initial_guess_translation : tuple, optional
            Initial guess for the translation (x, y) in pixels.

        initial_guess_rotation : float, optional
            Initial guess for the rotation in degrees.

        **kwargs : dict
            Additional keyword arguments specific to the alignment method.

        Returns
        -------
        offsets: tuple
            The calculated offsets (x_offset, y_offset) in pixels.

        rotation: float
            The calculated rotation in degrees.

        sigma_translation: tuple
            The uncertainty in the calculated translation (sigma_x, sigma_y) in pixels.

        sigma_rotation: float
            The uncertainty in the calculated rotation in degrees.

        success: bool
            True if the alignment was successful, False otherwise.
        """
        pass


class PointSources(AligmentMethod):

    def __init__(self, logger=None):
        super().__init__("PointSources", "0.0.0", logger=logger)

        if self.logger:
            self.logger.info(f"Initialized {self.name} alignment method, version {self.version}.")

    def run(
        self,
        reference_frame,
        input_frame,
        obs_name,
        save_QA=True,
        show_QA=False,
        initial_guess_translation=(None, None),
        initial_guess_rotation=None,
        **kwargs
    ):
        """
        Align the target image to the reference image using point sources.

        Parameters
        ----------
        reference_frame : np.2darray
            The reference image to which the target image will be aligned.
            Must be the same shape and scale as the input_frame.

        input_frame : np.2darray
            The target image that needs to be aligned to the reference image.

        obs_name: str
            Name of the observation, used for saving QA plots and logging.

        save_QA : bool, optional
            Whether to save Quality Assurance (QA) plots. Default is True.

        show_QA : bool, optional
            Whether to display Quality Assurance (QA) plots. Default is False.

        initial_guess_translation : tuple, optional
            Initial guess for the translation (x, y) in pixels.

        initial_guess_rotation : float, optional
            Initial guess for the rotation in degrees.

        **kwargs : dict
            Additional keyword arguments specific to the alignment method.

        Returns
        -------
        offsets: tuple
            The calculated offsets (x_offset, y_offset) in pixels.

        rotation: float
            The calculated rotation in degrees.

        sigma_translation: tuple
            The uncertainty in the calculated translation (sigma_x, sigma_y) in pixels.

        sigma_rotation: float
            The uncertainty in the calculated rotation in degrees.

        success: bool
            True if the alignment was successful, False otherwise.
        """

        good_data = self.check_input(reference_frame, input_frame)

        if not good_data:
            if self.logger:
                self.logger.error("Invalid input data for alignment.")

            return self.FAILED_RETURN

        # we find matching sources in both images using astroalign

        _, (source_list, target_list) = aa.find_transform(input_frame, reference_frame)

        if self.logger:
            self.logger.info(
                f"Found {len(source_list)} matching sources for alignment."
            )
            self.logger.info(
                f"Source coordinates:\n {source_list}"
            )
            self.logger.info(
                f"Target coordinates:\n {target_list}"
            )


        if save_QA:
            fig = plt.figure(figsize=(10, 5), facecolor="black")

            ax1 = plt.subplot(1, 2, 1)
            ax1.set_facecolor("black")
            ax1.imshow(reference_frame, cmap="gray")
            ax1.scatter(target_list[:, 0], target_list[:, 1], c="r", marker="x")
            ax1.set_title("Reference Frame with Matched Sources", color="white")
            ax1.tick_params(colors="white")

            ax2 = plt.subplot(1, 2, 2)
            ax2.set_facecolor("black")
            ax2.imshow(input_frame, cmap="gray")
            ax2.scatter(source_list[:, 0], source_list[:, 1], c="b", marker="o")
            ax2.set_title("Input Frame with Matched Sources", color="white")
            ax2.tick_params(colors="white")

            plt.title(f"Matched Sources for Observation: {obs_name}", color="white")

            plt.tight_layout()

            for (xt, yt), (xs, ys) in zip(target_list, source_list):
                con = ConnectionPatch(
                    xyA=(xt, yt), coordsA=ax1.transData,
                    xyB=(xs, ys), coordsB=ax2.transData,
                    axesA=ax1, axesB=ax2,
                    color="yellow", linewidth=0.5, alpha=0.6,
                    clip_on=False
                )
                fig.add_artist(con)

            plt.savefig(f"{self.QA_path}/matched_sources_{obs_name}.png", facecolor="black")
            if self.logger:
                self.logger.info(f"Saved QA plot of matched sources to {self.QA_path}/matched_sources.png")
            if show_QA:
                plt.show()
            else:
                plt.close()