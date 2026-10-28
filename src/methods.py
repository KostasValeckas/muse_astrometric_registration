from abc import ABC, abstractmethod
from asyncio.log import logger
import os
import astroalign as aa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import os
from scipy.optimize import least_squares
from scipy.ndimage import shift, rotate
import numpy.fft as fft
from scipy.ndimage import maximum_filter
from geometry import reproject_image_to_target_wcs, make_window_wcs
from skimage.transform import warp_polar
from skimage.registration import phase_cross_correlation
from filters import (
    remove_nans,
    remove_bright_sources_simple,
    estimate_nan_window,
    dog_passband,
    dog,
    polar_rings,
    radial_window,
    tukey2d,
)
from numpy.fft import rfft, irfft, rfftfreq, fft2, ifft2, fftfreq
from scipy.ndimage import shift as ndi_shift
from scipy.optimize import minimize
import sys
from contextlib import contextmanager

from skimage.filters import window, difference_of_gaussians

from scipy.fft import fft2, fftshift


import spacepylot.alignment as align

import spacepylot.plotting as pl
from spacepylot.alignment_utilities import TranslationTransform
import logging


# plot styling
plt.rcParams.update(
    {
        "figure.facecolor": "black",
        "axes.facecolor": "black",
        "axes.labelcolor": "white",
        "axes.titlecolor": "white",
        "text.color": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "legend.facecolor": "black",
        "legend.edgecolor": "white",
        "savefig.facecolor": "black",
        "savefig.edgecolor": "black",
    }
)


# this is used to redirect printing from external methods to the logger, so that we can capture all output in the log file
@contextmanager
def redirect_stdout_to_logger(logger, level=logging.INFO):
    class LoggerWriter:
        def write(self, message):
            message = message.rstrip()
            if message:
                logger.log(level, message)

        def flush(self):
            pass

    original_stdout = sys.stdout
    sys.stdout = LoggerWriter()
    try:
        yield
    finally:
        sys.stdout = original_stdout


class AligmentMethod(ABC):
    """
    Abstract base class for all alignment methods.

    This class defines the interface that all alignment methods must implement.
    """

    def __init__(self, name, version, QA_path=None, logger=None):
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

    def check_input(
        self,
        reference_frame,
        input_frame,
        skip_shape_check=False,
        skip_type_check=False,
        skip_nan_check=False,
    ):
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

        if not skip_shape_check and reference_frame.shape != input_frame.shape:
            if self.logger:
                self.logger.error("Shape mismatch between reference and input frames.")

            return False

        if not isinstance(reference_frame, type(input_frame)) and not skip_type_check:
            if self.logger:
                self.logger.error("Type mismatch between reference and input frames.")

            return False

        if not skip_nan_check and (
            np.isnan(reference_frame).any() or np.isnan(input_frame).any()
        ):
            if self.logger:
                self.logger.error("NaN values found in reference or input frames.")

            return False

        if self.logger:
            self.logger.info("Input frames are valid for alignment.")

        return True

    def shift_image(self, image, dx, dy, dtheta):
        """
        Simple image shift and rotation for QA purposes.

        Uses scipy.ndimage.shift and scipy.ndimage.rotate.

        Parameters
        ----------
        image : np.2darray
            The image to be shifted and rotated.

        dx : float
            Shift in the x-direction (pixels).

        dy : float
            Shift in the y-direction (pixels).

        dtheta : float
            Rotation angle (degrees).

        Returns
        -------
        np.2darray
            The shifted and rotated image.
        """

        # we need to remove any NaN values from the image before shifting and rotating
        # work on a copy for good measure

        image = np.copy(image)

        image = remove_nans(image, fill_value=0, logger=self.logger)

        # Shift the image
        shifted_image = shift(image, shift=(dy, dx), mode="nearest")

        # Rotate the image around its center
        rotated_image = rotate(
            shifted_image, angle=dtheta, reshape=False, mode="nearest"
        )

        return rotated_image

    def _plot_contour_overlay(
        self,
        ax,
        reference_frame,
        input_frame,
        title,
        n_levels=2,
        lower_per=50,
        upper_per=99,
    ):
        """
        Overlay contour maps of two frames on a single axis.

        Contour levels are set per image from percentiles of its own pixel
        distribution, so the same structures are traced in both images.
        Reference contours are red, input contours are blue.
        """
        for frame, color in ((reference_frame, "red"), (input_frame, "blue")):
            levels = np.unique(
                np.nanpercentile(frame, np.linspace(lower_per, upper_per, n_levels))
            )
            if len(levels) < 2:
                # Flat or degenerate image, nothing sensible to contour
                continue
            ax.contour(frame, levels=levels, colors=color, linewidths=0.8)

        # contour() uses a lower-left origin; flip to match imshow orientation
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("X Pixel")
        ax.set_ylabel("Y Pixel")

    def plot_difference_image(
        self,
        reference_frame,
        input_frame,
        dx,
        dy,
        dtheta,
        obs_name,
        show=False,
        custom_title=None,
        v_lower_per=15,
        v_upper_per=85,
    ):
        """
        Plot the difference image before and after alignment.

        Parameters
        ----------
        reference_frame : np.2darray
            The reference image.

        input_frame : np.2darray
            The target image.

        dx: float
            Shift in the x-direction (pixels).

        dy: float
            Shift in the y-direction (pixels).

        dtheta: float
            Rotation angle (degrees).

        obs_name: str
            Name of the observation, used for saving QA plots and logging.

        show: bool, optional
            If True, display the plot interactively. Default is False.

        custom_title: str, optional
            Custom title for the plot. If None, a default title will be used.
        """

        # normalize both images to unity

        reference_frame = reference_frame / np.nanmax(reference_frame)
        input_frame = input_frame / np.nanmax(input_frame)

        aligned_frame = self.shift_image(input_frame, dx, dy, dtheta)

        diff_prior = reference_frame - input_frame
        diff_post = reference_frame - aligned_frame

        _, axes = plt.subplots(2, 3, figsize=(18, 12))

        ref_vmin, ref_vmax = np.nanpercentile(
            reference_frame, v_lower_per
        ), np.nanpercentile(reference_frame, v_upper_per)
        in_vmin, in_vmax = np.nanpercentile(input_frame, v_lower_per), np.nanpercentile(
            input_frame, v_upper_per
        )

        axes[0, 0].imshow(reference_frame, cmap="Reds", vmin=ref_vmin, vmax=ref_vmax)
        axes[0, 0].imshow(
            input_frame, cmap="Blues", vmin=in_vmin, vmax=in_vmax, alpha=0.5
        )
        axes[0, 0].set_title("Before Alignment (Red = Reference, Blue = Input)")
        axes[0, 0].set_xlabel("X Pixel")
        axes[0, 0].set_ylabel("Y Pixel")

        axes[0, 1].imshow(reference_frame, cmap="Reds", vmin=ref_vmin, vmax=ref_vmax)
        axes[0, 1].imshow(
            aligned_frame, cmap="Blues", vmin=in_vmin, vmax=in_vmax, alpha=0.5
        )
        axes[0, 1].set_title("After Alignment (Red = Reference, Blue = Aligned)")
        axes[0, 1].set_xlabel("X Pixel")
        axes[0, 1].set_ylabel("Y Pixel")

        axes[1, 0].imshow(
            diff_prior,
            cmap="gray",
            vmin=np.nanpercentile(diff_prior, v_lower_per),
            vmax=np.nanpercentile(diff_prior, v_upper_per),
        )
        axes[1, 0].set_title("Difference Before Alignment")
        axes[1, 0].set_xlabel("X Pixel")
        axes[1, 0].set_ylabel("Y Pixel")

        axes[1, 1].imshow(
            diff_post,
            cmap="gray",
            vmin=np.nanpercentile(diff_post, v_lower_per),
            vmax=np.nanpercentile(diff_post, v_upper_per),
        )
        axes[1, 1].set_title("Difference After Alignment")
        axes[1, 1].set_xlabel("X Pixel")
        axes[1, 1].set_ylabel("Y Pixel")

        # Third column: overlayed contour maps
        self._plot_contour_overlay(
            axes[0, 2],
            reference_frame,
            input_frame,
            "Contours Before Alignment (Red = Reference, Blue = Input)",
        )
        self._plot_contour_overlay(
            axes[1, 2],
            reference_frame,
            aligned_frame,
            "Contours After Alignment (Red = Reference, Blue = Aligned)",
        )

        for ax in axes.ravel():
            ax.set_aspect("equal")

        if custom_title is not None:
            plt.suptitle(
                custom_title
                + f"\n Offsets: dx={dx:.2f}, dy={dy:.2f}, dtheta={dtheta:.2f} deg, method: {self.name}"
            )
        else:
            plt.suptitle(
                f"Difference Image for {obs_name}, method: {self.name}\n Offsets: dx={dx:.2f}, dy={dy:.2f}, dtheta={dtheta:.2f} deg\n"
            )
        plt.savefig(f"{self.QA_path}/difference_image_{obs_name}_{self.name}.png")

        if self.logger:
            self.logger.info(
                f"Saved difference image to {self.QA_path}/difference_image_{obs_name}_{self.name}.png"
            )

        if show:
            plt.show()
        else:
            plt.close()

    def plot_failed_aligment(
        self,
        reference_frame,
        input_frame,
        obs_name,
        show=False,
        v_lower_per=15,
        v_upper_per=85,
    ):
        """
        Plot the images separately and overlayed + the difference image and
        contour overlay for a failed alignment.

        Parameters
        ----------
        reference_frame : np.2darray
            The reference image.

        input_frame : np.2darray
            The target image.

        obs_name: str
            Name of the observation, used for saving QA plots and logging.

        show: bool, optional
            If True, display the plot interactively. Default is False.
        """

        # normalize both images to unity

        reference_frame = reference_frame / np.nanmax(reference_frame)
        input_frame = input_frame / np.nanmax(input_frame)

        diff_prior = reference_frame - input_frame

        _, axes = plt.subplots(2, 3, figsize=(18, 12))

        ref_vmin, ref_vmax = np.nanpercentile(
            reference_frame, v_lower_per
        ), np.nanpercentile(reference_frame, v_upper_per)
        in_vmin, in_vmax = np.nanpercentile(input_frame, v_lower_per), np.nanpercentile(
            input_frame, v_upper_per
        )

        diff_vmin, diff_vmax = np.nanpercentile(
            diff_prior, v_lower_per
        ), np.nanpercentile(diff_prior, v_upper_per)

        axes[0, 0].imshow(reference_frame, cmap="Reds", vmin=ref_vmin, vmax=ref_vmax)
        axes[0, 0].imshow(
            input_frame, cmap="Blues", vmin=in_vmin, vmax=in_vmax, alpha=0.5
        )
        axes[0, 0].set_title("Overlayed (Red = Reference, Blue = Input)")
        axes[0, 0].set_xlabel("X Pixel")
        axes[0, 0].set_ylabel("Y Pixel")

        axes[0, 1].imshow(diff_prior, cmap="gray", vmin=diff_vmin, vmax=diff_vmax)
        axes[0, 1].set_title("Difference Image (Reference - Input)")
        axes[0, 1].set_xlabel("X Pixel")
        axes[0, 1].set_ylabel("Y Pixel")

        # vmin/vmax are already percentile values, so pass them directly
        axes[1, 0].imshow(reference_frame, cmap="gray", vmin=ref_vmin, vmax=ref_vmax)
        axes[1, 0].set_title("Raw Reference Image")
        axes[1, 0].set_xlabel("X Pixel")
        axes[1, 0].set_ylabel("Y Pixel")

        axes[1, 1].imshow(input_frame, cmap="gray", vmin=in_vmin, vmax=in_vmax)
        axes[1, 1].set_title("Raw Input Image")
        axes[1, 1].set_xlabel("X Pixel")
        axes[1, 1].set_ylabel("Y Pixel")

        # Third column: overlayed contour maps
        # Top: contours of both frames on a blank background
        self._plot_contour_overlay(
            axes[0, 2],
            reference_frame,
            input_frame,
            "Contours (Red = Reference, Blue = Input)",
        )

        # Bottom: same contours drawn on top of the raw reference image
        axes[1, 2].imshow(reference_frame, cmap="gray", vmin=ref_vmin, vmax=ref_vmax)
        for frame, color in ((reference_frame, "red"), (input_frame, "blue")):
            levels = np.unique(np.nanpercentile(frame, np.linspace(50, 99, 6)))
            if len(levels) >= 2:
                axes[1, 2].contour(frame, levels=levels, colors=color, linewidths=0.8)
        axes[1, 2].set_title("Contours on Reference Image")
        axes[1, 2].set_xlabel("X Pixel")
        axes[1, 2].set_ylabel("Y Pixel")

        for ax in axes.ravel():
            ax.set_aspect("equal")

        plt.suptitle(f"\n Failed aligment for {obs_name}, method: {self.name}")

        plt.savefig(f"{self.QA_path}/failed_aligment_{obs_name}_{self.name}.png")

        if self.logger:
            self.logger.info(
                f"Saved failed alignment image to {self.QA_path}/failed_aligment_{obs_name}_{self.name}.png"
            )

        if show:
            plt.show()
        else:
            plt.close()

    @abstractmethod
    def run(
        self,
        reference_frame,
        input_frame,
        obs_name,
        reference_wcs=None,
        input_wcs=None,
        initial_guess_translation=(None, None),
        initial_guess_rotation=None,
        save_QA=True,
        show_QA=False,
        skip_rotation_estimation=False,
        **kwargs,
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

        skip_rotation_estimation : bool, optional
            If True, the rotation estimation will be skipped.

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

        # Expected return values for the aligment methods

        (tx, ty), rot, (tx_err, ty_err), rot_err, success = self.FAILED_RETURN

        return (
            (tx, ty),
            rot,
            (tx_err, ty_err),
            rot_err,
            success,
        )


class NoAlignment(AligmentMethod):
    """
    Dummy for testing pipeline logic functionality.
    """

    def __init__(self, logger=None, QA_path=None):
        super().__init__("NoAlignment", "0.0.0", logger=logger, QA_path=QA_path)

        if self.logger:
            self.logger.info(
                f"Initialized {self.name} alignment method, version {self.version}."
            )

    def run(
        self,
        reference_frame,
        input_frame,
        obs_name,
        reference_wcs=None,
        input_wcs=None,
        initial_guess_translation=(None, None),
        initial_guess_rotation=None,
        skip_rotation_estimation=False,
        save_QA=True,
        show_QA=False,
        **kwargs,
    ):
        """
        Dummy alignment method that does nothing.

        Returns zero offsets and rotation with no uncertainty.
        """

        if self.logger:
            self.logger.info("NoAlignment method called. No alignment performed.")

        # plot the difference image for QA testing purposes

        self.plot_difference_image(
            reference_frame,
            input_frame,
            0.0,
            0.0,
            0.0,
            obs_name,
            show=show_QA,
            custom_title="NoAlignment method called. No alignment performed.",
        )

        return (0.0, 0.0), 0.0, (0.0, 0.0), 0.0, True


class PointSources(AligmentMethod):

    def __init__(
        self,
        logger=None,
        MIN_MATCHES_FRACTION=0.5,
        detection_sigma=5,
        NUM_NEAREST_NEIGHBOURS=3,
        PIXEL_TOL=5,
        min_area=10,
        outlier_threshold=0.5,
        max_iterations=5,
        QA_path=None,
    ):

        self.MIN_MATCHES_FRACTION = MIN_MATCHES_FRACTION
        self.detection_sigma = detection_sigma
        self.NUM_NEAREST_NEIGHBOURS = NUM_NEAREST_NEIGHBOURS
        self.PIXEL_TOL = PIXEL_TOL
        self.min_area = min_area

        self.outlier_threshold = outlier_threshold
        self.max_iterations = max_iterations

        super().__init__("PointSources", "0.0.0", logger=logger, QA_path=QA_path)

        if self.logger:
            self.logger.info(
                f"Initialized {self.name} alignment method, version {self.version}."
            )

    def fit_euclidean_transform(self, source, target, centre, p0=[0.0, 0.0, 0.0]):
        # TODO: mostly taken from ChatGPT, seems correct but check! (K.V.)
        """
        Fit rotation about `centre` + translation.

        Wrapper for least_squares to fit a 2D Euclidean transform to matched source and target coordinates.

        Model:
            target = centre + R(theta) @ (source - centre) + [tx, ty]

        Parameters
        ----------
        source : ndarray
            Nx2 array of source coordinates.
        target : ndarray
            Nx2 array of target coordinates.
        centre : ndarray
            The centre of rotation.
        p0 : list, optional
            Initial guess for the parameters [theta, tx, ty].

        Returns
        -------
        theta : float
            Rotation in radians.
        translation : ndarray
            [tx, ty] in pixels.
        theta_err : float
            1-sigma uncertainty on rotation, radians.
        translation_err : ndarray
            1-sigma uncertainties on tx, ty, pixels.
        """

        source = np.asarray(source, dtype=float)
        target = np.asarray(target, dtype=float)
        centre = np.asarray(centre, dtype=float)

        a = source - centre

        def _residuals(p):
            theta, tx, ty = p

            c = np.cos(theta)
            s = np.sin(theta)

            rotated = np.column_stack(
                [
                    c * a[:, 0] - s * a[:, 1],
                    s * a[:, 0] + c * a[:, 1],
                ]
            )

            prediction = centre + rotated + np.array([tx, ty])

            return (prediction - target).ravel()

        result = least_squares(
            _residuals,
            p0,
            args=(),
        )

        theta, tx, ty = result.x

        # Residual variance
        n = source.shape[0]
        n_parameters = 3
        dof = 2 * n - n_parameters

        chi2 = np.sum(result.fun**2)
        sigma2 = chi2 / dof

        # Parameter covariance
        covariance = sigma2 * np.linalg.inv(result.jac.T @ result.jac)

        parameter_errors = np.sqrt(np.diag(covariance))

        theta_err = parameter_errors[0]
        translation_err = parameter_errors[1:3]

        return (
            theta,
            np.array([tx, ty]),
            theta_err,
            translation_err,
            covariance,
        )

    def fit_translation(self, source, target, p0=[0.0, 0.0]):
        # TODO: mostly taken from ChatGPT, seems correct but check! (K.V.)
        """
        Fit translation (fixed rotation counterpart to `fit_euclidean_transform`).

        Wrapper for least_squares to fit a 2D Euclidean transform to matched source and target coordinates with rotation fixed at 0.

        Parameters
        ----------
        source : ndarray
            Nx2 array of source coordinates.
        target : ndarray
            Nx2 array of target coordinates.

        p0 : list, optional
            Initial guess for the parameters [tx, ty].

        Returns
        -------
        translation : ndarray
            [tx, ty] in pixels.
\
        translation_err : ndarray
            1-sigma uncertainties on tx, ty, pixels.
        """

        self.logger.warning("Skipping rotation estimation as per user request.")

        source = np.asarray(source, dtype=float)
        target = np.asarray(target, dtype=float)

        def _residuals(p):
            tx, ty = p

            prediction = source + np.array([tx, ty])

            return (prediction - target).ravel()

        result = least_squares(
            _residuals,
            p0,
            args=(),
        )

        tx, ty = result.x

        # Residual variance
        n = source.shape[0]
        n_parameters = 2
        dof = 2 * n - n_parameters

        chi2 = np.sum(result.fun**2)
        sigma2 = chi2 / dof

        # Parameter covariance
        covariance = sigma2 * np.linalg.inv(result.jac.T @ result.jac)

        parameter_errors = np.sqrt(np.diag(covariance))

        translation_err = parameter_errors[0:2]

        return (
            np.array([tx, ty]),
            translation_err,
            covariance,
        )

    def fit_points(
        self, source, target, centre, p0=[0.0, 0.0, 0.0], skip_rotation_estimation=False
    ):
        """
        A wrapper function that calls either `fit_euclidean_transform` or `fit_translation` based on the `skip_rotation_estimation` flag.
        """

        if skip_rotation_estimation:

            offsets, translation_err, covariance = self.fit_translation(
                source, target, p0=p0[1:3]
            )

            theta = 0.0

            theta_err = 0.0

            return (
                theta,
                offsets,
                theta_err,
                translation_err,
                covariance,
            )

        else:
            return self.fit_euclidean_transform(source, target, centre, p0=p0)

    # quickfix to non-deterministic behavior of astroalign, which uses np.random.default_rng() internally
    _original_default_rng = (
        np.random.default_rng
    )  # class-level, captured once at import time

    def _seeded_default_rng(self, *args, **kwargs):
        return PointSources._original_default_rng(42)

    def run(
        self,
        reference_frame,
        input_frame,
        obs_name,
        reference_wcs=None,
        input_wcs=None,
        save_QA=True,
        show_QA=False,
        initial_guess_translation=(None, None),
        initial_guess_rotation=None,
        skip_rotation_estimation=False,
        **kwargs,
    ):
        """
        Align the target image to the reference image using point sources.

        The transformation is iteratively fitted using a 2D Euclidean transform
        consisting of rotation about the image centre plus translation.

        After each fit, matched sources with residuals greater than defined threshold
        in either the X or Y direction are rejected. The transformation is then
        refitted using the remaining sources. Iteration continues until no
        additional outliers are found or max_iterations is reached.

        Parameters
        ----------
        reference_frame : numpy.ndarray
            Reference image.

        input_frame : numpy.ndarray
            Image to be aligned to the reference frame.

        obs_name : str
            Observation name used for logging and QA filenames.

        save_QA : bool, optional
            Save QA plots.

        show_QA : bool, optional
            Display QA plots interactively.

        initial_guess_translation : tuple, optional
            Initial guess for translation. Currently not used directly by the
            Euclidean least-squares fitting.

        initial_guess_rotation : float, optional
            Initial guess for rotation. Currently not used directly by the
            Euclidean least-squares fitting.

        skip_rotation_estimation : bool, optional
            If True, the rotation estimation will be skipped.

        **kwargs
            max_iterations : int, optional
                Maximum number of iterative fitting/rejection passes.
                Default is 10.

        Returns
        -------
        offsets : tuple
            Calculated offsets (x_offset, y_offset) in pixels.

        rotation : float
            Calculated rotation in degrees.

        sigma_translation : tuple
            Uncertainty in the calculated translation
            (sigma_x, sigma_y) in pixels.

        sigma_rotation : float
            Uncertainty in the calculated rotation in degrees.

        success : bool
            True if alignment was successful, False otherwise.
        """

        # take copies to avoid modifying the original frames

        reference_frame = reference_frame.copy()
        input_frame = input_frame.copy()

        # set all nans zerous:

        reference_frame = remove_nans(reference_frame, fill_value=0, logger=self.logger)

        input_frame = remove_nans(input_frame, fill_value=0, logger=self.logger)

        # check for other non-numerical things

        good_data = self.check_input(reference_frame, input_frame)

        # remove bright sources from both:

        reference_frame, _ = remove_bright_sources_simple(
            reference_frame, logger=self.logger
        )

        input_frame, _ = remove_bright_sources_simple(input_frame, logger=self.logger)

        # now, find all zero's in both frames and build a mask

        reference_mask = reference_frame == 0
        input_mask = input_frame == 0
        mask = reference_mask | input_mask

        # take copy of the original frames, and create masked arrays and apply the masks

        input_copy = np.copy(input_frame)
        reference_copy = np.copy(reference_frame)

        reference_frame = np.ma.masked_array(reference_copy, mask=mask)
        input_frame = np.ma.masked_array(input_copy, mask=mask)

        if not good_data:
            if self.logger:
                self.logger.error("Invalid input data for alignment.")
            return self.FAILED_RETURN

        # ------------------------------------------------------------------
        # Find matching sources using astroalign
        # ------------------------------------------------------------------\

        original = aa._np.random.default_rng
        aa._np.random.default_rng = self._seeded_default_rng

        # set default parameters for astroalign defined in the class constructor

        aa.MIN_MATCHES_FRACTION = self.MIN_MATCHES_FRACTION
        aa.NUM_NEAREST_NEIGHBOURS = self.NUM_NEAREST_NEIGHBOURS
        aa.PIXEL_TOL = self.PIXEL_TOL

        try:
            _, (source_list, target_list) = aa.find_transform(
                input_frame,
                reference_frame,
                detection_sigma=self.detection_sigma,
                min_area=self.min_area,
            )
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to find matching sources: {e}")

            recovered = False

            # some recovery options

            if (
                "List of matching triangles exhausted before an acceptable transformation was found"
                in str(e)
            ):
                if self.logger:
                    self.logger.warning(
                        "astroalign failed to find matching sources, trying to lower MIN_MATCHES_FRACTION..."
                    )

                    for MIN_MATCHES_FRACTION in [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]:
                        try:

                            aa.MIN_MATCHES_FRACTION = 0.5
                            _, (source_list, target_list) = aa.find_transform(
                                input_frame,
                                reference_frame,
                                detection_sigma=self.detection_sigma,
                                min_area=self.min_area,
                            )
                            recovered = True
                            if self.logger:
                                self.logger.info(
                                    f"Successfully found matching sources with MIN_MATCHES_FRACTION={MIN_MATCHES_FRACTION}"
                                )
                            break
                        except Exception as e_inner:
                            if self.logger:
                                self.logger.warning(
                                    f"Failed to find matching sources with MIN_MATCHES_FRACTION={MIN_MATCHES_FRACTION}: {e_inner}"
                                )

                    # reset MIN_MATCHES_FRACTION to default

                    aa.MIN_MATCHES_FRACTION = self.MIN_MATCHES_FRACTION

            # try lowering detection limit detection_sigma=5 in find_transform
            if not recovered:

                if self.logger:
                    self.logger.warning(
                        "astroalign failed to find matching sources, trying to lower detection_sigma..."
                    )

                for detection_sigma in [5, 4, 3, 2, 1]:
                    try:
                        _, (source_list, target_list) = aa.find_transform(
                            input_frame,
                            reference_frame,
                            detection_sigma=detection_sigma,
                            min_area=self.min_area,
                        )
                        recovered = True
                        if self.logger:
                            self.logger.info(
                                f"Successfully found matching sources with detection_sigma={detection_sigma}"
                            )
                        break
                    except Exception as e_inner:
                        if self.logger:
                            self.logger.warning(
                                f"Failed to find matching sources with detection_sigma={detection_sigma}: {e_inner}"
                            )
            # try increasing NUM_NEAREST_NEIGHBOUR

            if not recovered:

                if self.logger:
                    self.logger.warning(
                        "astroalign failed to find matching sources, trying to change NUM_NEAREST_NEIGHBOURS..."
                    )

                for NUM_NEAREST_NEIGHBOURS in [2, 3, 4, 5, 6, 7, 8]:
                    try:
                        aa.NUM_NEAREST_NEIGHBOURS = NUM_NEAREST_NEIGHBOURS
                        _, (source_list, target_list) = aa.find_transform(
                            input_frame,
                            reference_frame,
                            min_area=self.min_area,
                            detection_sigma=self.detection_sigma,
                        )
                        recovered = True
                        if self.logger:
                            self.logger.info(
                                f"Successfully found matching sources with NUM_NEAREST_NEIGHBOURS={NUM_NEAREST_NEIGHBOURS}"
                            )
                        break
                    except Exception as e_inner:
                        if self.logger:
                            self.logger.warning(
                                f"Failed to find matching sources with NUM_NEAREST_NEIGHBOURS={NUM_NEAREST_NEIGHBOURS}: {e_inner}"
                            )

                # reset to default

                aa.NUM_NEAREST_NEIGHBOURS = self.NUM_NEAREST_NEIGHBOURS

            # try increasing pixel_tolerance in find_transform

            if not recovered:

                if self.logger:
                    self.logger.warning(
                        "astroalign failed to find matching sources, trying to increase pixel_tolerance..."
                    )

                for pixel_tolerance in [2, 3, 4, 5, 6, 7]:

                    aa.PIXEL_TOL = pixel_tolerance

                    try:
                        _, (source_list, target_list) = aa.find_transform(
                            input_frame,
                            reference_frame,
                            min_area=self.min_area,
                            detection_sigma=self.detection_sigma,
                        )
                        recovered = True
                        if self.logger:
                            self.logger.info(
                                f"Successfully found matching sources with pixel_tolerance={pixel_tolerance}"
                            )
                        break
                    except Exception as e_inner:
                        if self.logger:
                            self.logger.warning(
                                f"Failed to find matching sources with pixel_tolerance={pixel_tolerance}: {e_inner}"
                            )

                    # reset to default

                aa.PIXEL_TOL = self.PIXEL_TOL

            # last call, try change min area
            if not recovered:

                if self.logger:
                    self.logger.error(
                        "astroalign failed to find matching sources, trying to change min_area..."
                    )

                for min_area in [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]:
                    try:
                        _, (source_list, target_list) = aa.find_transform(
                            input_frame,
                            reference_frame,
                            detection_sigma=self.detection_sigma,
                            min_area=min_area,
                        )
                        recovered = True
                        if self.logger:
                            self.logger.info(
                                f"Successfully found matching sources with min_area={min_area}"
                            )
                        break
                    except Exception as e_inner:
                        if self.logger:
                            self.logger.warning(
                                f"Failed to find matching sources with min_area={min_area}: {e_inner}"
                            )

            # give up
            if save_QA and not recovered:

                if self.logger:
                    self.logger.error("astroalign failed.")

                self.plot_failed_aligment(
                    reference_frame,
                    input_frame,
                    obs_name,
                    show=show_QA,
                )

                aa._np.random.default_rng = original

                return self.FAILED_RETURN

        finally:
            aa._np.random.default_rng = original

        if self.logger:
            self.logger.info(
                f"Found {len(source_list)} matching sources for alignment."
            )
            self.logger.info(f"Source coordinates:\n{source_list}")
            self.logger.info(f"Target coordinates:\n{target_list}")

        # Need at least 3 points for a reliable 2D Euclidean transform
        if len(source_list) < 3:
            if self.logger:
                self.logger.error("Not enough matched sources for reliable alignment.")

            self.plot_failed_aligment(
                reference_frame,
                input_frame,
                obs_name,
                show=show_QA,
            )

            return self.FAILED_RETURN

        # ------------------------------------------------------------------
        # Initial matched-source QA plot
        # ------------------------------------------------------------------

        if save_QA:
            fig = plt.figure(figsize=(10, 5))

            ax1 = plt.subplot(1, 2, 1)
            ax1.imshow(
                reference_frame,
                cmap="gray",
                vmin=np.nanpercentile(reference_frame.compressed(), 5),
                vmax=np.nanpercentile(reference_frame.compressed(), 95),
            )
            ax1.scatter(
                target_list[:, 0],
                target_list[:, 1],
                c="r",
                marker="x",
            )
            ax1.set_title(
                "Reference Frame with Matched Sources",
            )

            ax2 = plt.subplot(1, 2, 2)
            ax2.imshow(
                input_frame,
                cmap="gray",
                vmin=np.nanpercentile(input_frame.compressed(), 5),
                vmax=np.nanpercentile(input_frame.compressed(), 95),
            )
            ax2.scatter(
                source_list[:, 0],
                source_list[:, 1],
                c="b",
                marker="o",
            )
            ax2.set_title(
                "Input Frame with Matched Sources",
            )

            fig.suptitle(
                f"Matched Sources for {obs_name}",
            )

            for (xt, yt), (xs, ys) in zip(
                target_list,
                source_list,
            ):
                con = ConnectionPatch(
                    xyA=(xt, yt),
                    coordsA=ax1.transData,
                    xyB=(xs, ys),
                    coordsB=ax2.transData,
                    axesA=ax1,
                    axesB=ax2,
                    color="yellow",
                    linewidth=0.5,
                    alpha=0.6,
                    clip_on=False,
                )
                fig.add_artist(con)

            plt.savefig(
                f"{self.QA_path}/matched_sources_{obs_name}.png",
            )

            if self.logger:
                self.logger.info(
                    f"Saved QA plot of matched sources to "
                    f"{self.QA_path}/matched_sources_{obs_name}.png"
                )

            if show_QA:
                plt.show()
            else:
                plt.close()

        # ------------------------------------------------------------------
        # Image centre
        # ------------------------------------------------------------------

        ny, nx = input_frame.shape[:2]

        centre = np.array(
            [
                (nx - 1) / 2,
                (ny - 1) / 2,
            ]
        )

        # ------------------------------------------------------------------
        # Initial guess for the transformation parameters
        # ------------------------------------------------------------------

        p0 = np.array(
            [
                (
                    np.radians(initial_guess_rotation)
                    if initial_guess_rotation is not None
                    else 0.0
                ),
                (
                    initial_guess_translation[0]
                    if initial_guess_translation[0] is not None
                    else 0.0
                ),
                (
                    initial_guess_translation[1]
                    if initial_guess_translation[1] is not None
                    else 0.0
                ),
            ]
        )

        # ------------------------------------------------------------------
        # Iterative fit and outlier rejection
        # ------------------------------------------------------------------

        max_iterations = self.max_iterations

        # Boolean mask referring to the ORIGINAL source lists.
        # True = currently included in the fit.
        good_mask = np.ones(
            len(source_list),
            dtype=bool,
        )

        # Keep track of the residual threshold for the final QA plot.
        rejection_threshold = self.outlier_threshold

        converged = False

        for iteration in range(max_iterations):

            source_fit = source_list[good_mask]
            target_fit = target_list[good_mask]

            # Make sure we still have enough sources.
            if len(source_fit) < 3:
                if self.logger:
                    self.logger.error(
                        "Too few matched sources remain after " "outlier rejection."
                    )

                self.plot_failed_aligment(
                    reference_frame,
                    input_frame,
                    obs_name,
                    show=show_QA,
                )

                return self.FAILED_RETURN

            if self.logger:
                self.logger.info(
                    f"Fit iteration {iteration + 1}: "
                    f"using {len(source_fit)} matched sources."
                )

            # --------------------------------------------------------------
            # Fit Euclidean transformation
            # --------------------------------------------------------------

            try:
                (theta, offset, theta_err, offset_err, _) = self.fit_points(
                    source_fit,
                    target_fit,
                    centre,
                    p0=p0,
                    skip_rotation_estimation=skip_rotation_estimation,
                )
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to fit transformation: {e}")

                self.plot_failed_aligment(
                    reference_frame,
                    input_frame,
                    obs_name,
                    show=show_QA,
                )
                return self.FAILED_RETURN

            tx, ty = offset

            # --------------------------------------------------------------
            # Calculate predicted target positions
            # --------------------------------------------------------------

            predicted_fit = centre + np.column_stack(
                [
                    (
                        np.cos(theta) * (source_fit[:, 0] - centre[0])
                        - np.sin(theta) * (source_fit[:, 1] - centre[1])
                        + tx
                    ),
                    (
                        np.sin(theta) * (source_fit[:, 0] - centre[0])
                        + np.cos(theta) * (source_fit[:, 1] - centre[1])
                        + ty
                    ),
                ]
            )

            # --------------------------------------------------------------
            # Calculate residuals
            # --------------------------------------------------------------

            residuals_fit = target_fit - predicted_fit

            # --------------------------------------------------------------
            # Find outliers
            #
            # A source is rejected if EITHER:
            #
            #     |dx| > self.outlier_threshold pixels
            #
            # or
            #
            #     |dy| > self.outlier_threshold pixels
            # --------------------------------------------------------------

            outlier_mask = (np.abs(residuals_fit[:, 0]) > rejection_threshold) | (
                np.abs(residuals_fit[:, 1]) > rejection_threshold
            )

            n_outliers = np.sum(outlier_mask)

            if self.logger:
                self.logger.info(
                    f"Fit iteration {iteration + 1}: "
                    f"{n_outliers} outliers found "
                    f"(threshold = {rejection_threshold:.2f} pixels)."
                )

            # --------------------------------------------------------------
            # No outliers -> convergence
            # --------------------------------------------------------------

            if n_outliers == 0:
                converged = True

                if self.logger:
                    self.logger.info("No outliers found. Iterative fitting converged.")

                break

            # --------------------------------------------------------------
            # Reject outliers
            #
            # Convert indices from the current subset back to indices in
            # the original source arrays.
            # --------------------------------------------------------------

            current_indices = np.where(good_mask)[0]

            rejected_indices = current_indices[outlier_mask]

            good_mask[rejected_indices] = False

            if self.logger:
                self.logger.info(
                    f"Rejected original source indices: " f"{rejected_indices.tolist()}"
                )

        # ------------------------------------------------------------------
        # Check convergence
        # ------------------------------------------------------------------

        if not converged:
            if self.logger:
                self.logger.warning(
                    f"Maximum number of fitting iterations "
                    f"({max_iterations}) reached."
                )

        # ------------------------------------------------------------------
        # Final fit using only accepted sources
        # ------------------------------------------------------------------

        source_fit = source_list[good_mask]
        target_fit = target_list[good_mask]

        if len(source_fit) < 3:
            if self.logger:
                self.logger.error(
                    "Too few matched sources remain for the final alignment."
                )

            self.plot_failed_aligment(
                reference_frame,
                input_frame,
                obs_name,
                show=show_QA,
            )

            return self.FAILED_RETURN

        if self.logger:
            self.logger.info(
                f"Performing final fit using " f"{len(source_fit)} accepted sources."
            )

        try:
            (theta, offset, theta_err, offset_err, _) = self.fit_points(
                source_fit,
                target_fit,
                centre,
                p0=p0,
                skip_rotation_estimation=skip_rotation_estimation,
            )
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed final transformation fit: {e}")

            self.plot_failed_aligment(
                reference_frame,
                input_frame,
                obs_name,
                show=show_QA,
            )

            return self.FAILED_RETURN

        # ------------------------------------------------------------------
        # Convert results to degrees
        # ------------------------------------------------------------------

        rot = -np.degrees(theta)
        rot_err = np.degrees(theta_err)

        tx, ty = offset
        tx_err, ty_err = offset_err

        # ------------------------------------------------------------------
        # Calculate final residuals for ALL original sources
        # ------------------------------------------------------------------

        predicted_all = centre + np.column_stack(
            [
                (
                    np.cos(theta) * (source_list[:, 0] - centre[0])
                    - np.sin(theta) * (source_list[:, 1] - centre[1])
                    + tx
                ),
                (
                    np.sin(theta) * (source_list[:, 0] - centre[0])
                    + np.cos(theta) * (source_list[:, 1] - centre[1])
                    + ty
                ),
            ]
        )

        residuals = target_list - predicted_all

        # Points that were rejected during the iterative fitting.
        final_outlier_mask = ~good_mask

        # ------------------------------------------------------------------
        # Residual QA plot
        # ------------------------------------------------------------------

        if save_QA:

            fig_residuals = plt.figure(
                figsize=(6, 6),
            )

            ax_residuals = fig_residuals.add_subplot(1, 1, 1)

            # --------------------------------------------------------------
            # Accepted points
            # --------------------------------------------------------------

            if np.any(good_mask):
                ax_residuals.scatter(
                    residuals[good_mask, 0],
                    residuals[good_mask, 1],
                    c="cyan",
                    marker="o",
                    label="Accepted",
                )

            # --------------------------------------------------------------
            # Rejected points
            # --------------------------------------------------------------

            if np.any(final_outlier_mask):
                ax_residuals.scatter(
                    residuals[final_outlier_mask, 0],
                    residuals[final_outlier_mask, 1],
                    c="red",
                    marker="x",
                    s=60,
                    label="Rejected",
                )

            # --------------------------------------------------------------
            # pixel rejection boundaries
            # --------------------------------------------------------------

            ax_residuals.axhline(
                rejection_threshold,
                color="yellow",
                linestyle="--",
                linewidth=0.8,
            )

            ax_residuals.axhline(
                -rejection_threshold,
                color="yellow",
                linestyle="--",
                linewidth=0.8,
            )

            ax_residuals.axvline(
                rejection_threshold,
                color="yellow",
                linestyle="--",
                linewidth=0.8,
            )

            ax_residuals.axvline(
                -rejection_threshold,
                color="yellow",
                linestyle="--",
                linewidth=0.8,
            )

            # Zero lines
            ax_residuals.axhline(
                0,
                color="white",
                linestyle="--",
                linewidth=0.5,
            )

            ax_residuals.axvline(
                0,
                color="white",
                linestyle="--",
                linewidth=0.5,
            )

            ax_residuals.set_aspect(
                "equal",
                adjustable="box",
            )

            ax_residuals.set_title(
                f"Point source fitting residuals for\n {obs_name}",
            )

            ax_residuals.set_xlabel(
                "Residual X (pixels)",
            )

            ax_residuals.set_ylabel(
                "Residual Y (pixels)",
            )

            ax_residuals.legend()

            # Make the QA plot symmetric around zero
            all_residuals = residuals.ravel()

            finite_residuals = all_residuals[np.isfinite(all_residuals)]

            if len(finite_residuals) > 0:

                max_residual = max(
                    rejection_threshold * 1.5,
                    np.max(np.abs(finite_residuals)) * 1.1,
                )

                ax_residuals.set_xlim(
                    -max_residual,
                    max_residual,
                )

                ax_residuals.set_ylim(
                    -max_residual,
                    max_residual,
                )

            plt.savefig(
                f"{self.QA_path}/residuals_{obs_name}.png",
            )

            if self.logger:
                self.logger.info(
                    f"Saved QA plot of residuals to "
                    f"{self.QA_path}/residuals_{obs_name}.png"
                )

            if show_QA:
                plt.show()
            else:
                plt.close()

            # ------------------------------------------------------------------
            # Difference image QA plot
            # ------------------------------------------------------------------

            self.plot_difference_image(
                reference_copy, input_copy, tx, ty, rot, obs_name, show=show_QA
            )

        # ------------------------------------------------------------------
        # Logging
        # ------------------------------------------------------------------

        if self.logger:
            self.logger.info(f"Alignment results for {obs_name}:")

            self.logger.info(
                f"Accepted sources: " f"{np.sum(good_mask)}/{len(good_mask)}"
            )

            self.logger.info(f"Rejected sources: " f"{np.sum(final_outlier_mask)}")

            self.logger.info(f"Residuals after alignment (in pixels):\n" f"{residuals}")

            self.logger.info(f"Rotation:    {rot:.4f} ± {rot_err:.4f} deg")

            self.logger.info(f"X offset:     {tx:.3f} ± {tx_err:.3f} pix")

            self.logger.info(f"Y offset:     {ty:.3f} ± {ty_err:.3f} pix")

        # ------------------------------------------------------------------
        # Return final alignment
        # ------------------------------------------------------------------

        return (
            (tx, ty),
            rot,
            (tx_err, ty_err),
            rot_err,
            True,
        )


class FFT(AligmentMethod):

    def __init__(
        self,
        oversample_factor=1,
        extra_rotation=6.0,
        control_tol=0.03,
        snr_min=10.0,
        sigma_start=0.6,
        sigma_end=16,
        sigma_num=126,
        angle_scan_start = -1,
        angle_scan_end = 1,
        angle_scan_steps = 100,
        window_compare_size = 100,
        logger=None,
        QA_path=None,
    ):

        self.oversample_factor = oversample_factor
        self.extra_rotation = extra_rotation
        self.control_tol = control_tol
        self.snr_min = snr_min
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.sigma_num = sigma_num

        self.angle_scan_start = angle_scan_start
        self.angle_scan_end = angle_scan_end
        self.angle_scan_steps = angle_scan_steps

        self.window_compare_size = window_compare_size

        self.angle_scan_array = np.linspace(self.angle_scan_start, self.angle_scan_end, self.angle_scan_steps)

        super().__init__("FFT", "0.0.0", logger=logger, QA_path=QA_path)

        if self.logger:
            self.logger.info(
                f"Initialized {self.name} alignment method, version {self.version}."
            )

    def angular_shift(self, A, B, w, expected=0.0, halfwidth=None, upsample=64):
        """Shift m (deg) such that A(theta) ~ B(theta - m), plus a peak SNR.

        C(m) = sum_r w_r sum_t A_r[t+m] B_r[t]  computed with an FFT along theta
        only; zero-padding the angular spectrum gives *exact* band-limited
        interpolation (no parabola/centroid bias).
        """
        n = A.shape[1]
        cross = (w[:, None] * rfft(A, axis=1) * np.conj(rfft(B, axis=1))).sum(axis=0)
        if n % 2 == 0:
            cross[-1] = 0.0  # Nyquist term is ambiguous
        c = irfft(cross, n=n * upsample)
        ang = (np.arange(c.size) * (180.0 / c.size) + 90.0) % 180.0 - 90.0
        if halfwidth is not None:
            d = (ang - expected + 90.0) % 180.0 - 90.0
            cs = np.where(np.abs(d) <= halfwidth, c, -np.inf)
        else:
            cs = c
        k = int(np.argmax(cs))
        med = np.median(c)
        mad = 1.4826 * np.median(np.abs(c - med))
        return ang[k], (c[k] - med) / mad

    def estimate_rotation_fixed_scale(
        self,
        reference,
        image,
        extra_rotation=0.0,
        sigmas=None,
        ratio=1.6,
        ctrl_tol=0.03,
        snr_min=30.0,
        search_halfwidth=1.0,
        min_ok=3,
        pad=2,
        upsample=64,
    ):
        """
        Returns dict(rot, rot_err, success, table).

        rot  : degrees, scipy convention: rotate(reference, rot) ~ image.
        table: per-sigma rows (sigma, a_main, a_ctrl, eps, snr_main, snr_ctrl, ok).

        `extra_rotation` should be larger than search_halfwidth + peak width so the
        true peak is separated from frame-fixed features (sensor striping, pixel-grid
        anisotropy) that all pile up at dtheta = 0.
        """

        # we take copies before we start altering the frames

        reference = reference.copy()
        image = image.copy()

        # remove nans and replace with medians

        median_ref = np.nanmedian(reference)
        median_img = np.nanmedian(image)

        reference = remove_nans(reference, fill_value=median_ref)
        image = remove_nans(image, fill_value=median_img)

        # TODO - if poor performace, try also to subtract background level later
        # reference = reference - np.nanmean(reference)
        # image = image - np.nanmean(image)

        # these are needed for filtering
        L = max(reference.shape)
        win = radial_window(reference.shape)

        # control image: the *same* data rotated by a known angle (spline, no NaNs
        # inside the inscribed disk)
        img_rot = rotate(
            image, extra_rotation, reshape=False, order=3, mode="constant", cval=0.0
        )

        # TODO: figure out what this does, for now just leave it
        if sigmas is None:  # geometric: band centre ~ 1/sigma
            sigmas = np.geomspace(1.0, 16.0, 32)

        # my polar parameterisation rotates opposite to scipy's, hence the minus
        S = -1.0
        rows, ok_eps = [], []
        for s in sigmas:
            f_lo, f_hi = dog_passband(s, ratio)  # dog - difference of Gaussians
            n_theta = max(128, 2 * int(2 * np.pi * f_hi * L))  # even
            rings = {}
            for name, im in (("ref", reference), ("img", image), ("rot", img_rot)):
                rings[name], f = polar_rings(
                    dog(im, s, ratio) * win, f_lo, f_hi, n_theta, pad
                )
            w = f / f.sum()  # ring information ~ |f|

            m_main, snr_main = self.angular_shift(
                rings["rot"],
                rings["ref"],
                w,
                S * extra_rotation,
                search_halfwidth,
                upsample,
            )
            m_ctrl, snr_ctrl = self.angular_shift(
                rings["rot"],
                rings["img"],
                w,
                S * extra_rotation,
                search_halfwidth,
                upsample,
            )
            a_main, a_ctrl = S * m_main, S * m_ctrl
            eps = a_main - a_ctrl  # bias-cancelled
            ok = (
                abs(a_ctrl - extra_rotation) < ctrl_tol
                and snr_main > snr_min
                and snr_ctrl > snr_min
            )
            rows.append((s, a_main, a_ctrl, eps, snr_main, snr_ctrl, ok))
            if ok:
                ok_eps.append(eps)

        table = np.array(rows, dtype=float)
        if len(ok_eps) >= min_ok:
            rot = float(np.median(ok_eps))
            err = float(np.std(ok_eps, ddof=1))  # conservative: sigmas correlated
            return dict(rot=rot, rot_err=err, success=True, table=table)
        return dict(rot=np.nan, rot_err=np.nan, success=False, table=table)

    def cross_spectrum(self, A, B, gamma=0.5):
        """

        Helper for translation estimation.

        X = FA conj(FB) / |FA conj(FB)|^gamma.
        gamma=0: plain cross-correlation; gamma=1: pure phase correlation."""
        X = fft2(A) * np.conj(fft2(B))
        if gamma:
            mag = np.abs(X)
            X = X / (mag + 1e-8 * mag.max()) ** gamma
        return X

    def integer_peak(self, X, expected, halfwidth):
        """

        Helper for translation estimation.

        Integer-lag peak of ifft2(X) inside |lag - expected| <= halfwidth.
        Returns (tx, ty) integer start point and a peak SNR (peak vs MAD)."""
        Ny, Nx = X.shape
        c = ifft2(X).real
        ly, lx = fftfreq(Ny, 1.0 / Ny), fftfreq(Nx, 1.0 / Nx)  # signed integer lags
        m = (np.abs(ly - expected[1]) <= halfwidth)[:, None] & (
            np.abs(lx - expected[0]) <= halfwidth
        )[None, :]
        iy, ix = np.unravel_index(np.argmax(np.where(m, c, -np.inf)), c.shape)
        med = np.median(c)
        mad = 1.4826 * np.median(np.abs(c - med))
        return (lx[ix], ly[iy]), (c[iy, ix] - med) / mad, c[iy, ix]

    def refine_peak(self, X, t0, scale):
        """Maximise the band-limited correlation C(t) continuously, starting at t0."""
        Ny, Nx = X.shape
        ky, kx = fftfreq(Ny), fftfreq(Nx)  # cycles / pixel
        Xs = X / scale

        def neg_c(t):
            ey = np.exp(2j * np.pi * ky * t[1])
            ex = np.exp(2j * np.pi * kx * t[0])
            v = ey @ Xs @ ex
            gx = ey @ Xs @ (2j * np.pi * kx * ex)
            gy = (2j * np.pi * ky * ey) @ Xs @ ex
            return -v.real, -np.array([gx.real, gy.real])

        res = minimize(
            neg_c,
            np.asarray(t0, float),
            jac=True,
            method="BFGS",
            options=dict(gtol=1e-9, maxiter=100),
        )
        return res.x, bool(res.success) and np.all(np.abs(res.x - t0) < 1.0)

    def plot_translation_progression(
        self,
        result,
        obs_name,
        agree_tol=0.1,
        show=False,
    ):
        """
        Plot how the tx and ty estimates evolve as the sigma loop progresses.

        Parameters
        ----------
        result : dict
            Output of `estimate_translation`.

        obs_name : str
            Name of the observation, used for saving QA plots and logging.

        agree_tol : float, optional
            The agreement threshold used in the estimation. Only used to draw
            the tolerance band around the final estimate. Default is 0.1.

        show : bool, optional
            If True, display the plot interactively. Default is False.
        """
        table = result["table"]
        sigma = table[:, 0]
        ok = table[:, 9] == 1
        in_group = np.isin(sigma, np.asarray(result["sigmas_used"], float))

        est = result["shift"]
        err = result["shift_err"]

        fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

        for ax, col, label, k in ((axes[0], 5, "tx", 0), (axes[1], 6, "ty", 1)):
            eps = table[:, col]

            # rejected sigmas (failed control / SNR / refinement checks)
            ax.scatter(
                sigma[~ok],
                eps[~ok],
                marker="x",
                color="gray",
                label="Rejected",
                zorder=2,
            )

            # accepted sigmas
            ax.scatter(
                sigma[ok],
                eps[ok],
                color="C0",
                label="Accepted",
                zorder=3,
            )

            # running mean of the accepted points
            if ok.any():
                running = np.cumsum(eps[ok]) / np.arange(1, ok.sum() + 1)
                ax.plot(
                    sigma[ok],
                    running,
                    color="C0",
                    ls="--",
                    lw=1,
                    label="Running mean (accepted)",
                    zorder=2,
                )

            # the points that formed the agreeing group
            if in_group.any():
                ax.scatter(
                    sigma[in_group],
                    eps[in_group],
                    s=160,
                    facecolors="none",
                    edgecolors="red",
                    linewidths=1.5,
                    label="Agreeing group",
                    zorder=4,
                )

            # final estimate with its error and the agreement tolerance
            if result["success"] and np.isfinite(est[k]):
                ax.axhline(est[k], color="red", lw=1, label="Final estimate")
                ax.axhspan(
                    est[k] - err[k],
                    est[k] + err[k],
                    color="red",
                    alpha=0.25,
                    label="Error of the mean",
                )
                ax.axhspan(
                    est[k] - agree_tol / 2,
                    est[k] + agree_tol / 2,
                    color="red",
                    alpha=0.07,
                    label=f"Agreement tol ({agree_tol})",
                )
                ax.set_title(f"{label} = {est[k]:.3f} ± {err[k]:.3f} px")
            else:
                ax.set_title(f"{label}: no agreement reached")

            ax.set_ylabel(f"{label} residual (px)")
            ax.grid(alpha=0.3)

        axes[1].set_xlabel("DoG sigma (px)")
        axes[0].set_xscale("log")  # axes are shared, so this applies to both
        axes[0].legend(fontsize=8, loc="best")

        status = "success" if result["success"] else "FAILED"
        fig.suptitle(
            f"Translation estimate progression for {obs_name}, method: {self.name} ({status})"
        )
        fig.tight_layout()

        plt.savefig(
            f"{self.QA_path}/translation_progression_{obs_name}_{self.name}.png"
        )

        if self.logger:
            self.logger.info(
                f"Saved translation progression plot to {self.QA_path}/translation_progression_{obs_name}_{self.name}.png"
            )

        if show:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def _find_agreeing_points(eps_list, n_agree, tol):
        """
        Find a group of at least `n_agree` points whose x values span <= tol
        and whose y values span <= tol (i.e. they fit in a tol x tol box).

        Parameters
        ----------
        eps_list : list of (ex, ey)
            Residual offsets of the accepted sigmas so far.

        Returns
        -------
        np.ndarray of int, or None
            Indices of the largest agreeing group, or None if no group of
            size >= n_agree exists.
        """
        pts = np.asarray(eps_list, float)
        if len(pts) < n_agree:
            return None

        best = None
        # The optimal box can always be shifted so its lower-left edges touch
        # a point's x and a point's y (not necessarily the same point).
        for x0 in pts[:, 0]:
            for y0 in pts[:, 1]:
                inside = np.where(
                    (pts[:, 0] >= x0 - 1e-12)
                    & (pts[:, 0] <= x0 + tol + 1e-12)
                    & (pts[:, 1] >= y0 - 1e-12)
                    & (pts[:, 1] <= y0 + tol + 1e-12)
                )[0]
                if len(inside) >= n_agree and (best is None or len(inside) > len(best)):
                    best = inside
        return best

    def estimate_translation(
        self,
        reference,
        image,
        control_offset=(10.0, 10.0),
        sigmas=None,
        ratio=1.6,
        gamma=0.0,
        taper=0.25,
        ctrl_tol=0.02,
        snr_min=20.0,
        search_halfwidth=5.0,
        n_agree=3,
        agree_tol=0.1,
        window_iters=2,
    ):
        """
        Returns dict(shift=(tx, ty), shift_err=(sx, sy), success, table, raw, n_used,
        sigmas_used).

        Loops over `sigmas` in order and stops as soon as `n_agree` accepted
        measurements agree to within `agree_tol` pixels in BOTH x and y. The
        returned shift is the mean of those agreeing measurements and
        shift_err is the standard error of that mean (std, ddof=1, / sqrt(n)).

        control_offset  : known (dx, dy) applied to `image` for the control pair.
                          Must exceed search_halfwidth + peak width so the dx=dy=0
                          artefacts (window, DFT edge effects) stay outside the search.
                          Use dx != dy: equal offsets cannot reveal an x/y swap.
        search_halfwidth: largest |residual| (px) you are willing to accept.
        n_agree         : number of accepted sigmas that must agree (default 3).
        agree_tol       : max spread (px) allowed in x and in y among those
                          points (default 0.1).
        table columns   : sigma, tx, ty, ctrl_x, ctrl_y, eps_x, eps_y, snr_main, snr_ctrl, ok
                          (only the sigmas processed before stopping)
        raw             : median uncorrected shift (main - nominal offset), for comparison.
        window_iters    : after the first estimate the reference window is moved onto the
                          same sky area as the image window and the estimate is redone.
                          The control cannot see this bias (control pair has identical
                          windows); it grows with the true shift and with sigma.
        """
        # we take copies before we start altering the frames

        reference = reference.copy()
        image = image.copy()

        # remove nans and replace with medians

        median_ref = np.nanmedian(reference)
        median_img = np.nanmedian(image)

        reference = remove_nans(reference, fill_value=median_ref)
        image = remove_nans(image, fill_value=median_img)
        cx, cy = control_offset

        win = tukey2d(reference.shape, taper)
        # zero-pad so the known shift never crops content and the correlation never wraps
        pad = int(np.ceil(max(abs(cx), abs(cy)) + search_halfwidth)) + 2

        if sigmas is None:
            sigmas = np.geomspace(1.0, 4.0, 10)

        rows = []
        good_rows = []  # rows that passed all quality checks
        agree_idx = None

        for s in sigmas:
            # DoG -> window -> pad, all in each image's OWN frame.  The window must
            # travel with the content: windowing *after* a shift pulls the estimate
            # toward zero by tens of milli-pixels.
            Dref, Dimg = dog(reference, s, ratio), dog(image, s, ratio)
            U = np.pad(Dimg * win, pad)  # unshifted image
            A = ndi_shift(
                U, (cy, cx), order=3, mode="constant", cval=0.0
            )  # shifted image

            # control: A is an exact shifted copy of U -> truth is (cx, cy)
            Xc = self.cross_spectrum(A, U, gamma)
            t0c, snr_c, pk_c = self.integer_peak(Xc, (cx, cy), search_halfwidth)
            tc, ok_c = self.refine_peak(Xc, t0c, pk_c * Xc.size)

            # main: A vs reference.  Image content at p <-> reference content at p - eps,
            # so the reference window is the image window displaced by -eps.
            w_ref = win
            for _ in range(window_iters + 1):
                R = np.pad(Dref * w_ref, pad)
                Xm = self.cross_spectrum(A, R, gamma)
                t0m, snr_m, pk_m = self.integer_peak(Xm, (cx, cy), search_halfwidth)
                tm, ok_m = self.refine_peak(Xm, t0m, pk_m * Xm.size)
                e = tm - (cx, cy)
                w_ref = ndi_shift(
                    win, (-e[1], -e[0]), order=3, mode="constant", cval=0.0
                )
            eps = tm - tc  # bias-cancelled residual
            ok = (
                ok_m
                and ok_c
                and np.all(np.abs(tc - (cx, cy)) < ctrl_tol)
                and snr_m > snr_min
                and snr_c > snr_min
            )
            row = (s, *tm, *tc, *eps, snr_m, snr_c, float(ok))
            rows.append(row)

            if ok:
                good_rows.append(row)
                agree_idx = self._find_agreeing_points(
                    [(r[5], r[6]) for r in good_rows], n_agree, agree_tol
                )
                if agree_idx is not None:
                    break  # enough accepted sigmas agree, no need to go further

        table = np.array(rows, float)
        raw = np.median(table[:, 1:3], axis=0) - (cx, cy)

        if agree_idx is not None:
            group = np.array(good_rows, float)[agree_idx]
            eps_group = group[:, 5:7]
            n = len(group)
            est = eps_group.mean(axis=0)
            err = eps_group.std(axis=0, ddof=1) / np.sqrt(n)  # error of the mean
            return dict(
                shift=tuple(est),
                shift_err=tuple(err),
                success=True,
                table=table,
                raw=tuple(raw),
                n_used=n,
                sigmas_used=tuple(group[:, 0]),
            )
        return dict(
            shift=(np.nan, np.nan),
            shift_err=(np.nan, np.nan),
            success=False,
            table=table,
            raw=tuple(raw),
            n_used=0,
            sigmas_used=(),
        )

    def run(
        self,
        reference_frame,
        input_frame,
        obs_name,
        reference_wcs=None,
        input_wcs=None,
        initial_guess_translation=(None, None),
        initial_guess_rotation=None,
        save_QA=True,
        show_QA=False,
        skip_rotation_estimation=False,
        **kwargs,
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

        # take a copy of the input frames to avoid modifying the originals
        reference_frame_backup = reference_frame.copy()
        input_frame_backup = input_frame.copy()

        # median used for filling out values in corners

        median = np.nanmedian(input_frame)

        ref_median = np.nanmedian(reference_frame)

        # we use this for estimating the best angle from a scan
        diffs = []

        for angle in self.angle_scan_array:

            reference_frame = reference_frame_backup
            input_frame = input_frame_backup

            # mask to track valid pixels for later diff image:

            valid = np.ones_like(input_frame, dtype=float)

            rotated_input_frame = rotate(
                            input_frame, angle, reshape=False, order=3, cval=median
                        )
            
            valid = (
                rotate(valid, angle, reshape=False, order=0, cval=0) > 0.5
            )

            result = self.estimate_translation(
                reference_frame,
                rotated_input_frame,
                control_offset=(10.2, 10.2),
                sigmas=np.geomspace(self.sigma_start, self.sigma_end, self.sigma_num),
                ratio=1.6,
                gamma=0.0,
                taper=0.25,
                ctrl_tol=self.control_tol,
                snr_min=10.0,
                search_halfwidth=20.0,
            )

            if  not result["success"]:

                self.logger.warning(f"Failed to estimate translation for {angle} degrees")

                diffs.append(np.nan)

                continue

            tx, ty = result["shift"]
            tx = -tx
            ty = -ty
            tx_err, ty_err = result["shift_err"]

            self.logger.info(f"Estimated translation: ({tx:.2f}, {ty:.2f}) +/- ({tx_err:.2f}, {ty_err:.2f}) for {angle} degrees")

            valid = (
                shift(
                    valid,
                    shift=(ty, tx),
                    order=0,
                    mode="constant",
                    cval=0,
                )
                > 0.5
            )

            shifted_input = shift(
                rotated_input_frame,
                shift=(ty, tx),
                order=3,
                mode="constant",
                cval=median,
            )
            

            center = np.array(reference_frame.shape) // 2
            reference_frame = reference_frame[
                center[0] - self.window_compare_size : center[0] + self.window_compare_size,
                center[1] - self.window_compare_size : center[1] + self.window_compare_size,
            ]
            shifted_input = shifted_input[
                center[0] - self.window_compare_size : center[0] + self.window_compare_size,
                center[1] - self.window_compare_size : center[1] + self.window_compare_size,
            ]
            valid = valid[
                center[0] - self.window_compare_size : center[0] + self.window_compare_size,
                center[1] - self.window_compare_size : center[1] + self.window_compare_size,
            ]

            mean_diff = np.mean(np.abs(((reference_frame/ref_median) - (shifted_input/median))[valid]))
            
            diffs.append(mean_diff)




        # now fit for minimum angle

        # fit a second order polynomial +/5 5 points from numerical minimum and calculate the minimum analycitacl
        min_angle_index = np.argmin(diffs)

        min_angle_left = max(0, min_angle_index - 5)
        min_angle_right = min(len(diffs), min_angle_index + 5)

        angle_fit_x = self.angle_scan_array[min_angle_left:min_angle_right]
        
        diff_array_y = diffs[min_angle_left:min_angle_right]

        p = np.polyfit(angle_fit_x, diff_array_y, 2)

        min_angle = -p[1] / (2 * p[0])

        # plot the diffs vs angles, the fit, and save/show

        plt.plot(self.angle_scan_array, diffs, "o", label = "Average difference image pixel value")
        plt.plot(self.angle_scan_array, np.polyval(p, self.angle_scan_array), "r--", label = "Model around minimum")
        plt.axvline(min_angle, color="red", linestyle="--", label=f"Estimated minimum: {min_angle:.2f}°")
        plt.xlabel("Rotation Angle (degrees)")
        plt.ylabel("Average pixel value of difference image")
        plt.title(f"Rotation angle scan for {obs_name}")
        plt.legend()

        if save_QA:
            plt.savefig(f"{self.QA_path}/angle_scan_model_{obs_name}.png")

        if show_QA:
            plt.show()
        else:
            plt.close()


        rot = min_angle

        rot_err = 0 # TODO: calculate this

        # final pass - shift by the modelled angle and estimate tx and ty

        # reset frames
        reference_frame = reference_frame_backup
        input_frame = input_frame_backup

        rotated_input_frame = rotate(
                        input_frame, rot, reshape=False, order=3, cval=median
                    )
        

        result = self.estimate_translation(
            reference_frame,
            rotated_input_frame,
            control_offset=(10.2, 10.2),
            sigmas=np.geomspace(self.sigma_start, self.sigma_end, self.sigma_num),
            ratio=1.6,
            gamma=0.0,
            taper=0.25,
            ctrl_tol=self.control_tol,
            snr_min=10.0,
            search_halfwidth=20.0,
        )
        if  not result["success"]:
            self.logger.warning(f"Failed FFT aligment for {obs_name}")

            self.plot_failed_aligment(reference_frame, input_frame, obs_name)
            
            return self.FAILED_RETURN

        self.plot_translation_progression(result, obs_name, show = show_QA)

        tx, ty = result["shift"]
        tx_err, ty_err = result["shift_err"]

        tx = -tx
        ty = -ty
        
        success = True

        self.plot_difference_image(
            reference_frame, input_frame, tx, ty, rot, obs_name, show=False
        )

        return (
            (tx, ty),
            rot,
            (tx_err, ty_err),
            rot_err,
            success,
        )


class OpticalFlow(AligmentMethod):

    def __init__(self, logger=None, QA_path=None):
        super().__init__("OpticalFlow", "0.0.0", logger=logger, QA_path=QA_path)

        if self.logger:
            self.logger.info(
                f"Initialized {self.name} alignment method, version {self.version}."
            )

    def run(
        self,
        reference_frame,
        input_frame,
        obs_name,
        reference_wcs=None,
        input_wcs=None,
        initial_guess_translation=(None, None),
        initial_guess_rotation=None,
        save_QA=True,
        show_QA=False,
        skip_rotation_estimation=False,
        **kwargs,
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

        # spacepylot implementation returns all 0's on failure. For identical
        # frames. we therefore check them manually and return success = True if they are identical.
        # If frames are different,  we set detect 0,0,0 and set them as failed aligments

        # take copies to avoid modifying the original frames
        reference_frame = reference_frame.copy()
        input_frame = input_frame.copy()

        if np.nansum(np.abs(reference_frame - input_frame)) < 1e-6:
            self.logger.info(
                "Reference and input frames have identical nansum = assuming this is identifal frames, skipping..."
            )
            return (
                (0.0, 0.0),
                0.0,
                (0.0, 0.0),
                0.0,
                True,
            )

        op = align.AlignOpticalFlow(input_frame, reference_frame)

        # solution (redirecy spacepylot prints to logger)
        with redirect_stdout_to_logger(logger):
            if not skip_rotation_estimation:
                op.get_translation_rotation()
            else:
                self.logger.warning("Skipping rotation estimation as per user request.")
                op.get_translation_rotation(homography_method=TranslationTransform)

        self.logger.info(
            f"Estimated translation: {op.translation}, Estimated rotation: {op.rotation_deg}"
        )

        ty, tx = op.translation

        rot = op.rotation_deg if not skip_rotation_estimation else 0.0

        if np.abs(tx - 0) < 1e-3 and np.abs(ty - 0) < 1e-3 and np.abs(rot - 0) < 1e-3:
            self.logger.warning(
                "No offset or rotation detected in optical flow - setting aligment as failed."
            )
            self.plot_failed_aligment(
                reference_frame, input_frame, obs_name, show=show_QA
            )
            return self.FAILED_RETURN

        # plot difference image

        self.plot_difference_image(
            reference_frame, input_frame, tx, ty, rot, obs_name, show=show_QA
        )

        # TODO: set error to 0 for now

        tx_err, ty_err = 0.0, 0.0
        rot_err = 0.0

        success = True

        return (
            (tx, ty),
            rot,
            (tx_err, ty_err),
            rot_err,
            success,
        )
