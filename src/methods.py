from abc import ABC, abstractmethod
from asyncio.log import logger
import os
import astroalign as aa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Ellipse
import os
from scipy.optimize import least_squares
from scipy.ndimage import shift, rotate, gaussian_filter
import numpy.fft as fft
from scipy.ndimage import maximum_filter
from geometry import reproject_image_to_target_wcs, make_window_wcs, matrix_to_parameters
from skimage.transform import warp_polar, EuclideanTransform
from skimage.measure import ransac
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
    interpolate_nans
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
import spacepylot.alignment_utilities as spacepylot_utils
import logging

from stats import ternary_search


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
    It also owns the angle-scan machinery shared by every method: a ternary
    search over a coarse angle range, a fine scan around its minimum, a robust
    parabola fit, and the corresponding QA plot.
    """

    def __init__(
        self,
        name,
        version,
        QA_path=None,
        logger=None,
        angle_scan_start=-1.0,
        angle_scan_end=1.0,
        angle_scan_steps=100,
        ternary_eps=0.2,
        window_compare_size=100,
        fine_scan_halfwidth=0.15,
        fine_scan_points=50,
        fit_outlier_sigma=1.5,
        fit_passes=2,
        fit_min_points=10,
        plot_scan_progression=False,
        scan_blur_sigma=1.0,
        scan_translation_tol=0.3,
        scan_translation_degree=1,
        angle_bootstrap_samples=200,
        angle_bootstrap_block=5,
        angle_bootstrap_seed=1,
        angle_fit_model="hyperbola",
    ):
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

        angle_scan_start, angle_scan_end : float
            Coarse angle range (degrees) searched by the ternary search.

        angle_scan_steps : int
            Kept for backwards compatibility; the ternary search determines
            the number of coarse evaluations adaptively.

        window_compare_size : int
            Half-size (pixels) of the central window used to compute the
            difference-image metric during the scan.

        ternary_eps : float
            The ternary search stops when its interval is narrower than this
            (degrees). It only has to locate the minimum roughly: the last,
            closely spaced steps are dominated by noise, and the fine scan
            takes over from there.

        fine_scan_halfwidth : float
            Half-width (degrees) of the fine scan centred on the ternary
            search minimum. Should be at least `ternary_eps` / 2 plus some
            margin, so the fine scan covers the final ternary interval.

        fine_scan_points : int
            Number of equally spaced angles in the fine scan.

        fit_outlier_sigma : float
            In the multi-pass parabola fit, points whose residual from the
            previous fit deviates by more than this many robust sigmas
            (1.4826 * MAD) are rejected before the next fit.

        fit_passes : int
            Number of fit passes: one fit of all points, then
            (`fit_passes` - 1) reject-and-refit passes.

        fit_min_points : int
            Minimum number of finite points needed for the fit.

        plot_scan_progression : bool
            If True, methods that support it save a translation-progression
            plot for every angle tried in the scan.

        scan_blur_sigma : float
            Sigma (pixels) of the gaussian applied to both frames before the
            difference-image metric is computed, so the interpolation
            smoothing of the shifted input does not bias the metric.
            0 disables the blur.

        scan_translation_tol : float
            Fine scan angles whose translation deviates from the median
            translation of the fine scan by more than this (pixels, in x or y)
            are left out of the translation trend fit.

        scan_translation_degree : int
            Degree of the polynomial fitted to tx(angle) and ty(angle) over the
            fine scan. All fine scan angles are then re-scored with the smooth
            trend translation instead of the individual estimates, so the jitter
            of the translation estimates does not leak into the metric.
            A negative value disables the smoothing (raw translations are used).

        angle_bootstrap_samples : int
            Number of resamples of the block bootstrap that gives the error of
            the scanned angle (default 200), see `_fit_angle_parabola`.

        angle_bootstrap_block : int
            Length (in fine scan points) of the residual blocks resampled by
            that bootstrap (default 5). Should be at least the correlation
            length of the fit residuals.

        angle_bootstrap_seed : int
            Seed of that bootstrap, so a rerun gives the same error (default 1).

        angle_fit_model : str
            Model fitted to the fine scan to find the minimum: "hyperbola"
            (default), f = sqrt(a (x - x_v)^2 + b) + c, or "parabola". The
            difference metric is a rounded V over the fine scan (parabolic only
            close to the minimum, linear further out); a parabola fitted to it
            is biased towards the longer side when the fine scan is not centred
            on the minimum (0.01 deg on synthetic data, ~12x the statistical
            error), the hyperbola follows both regimes. See `_fit_angle_parabola`.
        """

        if angle_fit_model not in ("hyperbola", "parabola"):
            raise ValueError(
                f"angle_fit_model must be 'hyperbola' or 'parabola', got "
                f"{angle_fit_model!r}"
            )

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

        # angle-scan configuration (shared by all methods)
        self.angle_scan_start = angle_scan_start
        self.angle_scan_end = angle_scan_end
        self.angle_scan_steps = angle_scan_steps
        self.ternary_eps = ternary_eps
        self.window_compare_size = window_compare_size
        self.fine_scan_halfwidth = fine_scan_halfwidth
        self.fine_scan_points = fine_scan_points
        self.fit_outlier_sigma = fit_outlier_sigma
        self.fit_passes = fit_passes
        if self.fit_passes < 2:
            raise ValueError("fit_passes must be >= 2")
        self.fit_min_points = fit_min_points
        self.plot_scan_progression = plot_scan_progression
        self.scan_blur_sigma = scan_blur_sigma
        self.scan_translation_tol = scan_translation_tol
        self.scan_translation_degree = scan_translation_degree
        self.angle_bootstrap_samples = angle_bootstrap_samples
        self.angle_bootstrap_block = angle_bootstrap_block
        self.angle_bootstrap_seed = angle_bootstrap_seed
        self.angle_fit_model = angle_fit_model

        # translations per scanned angle and the frames of the current scan,
        # filled by _scan_difference_metric
        self._scan_translations = {}
        self._scan_frames = None

        self.angle_scan_array = np.linspace(
            self.angle_scan_start, self.angle_scan_end, self.angle_scan_steps
        )

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
            Rotation angle (degrees), positive counter-clockwise as the frame
            is displayed with origin upper (scipy.ndimage.rotate convention).

        The image is first rotated about its centre and then shifted, the same
        order as the alignment methods and `transformation_matrix` use.

        Returns
        -------
        np.2darray
            The rotated and shifted image.
        """

        # we need to remove any NaN values from the image before shifting and rotating
        # work on a copy for good measure

        image = np.copy(image)

        image = remove_nans(image, fill_value=0, logger=self.logger)

        # Rotate the image around its center
        rotated_image = rotate(image, angle=dtheta, reshape=False, mode="nearest")

        # Shift the image
        shifted_image = shift(rotated_image, shift=(dy, dx), mode="nearest")

        return shifted_image

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

        # masked arrays are not supported by the nan-functions below (e.g.
        # np.nanpercentile ignores the mask and can crash), so masked pixels
        # are turned into NaNs on plain arrays

        reference_frame = np.ma.filled(np.asanyarray(reference_frame, dtype=float), np.nan)
        input_frame = np.ma.filled(np.asanyarray(input_frame, dtype=float), np.nan)

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

    # ------------------------------------------------------------------
    # Shared angle-scan machinery
    # ------------------------------------------------------------------

    @staticmethod
    def _rotate_points(points, centre, theta):
        """
        Rotate `points` around `centre` by `theta` radians.

        Uses the same convention as `fit_euclidean_transform`:
            rotated = centre + R(theta) @ (points - centre)
        """
        points = np.asarray(points, dtype=float)
        centre = np.asarray(centre, dtype=float)
        delta = points - centre

        c, s = np.cos(theta), np.sin(theta)

        return centre + np.column_stack(
            [
                c * delta[:, 0] - s * delta[:, 1],
                s * delta[:, 0] + c * delta[:, 1],
            ]
        )

    def _scan_difference_metric(
        self,
        reference_frame,
        input_frame,
        angle,
        estimate_translation,
        record=True,
    ):
        """
        Evaluate one trial rotation angle of the angle scan.

        The input frame is rotated by `angle` (scipy convention, the same one
        used by `run()`), the method-specific translation estimator is called
        on the rotated frame, and the mean absolute difference between the
        aligned frames is measured in a central window.

        To keep the metric comparable between angles:

        * The difference is always averaged over the same, fixed set of pixels:
          the central window, minus pixels where the reference is not finite.
          If the (rotated and shifted) input frame does not cover this whole
          set (plus the blur margin), the angle is not scored (NaN).
          Interior NaNs of the input are filled with its median.

        * Both frames are blurred with the same gaussian
          (`scan_blur_sigma`) before differencing, so the smoothing caused by
          the sub-pixel interpolation of the input does not bias the metric.

        * If `record` is True, the translation is recorded in
          `self._scan_translations` (per angle) and the frames in
          `self._scan_frames`, so `_scan_for_best_angle` can fit a smooth
          translation trend and re-score the angles with it.

        Parameters
        ----------
        reference_frame, input_frame : np.ndarray
            Frames being aligned.

        angle : float
            Trial rotation angle in degrees.

        estimate_translation : callable
            Called as `estimate_translation(rotated_input)` and must return a
            dict with at least `shift=(tx, ty)` and `success`.

        Returns
        -------
        diff : float
            Mean absolute normalised difference, or NaN if the translation
            estimate failed for this angle or the input does not cover the
            comparison set.

        result : dict or None
            The translation result returned by the callback (useful for
            optional per-angle QA).
        """
        if record:
            self._scan_frames = (reference_frame, input_frame)

        reference_data = np.ma.filled(
            np.asanyarray(reference_frame, dtype=float), np.nan
        )
        input_data = np.ma.filled(np.asanyarray(input_frame, dtype=float), np.nan)

        reference_median = np.nanmedian(reference_data)
        input_median = np.nanmedian(input_data)

        reference_median = (
            float(reference_median) if np.isfinite(reference_median) else 0.0
        )
        input_median = float(input_median) if np.isfinite(input_median) else 0.0

        # masks and NaNs are filled with the median before interpolation
        input_filled = np.nan_to_num(input_data, nan=input_median)

        rotated_input = rotate(
            input_filled,
            angle,
            reshape=False,
            order=3,
            cval=input_median,
        )

        # footprint of the input frame (frame coverage only, interior NaNs are
        # filled above and stay part of the comparison)
        footprint = (
            rotate(
                np.ones(input_data.shape),
                angle,
                reshape=False,
                order=0,
                cval=0.0,
            )
            > 0.5
        )

        result = estimate_translation(rotated_input)

        if result is None or not result.get("success", False):
            return np.nan, result

        tx, ty = result.get("shift", (np.nan, np.nan))

        if not np.all(np.isfinite([tx, ty])):
            return np.nan, result

        # book-keep the translation for the translation trend in the scan
        if record:
            self._scan_translations[float(angle)] = (float(tx), float(ty))

        footprint = (
            shift(
                footprint.astype(float),
                shift=(ty, tx),
                order=0,
                mode="constant",
                cval=0.0,
            )
            > 0.5
        )

        shifted_input = shift(
            rotated_input,
            shift=(ty, tx),
            order=3,
            mode="constant",
            cval=input_median,
        )

        # blur both frames with the same kernel
        reference_filled = np.nan_to_num(reference_data, nan=reference_median)

        if self.scan_blur_sigma > 0:
            reference_blurred = gaussian_filter(reference_filled, self.scan_blur_sigma)
            input_blurred = gaussian_filter(shifted_input, self.scan_blur_sigma)
        else:
            reference_blurred = reference_filled
            input_blurred = shifted_input

        # central comparison window, clipped to the frame size
        ny, nx = reference_data.shape
        cy, cx = ny // 2, nx // 2
        half_y = int(min(self.window_compare_size, cy))
        half_x = int(min(self.window_compare_size, cx))

        window = (
            slice(cy - half_y, cy + half_y),
            slice(cx - half_x, cx + half_x),
        )

        # the fixed comparison set: same pixels for every angle
        fixed_set = np.isfinite(reference_data[window])

        if not np.any(fixed_set):
            return np.nan, result

        # the input must cover the whole window, plus the blur margin (otherwise
        # the median fill outside the frame is blurred into the window)
        margin = int(np.ceil(4 * self.scan_blur_sigma))
        covered = footprint[
            max(cy - half_y - margin, 0) : cy + half_y + margin,
            max(cx - half_x - margin, 0) : cx + half_x + margin,
        ]

        if not np.all(covered):
            if self.logger:
                self.logger.warning(
                    f"Angle {angle:.4f} deg: translated input ({tx:.2f}, {ty:.2f}) "
                    f"does not cover the comparison window, angle not scored."
                )
            return np.nan, result

        ref_scale = reference_median if reference_median != 0 else 1.0
        input_scale = input_median if input_median != 0 else 1.0

        mean_diff = float(
            np.mean(
                np.abs(
                    (reference_blurred[window] / ref_scale)
                    - (input_blurred[window] / input_scale)
                )[fixed_set]
            )
        )

        return mean_diff, result

    def _fit_angle_parabola(self, angles, diffs):
        """
        Multi-pass (default two) fit of the fine scan model to the fine angle
        scan, to find its minimum. (The name is historical: the model is set by
        `angle_fit_model`.)

        Model
        -----
        "hyperbola" (default): f = sqrt(a (x - x_v)^2 + b) + c, a, b > 0.
        Close to the minimum (|x - x_v| << sqrt(b / a)) this is a parabola,
        c + sqrt(b) + a / (2 sqrt(b)) (x - x_v)^2; further out it becomes the
        straight arms c + sqrt(a) |x - x_v| of a V. b sets where one turns into
        the other, so it follows the rounded-V shape of the difference metric
        over the fine scan. The minimum is exactly at x = x_v, a fit
        parameter. Fitted with scipy.optimize.least_squares, started from a
        parabola fit.

        Why not a parabola: over the fine scan the metric is a rounded V, and
        the fine scan is centred on the rough ternary minimum, not on the true
        one. A parabola fitted to a lopsided V is pulled towards the longer
        side (synthetic test: 0.310 instead of 0.300 degrees, ~12x the
        statistical error, with a clear wave in the residuals). "parabola"
        (A x^2 + B x + C, minimum at -B / 2A) is kept as an option.

        Pass 1 fits all finite points. In each further pass, points whose
        residual (from the previous fit, evaluated for all points) deviates
        from the median residual by more than `fit_outlier_sigma` robust
        sigmas (1.4826 * MAD) are rejected and the survivors are refitted.

        The fit is only accepted if the model has a minimum (parabola: A > 0,
        hyperbola: a, b > 0) and it lies inside the full angle search range
        [`angle_scan_start`, `angle_scan_end`]. The vertex may lie outside the
        fine scan itself (the fine scan is centred on the rough ternary search
        minimum, which can be somewhat off).

        Error of the vertex
        -------------------
        The formal least-squares error (covariance of x_v, for the parabola
        propagated to x_v = -B / 2A) assumes independent points, but neighbouring fine scan
        angles compare almost the same pixels and the residuals are correlated
        (they show a smooth wave rather than random scatter), so the formal
        error is too small. The reported error comes from a moving-block
        residual bootstrap instead:

        1. residuals r = d - fit for ALL finite points (in angle order),
           including the ones rejected as outliers;
        2. `angle_bootstrap_samples` times: build a new scan d* = fit + r*,
           where r* is made of randomly chosen contiguous blocks of
           `angle_bootstrap_block` residuals (so their correlation is kept),
           and repeat the WHOLE fit on it, outlier rejection included
           (`_multipass_parabola`), taking its vertex (draws with a
           non-smiling parabola are dropped);
        3. error = half the 16-84 percentile width of the vertices.

        Resampling only the kept residuals and refitting in one pass would
        underestimate the error by about 2x (tested on synthetic scans): the
        aggressive rejection trims the residual distribution, and the rejection
        itself adds scatter to the vertex. With the full refit the error is
        about 0.75-0.8x the true scatter of the vertex over noise realisations
        (white and correlated synthetic noise), i.e. still slightly optimistic,
        as residual bootstraps are.

        No new image comparisons are needed. The formal error is logged next to
        it for comparison.

        Returns
        -------
        dict or None
            coeffs (model parameters, in x = angle - x0), model, x0,
            vertex (deg, the minimum), vertex_err (deg,
            bootstrap), vertex_err_formal (deg), angles, diffs (finite points
            only), keep (bool mask of points used in the final pass),
            mean_resid (mean |residual| of the kept points), or None if the fit
            failed.
        """
        angles = np.asarray(angles, float)
        diffs = np.asarray(diffs, float)

        finite = np.isfinite(angles) & np.isfinite(diffs)
        a, d = angles[finite], diffs[finite]

        if len(a) < self.fit_min_points:
            if self.logger:
                self.logger.warning(
                    f"Only {len(a)} finite fine scan points, need "
                    f"{self.fit_min_points} for the fit"
                )
            return None

        # centre the angles for a well-conditioned fit
        x0 = 0.5 * (a.min() + a.max())
        x = a - x0

        params, keep, cov = self._multipass_parabola(x, d)

        if params is None:
            if self.logger:
                self.logger.warning(
                    f"Fine scan {self.angle_fit_model} fit failed ({int(keep.sum())} "
                    f"points survive outlier rejection, need {self.fit_min_points})"
                )
            return None

        vertex_x = self._angle_model_vertex(params)

        if vertex_x is None:
            if self.logger:
                self.logger.warning(
                    f"Fine scan {self.angle_fit_model} has no minimum "
                    f"(params {np.round(params, 6)})"
                )
            return None

        vertex = x0 + vertex_x
        if not (self.angle_scan_start <= vertex <= self.angle_scan_end):
            if self.logger:
                self.logger.warning(
                    f"Fine scan {self.angle_fit_model} minimum {vertex:.4f} lies "
                    f"outside the angle search range [{self.angle_scan_start:.4f}, "
                    f"{self.angle_scan_end:.4f}]"
                )
            return None

        if not (a.min() <= vertex <= a.max()) and self.logger:
            self.logger.info(
                f"Fine scan {self.angle_fit_model} minimum {vertex:.4f} lies "
                f"outside the fine scan [{a.min():.4f}, {a.max():.4f}] (extrapolated)"
            )

        mean_resid = float(
            np.mean(np.abs(d[keep] - self._angle_model_eval(params, x[keep])))
        )

        vertex_err, vertex_err_formal = self._vertex_error(x, d, params, keep, cov)

        if self.logger:
            self.logger.info(
                f"Fine scan vertex error: {vertex_err:.5f} degrees (block "
                f"bootstrap), {vertex_err_formal:.5f} degrees (formal, "
                f"independent points)"
            )

        return dict(
            coeffs=params,
            model=self.angle_fit_model,
            x0=x0,
            vertex=float(vertex),
            vertex_err=float(vertex_err),
            vertex_err_formal=float(vertex_err_formal),
            angles=a,
            diffs=d,
            keep=keep,
            mean_resid=mean_resid,
        )

    # ------------------------------------------------------------------
    # Fine scan models
    #
    #   parabola : params = (A, B, C), f = A x^2 + B x + C, vertex -B / 2A
    #   hyperbola: params = (a, b, c, x_v), f = sqrt(a (x - x_v)^2 + b) + c,
    #              minimum exactly at x_v (a, b > 0)
    #
    # x is the angle minus the centre of the fine scan (well conditioned).
    # ------------------------------------------------------------------

    def _angle_model_eval(self, params, x):
        """Evaluate the fine scan model `params` at x (= angle - x0)."""
        params = np.asarray(params, float)

        if self.angle_fit_model == "parabola":
            return np.polyval(params, x)

        a, b, c, x_v = params
        return np.sqrt(a * (x - x_v) ** 2 + b) + c

    def _angle_model_vertex(self, params):
        """Position (in x = angle - x0) of the minimum of the model, or None
        if the model has no minimum."""
        params = np.asarray(params, float)

        if not np.all(np.isfinite(params)):
            return None

        if self.angle_fit_model == "parabola":
            A, B, _ = params
            return None if A <= 0 else float(-B / (2 * A))

        a, b, _, x_v = params
        return None if (a <= 0 or b <= 0) else float(x_v)

    def _angle_model_fit(self, x, d, p0=None):
        """
        Least-squares fit of the fine scan model to (x, d).

        Parabola: linear (np.polyfit). Hyperbola: nonlinear
        (scipy.optimize.least_squares with a > 0, b > 0), started from `p0`
        if given, else from a parabola fit: its vertex for x_v, the slopes of
        the outer thirds of the scan for the arms (sqrt(a)), and its curvature
        A = a / (2 sqrt(b)) near the minimum for b.

        Returns (params, covariance), covariance from the jacobian scaled by
        the residual variance (NaN if it cannot be computed), or (None, None)
        if the fit failed.
        """
        if self.angle_fit_model == "parabola":
            try:
                params, cov = np.polyfit(x, d, 2, cov=True)
            except (ValueError, np.linalg.LinAlgError):
                try:
                    params, cov = np.polyfit(x, d, 2), np.full((3, 3), np.nan)
                except (ValueError, np.linalg.LinAlgError):
                    return None, None
            return params, cov

        if p0 is None:
            A, B, C = np.polyfit(x, d, 2)
            x_v0 = -B / (2 * A) if A > 0 else x[np.argmin(d)]
            x_v0 = float(np.clip(x_v0, x.min(), x.max()))

            # arm slope from the outer thirds of the scan
            order = np.argsort(x)
            third = max(2, len(x) // 3)
            slopes = [
                abs(np.polyfit(x[order[:third]], d[order[:third]], 1)[0]),
                abs(np.polyfit(x[order[-third:]], d[order[-third:]], 1)[0]),
            ]
            a0 = max(np.mean(slopes), 1e-12) ** 2

            # curvature at the bottom: A = a / (2 sqrt(b))
            sqrt_b0 = a0 / (2 * A) if A > 0 else np.ptp(d) + 1e-12
            b0 = max(sqrt_b0, 1e-12) ** 2
            c0 = float(np.min(d)) - np.sqrt(b0)

            p0 = (a0, b0, c0, x_v0)

        p0 = np.asarray(p0, float)
        p0[0] = max(p0[0], 1e-12)
        p0[1] = max(p0[1], 1e-12)

        def residuals(p):
            return self._angle_model_eval(p, x) - d

        try:
            result = least_squares(
                residuals,
                p0,
                bounds=([0.0, 0.0, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf]),
                x_scale="jac",
            )
        except (ValueError, np.linalg.LinAlgError):
            return None, None

        if not result.success:
            return None, None

        params = result.x

        dof = max(len(x) - len(params), 1)
        sigma2 = float(np.sum(result.fun**2) / dof)
        try:
            cov = sigma2 * np.linalg.inv(result.jac.T @ result.jac)
        except np.linalg.LinAlgError:
            cov = np.full((4, 4), np.nan)

        return params, cov

    def _multipass_parabola(self, x, d, p0=None):
        """
        The multi-pass fit of the fine scan model (`angle_fit_model`) used by
        `_fit_angle_parabola` (and repeated by its bootstrap): pass 1 fits all
        points; every further pass rejects points whose residual deviates from
        the median residual by more than `fit_outlier_sigma` robust sigmas and
        refits the survivors. Residuals of ALL points are recomputed from each
        new fit, so a point rejected earlier can be let back in.

        Returns (params, keep, covariance of the last pass), or
        (None, keep, None) if the fit fails or fewer than `fit_min_points`
        points survive.
        """
        keep = np.ones(len(x), dtype=bool)
        params, cov = self._angle_model_fit(x, d, p0=p0)

        if params is None:
            return None, keep, None

        for _ in range(self.fit_passes - 1):
            r = d - self._angle_model_eval(params, x)
            med = np.median(r)
            sigma = 1.4826 * np.median(np.abs(r - med))
            if sigma <= 0:
                sigma = np.std(r)
            if sigma > 0:
                keep = np.abs(r - med) <= self.fit_outlier_sigma * sigma
            else:
                keep = np.ones(len(x), dtype=bool)

            if keep.sum() < self.fit_min_points:
                return None, keep, None

            params, cov = self._angle_model_fit(x[keep], d[keep], p0=params)

            if params is None:
                return None, keep, None

        return params, keep, cov

    def _vertex_error(self, x, d, params, keep, cov):
        """
        Error of the minimum of the fine scan model `params` fitted to the
        points (x, d) (x = angle - x0, in angle order; `keep` = points used in
        the final pass, `cov` = covariance of that pass).

        Returns (bootstrap error, formal error) in degrees, see
        `_fit_angle_parabola`. NaN if it cannot be estimated.
        """
        # formal error, assuming independent points
        vertex_err_formal = np.nan
        if cov is not None and np.all(np.isfinite(cov)):
            if self.angle_fit_model == "parabola":
                # propagate (A, B) to x_v = -B / (2A)
                A, B, _ = params
                J = np.array([B / (2 * A**2), -1.0 / (2 * A)])
                vertex_err_formal = float(np.sqrt(J @ cov[:2, :2] @ J))
            else:
                # x_v is a fit parameter
                vertex_err_formal = float(np.sqrt(cov[3, 3]))

        # moving-block residual bootstrap of ALL points, repeating the full
        # fit (with outlier rejection) for every draw
        model = self._angle_model_eval(params, x)
        residuals = d - model

        n = len(x)
        block = int(np.clip(self.angle_bootstrap_block, 1, n))
        n_blocks = int(np.ceil(n / block))

        rng = np.random.default_rng(self.angle_bootstrap_seed)

        vertices = []
        for _ in range(int(self.angle_bootstrap_samples)):
            starts = rng.integers(0, n - block + 1, size=n_blocks)
            resampled = np.concatenate([residuals[k : k + block] for k in starts])[:n]

            p, _, _ = self._multipass_parabola(x, model + resampled, p0=params)
            vertex = None if p is None else self._angle_model_vertex(p)
            if vertex is not None:
                vertices.append(vertex)

        if len(vertices) < 0.5 * self.angle_bootstrap_samples:
            if self.logger:
                self.logger.warning(
                    f"Only {len(vertices)}/{self.angle_bootstrap_samples} bootstrap "
                    f"{self.angle_fit_model} fits have a minimum, vertex error unknown"
                )
            return np.nan, vertex_err_formal

        p16, p84 = np.percentile(vertices, [16, 84])

        return float(0.5 * (p84 - p16)), vertex_err_formal

    def plot_angle_scan(self, scan_result, obs_name, show=False):
        """
        Plot the coarse ternary-search points, the fine scan, the fitted
        parabola and the final minimum for one angle scan, and the translation
        found at every scanned angle (with the outlier rejection band).
        """
        coarse_angles = np.asarray(scan_result["coarse_angles"], float)
        coarse_diffs = np.asarray(scan_result["coarse_diffs"], float)
        fine_angles = np.asarray(scan_result["fine_angles"], float)
        fine_diffs = np.asarray(scan_result["fine_diffs"], float)
        fit = scan_result["fit"]
        min_angle = scan_result["angle"]
        angle_err = float(scan_result.get("angle_err", 0.0))

        n_fine = len(fine_angles)
        coarse_translations = np.asarray(
            scan_result.get("coarse_translations", np.full((len(coarse_angles), 2), np.nan)),
            float,
        )
        fine_translations = np.asarray(
            scan_result.get("fine_translations", np.full((n_fine, 2), np.nan)), float
        )
        median_translation = np.asarray(
            scan_result.get("median_translation", (np.nan, np.nan)), float
        )
        outliers = np.asarray(
            scan_result.get("translation_outliers", np.zeros(n_fine, dtype=bool)),
            bool,
        )
        smoothed = bool(scan_result.get("smoothed", False))
        fine_diffs_raw = np.asarray(scan_result.get("fine_diffs_raw", fine_diffs), float)
        trend_translations = np.asarray(
            scan_result.get("trend_translations", np.full((n_fine, 2), np.nan)), float
        )

        fine_ok = np.isfinite(fine_diffs)
        coarse_ok = np.isfinite(coarse_diffs)

        def _draw_angle_scan(ax):
            ax.plot(
                coarse_angles[coarse_ok],
                coarse_diffs[coarse_ok],
                "o",
                color="gray",
                markersize=4,
                label="Ternary search points",
            )
            if smoothed:
                raw_ok = np.isfinite(fine_diffs_raw)
                ax.plot(
                    fine_angles[raw_ok],
                    fine_diffs_raw[raw_ok],
                    ".",
                    color="gray",
                    alpha=0.5,
                    markersize=4,
                    label="Fine scan, raw translations",
                )
            if fit is not None:
                a, d, keep = fit["angles"], fit["diffs"], fit["keep"]
                ax.plot(
                    a[keep],
                    d[keep],
                    "o",
                    color="cyan",
                    markersize=4,
                    label=f"Fine scan, used ({int(keep.sum())})",
                )
                if (~keep).any():
                    ax.plot(
                        a[~keep],
                        d[~keep],
                        "x",
                        color="red",
                        markersize=7,
                        label=f"Fine scan, rejected ({int((~keep).sum())})",
                    )
                # extend the curve to the vertex if it lies outside the fine scan
                grid = np.linspace(
                    min(fine_angles.min(), fit["vertex"]),
                    max(fine_angles.max(), fit["vertex"]),
                    200,
                )
                ax.plot(
                    grid,
                    self._angle_model_eval(fit["coeffs"], grid - fit["x0"]),
                    color="green",
                    label=f"{self.angle_fit_model.capitalize()} fit ({self.fit_passes}-pass)",
                )
            else:
                ax.plot(
                    fine_angles[fine_ok & ~outliers],
                    fine_diffs[fine_ok & ~outliers],
                    "o",
                    color="cyan",
                    markersize=4,
                    label="Fine scan (fit failed)",
                )
            if outliers.any():
                # outliers refer to the raw translation estimates
                ax.plot(
                    fine_angles[outliers],
                    fine_diffs_raw[outliers],
                    "x",
                    color="magenta",
                    markersize=7,
                    label=f"Fine scan, translation outlier ({int(outliers.sum())})",
                )
            ax.axvline(
                min_angle,
                color="red",
                linestyle="--",
                label=f"Estimated minimum: {min_angle:.4f} +/- {angle_err:.4f}°",
            )
            if np.isfinite(angle_err) and angle_err > 0:
                ax.axvspan(
                    min_angle - angle_err,
                    min_angle + angle_err,
                    color="red",
                    alpha=0.15,
                )
            ax.set_xlabel("Rotation Angle (degrees)")
            ax.set_ylabel(
                f"Mean |difference| (blurred, sigma = {self.scan_blur_sigma} px)"
            )
            ax.legend()

        # top row: scan, zoom on the fine scan, translations; bottom row: the
        # residuals of the parabola fit and of the translation trend
        fig = plt.figure(figsize=(21, 8))
        grid = fig.add_gridspec(2, 3, height_ratios=[3, 1])

        axes = [
            fig.add_subplot(grid[:, 0]),
            fig.add_subplot(grid[0, 1]),
            fig.add_subplot(grid[0, 2]),
        ]
        ax_fit_res = fig.add_subplot(grid[1, 1], sharex=axes[1])
        ax_t_res = fig.add_subplot(grid[1, 2], sharex=axes[2])

        _draw_angle_scan(axes[0])
        axes[0].set_title(f"Rotation angle scan for {obs_name}, method: {self.name}")

        _draw_angle_scan(axes[1])
        span = fine_angles.max() - fine_angles.min()
        zoom_lo = min(fine_angles.min(), min_angle)
        zoom_hi = max(fine_angles.max(), min_angle)
        if span > 0:
            axes[1].set_xlim(
                zoom_lo - 0.1 * span,
                zoom_hi + 0.1 * span,
            )
        if fine_ok.any():
            lo, hi = np.min(fine_diffs[fine_ok]), np.max(fine_diffs[fine_ok])
            pad_y = 0.1 * (hi - lo) if hi > lo else 0.1 * abs(hi) + 1e-6
            axes[1].set_ylim(lo - pad_y, hi + pad_y)
        axes[1].set_title(
            "Zoom on fine scan"
            + (" (scored with the smooth translation trend)" if smoothed else "")
        )

        # ------------------------------------------------------------------
        # Translation per scanned angle, with the outlier rejection band
        # ------------------------------------------------------------------

        ax_t = axes[2]

        # angles with a translation but no score (input did not cover the window)
        has_t = np.all(np.isfinite(fine_translations), axis=1)
        not_scored = has_t & ~fine_ok
        used = has_t & fine_ok & ~outliers

        for k, (name, color) in enumerate((("tx", "C0"), ("ty", "C1"))):
            ax_t.plot(
                coarse_angles,
                coarse_translations[:, k],
                "o",
                color="gray",
                markersize=3,
                label="Ternary search" if k == 0 else None,
            )
            ax_t.plot(
                fine_angles[used],
                fine_translations[used, k],
                "o",
                color=color,
                markersize=4,
                label=f"Fine scan {name}",
            )
            if not_scored.any():
                ax_t.plot(
                    fine_angles[not_scored],
                    fine_translations[not_scored, k],
                    "o",
                    markerfacecolor="none",
                    markeredgecolor="gray",
                    markersize=6,
                    label=(
                        f"Not scored, no window coverage ({int(not_scored.sum())})"
                        if k == 0
                        else None
                    ),
                )
            if outliers.any():
                ax_t.plot(
                    fine_angles[outliers],
                    fine_translations[outliers, k],
                    "x",
                    color="magenta",
                    markersize=7,
                    label="Translation outlier" if k == 0 else None,
                )
            if smoothed:
                ax_t.plot(
                    fine_angles,
                    trend_translations[:, k],
                    color="white",
                    linewidth=1.0,
                    label="Translation trend" if k == 0 else None,
                )
            if np.isfinite(median_translation[k]):
                ax_t.axhspan(
                    median_translation[k] - self.scan_translation_tol,
                    median_translation[k] + self.scan_translation_tol,
                    color=color,
                    alpha=0.15,
                    label=f"Median {name} +/- {self.scan_translation_tol} px",
                )

        # the translation chosen at the final angle (from the trend)
        trend_translation = scan_result.get("trend_translation")
        trend_translation_err = scan_result.get("trend_translation_err")

        if trend_translation is not None:
            ax_t.axvline(min_angle, color="red", linestyle="--", linewidth=1.0)
            for k, (name, color) in enumerate((("tx", "C0"), ("ty", "C1"))):
                ax_t.errorbar(
                    min_angle,
                    trend_translation[k],
                    yerr=trend_translation_err[k],
                    xerr=angle_err if np.isfinite(angle_err) else None,
                    fmt="*",
                    color="red",
                    markeredgecolor=color,
                    markersize=14,
                    capsize=4,
                    zorder=5,
                    label=(
                        f"Chosen {name} = {trend_translation[k]:.3f} +/- "
                        f"{trend_translation_err[k]:.3f} px"
                    ),
                )

        # zoom on the fine scan, the coarse points can be far off (but keep
        # the final angle in view, it can be extrapolated)
        if span > 0:
            ax_t.set_xlim(
                min(fine_angles.min(), min_angle) - 0.1 * span,
                max(fine_angles.max(), min_angle) + 0.1 * span,
            )
        # y range from the scored angles only (unscored ones can be far off)
        finite_t = fine_translations[has_t & fine_ok]
        if len(finite_t) > 0:
            lo, hi = finite_t.min(), finite_t.max()
            if np.all(np.isfinite(median_translation)):
                lo = min(lo, median_translation.min() - self.scan_translation_tol)
                hi = max(hi, median_translation.max() + self.scan_translation_tol)
            pad_y = 0.1 * (hi - lo) if hi > lo else 0.1
            ax_t.set_ylim(lo - pad_y, hi + pad_y)
        ax_t.set_ylabel("Translation (pixels)")
        ax_t.set_title("Translation per scanned angle")
        ax_t.legend(fontsize=8)
        ax_t.tick_params(labelbottom=False)

        # ------------------------------------------------------------------
        # Residuals of the parabola fit
        # ------------------------------------------------------------------

        axes[1].tick_params(labelbottom=False)
        axes[1].set_xlabel("")

        if fit is not None:
            a, d, keep = fit["angles"], fit["diffs"], fit["keep"]
            residuals = d - self._angle_model_eval(fit["coeffs"], a - fit["x0"])

            ax_fit_res.plot(
                a[keep], residuals[keep], "o", color="cyan", markersize=4, label="Used"
            )
            if (~keep).any():
                ax_fit_res.plot(
                    a[~keep],
                    residuals[~keep],
                    "x",
                    color="red",
                    markersize=7,
                    label="Rejected",
                )

            rms = float(np.sqrt(np.mean(residuals[keep] ** 2)))
            ax_fit_res.set_title(
                f"{self.angle_fit_model.capitalize()} fit residuals (rms of used "
                f"points = {rms:.3g})",
                fontsize=10,
            )
            ax_fit_res.legend(fontsize=8)

            # y range from the used points, rejected points may be far off
            r_max = np.max(np.abs(residuals[keep])) if keep.any() else 1.0
            ax_fit_res.set_ylim(-1.5 * r_max, 1.5 * r_max)
        else:
            ax_fit_res.set_title(
                f"{self.angle_fit_model.capitalize()} fit residuals (fit failed)",
                fontsize=10,
            )

        ax_fit_res.axhline(0.0, color="gray", linewidth=0.8)
        ax_fit_res.set_xlabel("Rotation Angle (degrees)")
        ax_fit_res.set_ylabel("Data - fit")
        ax_fit_res.grid(alpha=0.3)

        # ------------------------------------------------------------------
        # Residuals of the translation trend (or of the median, if the scan was
        # not smoothed)
        # ------------------------------------------------------------------

        if smoothed:
            model = trend_translations
            model_name = "translation trend"
        else:
            model = np.tile(median_translation, (n_fine, 1))
            model_name = "median translation"

        t_residuals = fine_translations - model
        r_used = []

        for k, (name, color) in enumerate((("tx", "C0"), ("ty", "C1"))):
            ax_t_res.plot(
                fine_angles[used],
                t_residuals[used, k],
                "o",
                color=color,
                markersize=4,
                label=name,
            )
            if outliers.any():
                ax_t_res.plot(
                    fine_angles[outliers],
                    t_residuals[outliers, k],
                    "x",
                    color="magenta",
                    markersize=7,
                    label="Translation outlier" if k == 0 else None,
                )
            r_used.append(t_residuals[used, k])

        r_used = np.concatenate(r_used) if len(r_used) else np.array([])
        r_used = r_used[np.isfinite(r_used)]

        if r_used.size > 0:
            rms_x = float(np.sqrt(np.nanmean(t_residuals[used, 0] ** 2)))
            rms_y = float(np.sqrt(np.nanmean(t_residuals[used, 1] ** 2)))
            ax_t_res.set_title(
                f"Residuals to the {model_name} (rms tx = {rms_x:.3f}, "
                f"ty = {rms_y:.3f} px)",
                fontsize=10,
            )
            r_max = np.max(np.abs(r_used))
            r_max = r_max if r_max > 0 else 0.01
            ax_t_res.set_ylim(-1.5 * r_max, 1.5 * r_max)
        else:
            ax_t_res.set_title(f"Residuals to the {model_name}", fontsize=10)

        ax_t_res.axhline(0.0, color="gray", linewidth=0.8)
        ax_t_res.set_xlabel("Rotation Angle (degrees)")
        ax_t_res.set_ylabel("Data - model (px)")
        ax_t_res.grid(alpha=0.3)
        ax_t_res.legend(fontsize=8)

        fig.tight_layout()

        filename = f"{self.QA_path}/angle_scan_model_{obs_name}_{self.name}.png"
        fig.savefig(filename)

        if self.logger:
            self.logger.info(f"Saved angle scan plot to {filename}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    def _scan_for_best_angle(
        self,
        evaluate_angle,
        obs_name,
        save_QA=True,
        show_QA=False,
    ):
        """
        Run the shared coarse + fine angle scan.

        Parameters
        ----------
        evaluate_angle : callable
            Called with an angle in degrees and returns the difference-image
            metric to minimise (NaN on failure).

        obs_name : str
            Observation name used for logging and QA filenames.

        save_QA, show_QA : bool
            Save/show the angle-scan QA plot.

        Returns
        -------
        dict or None
            angle, angle_err, ternary_min_angle, coarse/fine scan points, their
            translations, the fine scan translation outliers and the parabola
            fit; None if the search failed.
        """
        # translations per angle are recorded by _scan_difference_metric
        self._scan_translations = {}
        self._scan_frames = None

        min_angle, _, tried = ternary_search(
            evaluate_angle,
            self.angle_scan_start,
            self.angle_scan_end,
            eps=self.ternary_eps,
            logger=self.logger,
        )

        if min_angle is None or tried is None:
            if self.logger:
                self.logger.warning(
                    f"Failed to find minimum angle for {obs_name} "
                    f"({self.name})"
                )
            return None

        ternary_min_angle = float(min_angle)
        coarse_angles, coarse_diffs = (
            np.asarray(v, float) for v in zip(*tried)
        )

        if self.logger:
            self.logger.info(
                f"Ternary search minimum: {ternary_min_angle:.4f} degrees "
                f"({len(coarse_angles)} evaluations)"
            )

        fine_angles = np.linspace(
            ternary_min_angle - self.fine_scan_halfwidth,
            ternary_min_angle + self.fine_scan_halfwidth,
            self.fine_scan_points,
        )

        if self.logger:
            self.logger.info(
                f"Fine scan: {self.fine_scan_points} angles in "
                f"[{fine_angles[0]:.4f}, {fine_angles[-1]:.4f}] degrees"
            )

        fine_diffs = np.array(
            [evaluate_angle(angle) for angle in fine_angles], dtype=float
        )

        def _translations(angles):
            return np.array(
                [
                    self._scan_translations.get(float(angle), (np.nan, np.nan))
                    for angle in angles
                ],
                dtype=float,
            ).reshape(-1, 2)

        coarse_translations = _translations(coarse_angles)
        fine_translations = _translations(fine_angles)

        # reject fine scan angles whose translation is far from the median
        scored = np.isfinite(fine_diffs) & np.all(np.isfinite(fine_translations), axis=1)
        translation_outliers = np.zeros(len(fine_angles), dtype=bool)
        median_translation = np.array([np.nan, np.nan])

        if np.any(scored):
            median_translation = np.median(fine_translations[scored], axis=0)
            translation_outliers = scored & np.any(
                np.abs(fine_translations - median_translation)
                > self.scan_translation_tol,
                axis=1,
            )

            if self.logger:
                self.logger.info(
                    f"Fine scan median translation: ({median_translation[0]:.3f}, "
                    f"{median_translation[1]:.3f}) px, "
                    f"{int(translation_outliers.sum())}/{int(scored.sum())} angles "
                    f"rejected (translation off by > {self.scan_translation_tol} px)"
                )

        # ------------------------------------------------------------------
        # Smooth translation trend: fit tx(angle), ty(angle) with a low-order
        # polynomial and re-score all fine angles with the trend translation.
        # The individual estimates jitter by a few hundredths of a pixel, which
        # changes the metric by more than the rotation itself does.
        # ------------------------------------------------------------------

        trend_ok = scored & ~translation_outliers
        trend_coeffs = None
        trend_translations = np.full((len(fine_angles), 2), np.nan)
        fine_diffs_raw = fine_diffs.copy()

        degree = int(self.scan_translation_degree)

        if (
            degree >= 0
            and self._scan_frames is not None
            and trend_ok.sum() >= degree + 2
        ):
            x0_trend = float(np.mean(fine_angles))

            trend_coeffs = [
                np.polyfit(
                    fine_angles[trend_ok] - x0_trend,
                    fine_translations[trend_ok, k],
                    degree,
                )
                for k in range(2)
            ]

            trend_translations = np.column_stack(
                [np.polyval(c, fine_angles - x0_trend) for c in trend_coeffs]
            )

            reference_frame, input_frame = self._scan_frames

            # cheap: only rotate, shift and difference, no new translation estimate
            fine_diffs = np.array(
                [
                    self._scan_difference_metric(
                        reference_frame,
                        input_frame,
                        angle,
                        lambda _rotated_input, t=t: dict(
                            shift=(float(t[0]), float(t[1])), success=True
                        ),
                        record=False,
                    )[0]
                    for angle, t in zip(fine_angles, trend_translations)
                ],
                dtype=float,
            )

            if self.logger:
                trend_resid = fine_translations[trend_ok] - trend_translations[trend_ok]
                self.logger.info(
                    f"Fine scan re-scored with a degree {degree} translation trend "
                    f"({int(trend_ok.sum())} angles used), rms jitter of the "
                    f"individual translations around it: "
                    f"({np.std(trend_resid[:, 0]):.4f}, {np.std(trend_resid[:, 1]):.4f}) px"
                )

            fit_diffs = fine_diffs

        else:
            if self.logger and degree >= 0:
                self.logger.warning(
                    f"Not enough translations for a degree {degree} trend "
                    f"({int(trend_ok.sum())} usable), using the raw translations."
                )

            fit_diffs = np.where(translation_outliers, np.nan, fine_diffs)

        fit = self._fit_angle_parabola(fine_angles, fit_diffs)

        if fit is None:
            min_angle = ternary_min_angle
            # the minimum is only known to within the final ternary bracket
            angle_err = 0.5 * self.ternary_eps
            if self.logger:
                self.logger.warning(
                    f"Fine scan fit failed, using ternary search minimum: "
                    f"{min_angle:.4f} degrees (error: half the final ternary "
                    f"bracket, {angle_err:.4f} degrees)"
                )
        else:
            min_angle = fit["vertex"]
            angle_err = fit["vertex_err"]

            if not np.isfinite(angle_err):
                # bootstrap failed: fall back to the formal error, else the
                # spacing of the fine scan
                angle_err = fit["vertex_err_formal"]
                if not np.isfinite(angle_err):
                    angle_err = 2 * self.fine_scan_halfwidth / max(
                        self.fine_scan_points - 1, 1
                    )
            if self.logger:
                self.logger.info(
                    f"Fine scan fit: minimum {min_angle:.4f} degrees, "
                    f"{int(fit['keep'].sum())}/{len(fit['keep'])} points used "
                    f"({int((~fit['keep']).sum())} rejected), "
                    f"mean residual {fit['mean_resid']:.4g}"
                )

        # translation at the final angle from the smooth trend, with the rms
        # scatter of the individual translations around the trend as error
        trend_translation = None
        trend_translation_err = None

        if trend_coeffs is not None:
            trend_translation = tuple(
                float(np.polyval(c, min_angle - x0_trend)) for c in trend_coeffs
            )
            trend_resid = fine_translations[trend_ok] - trend_translations[trend_ok]
            trend_translation_err = tuple(
                float(v) for v in np.sqrt(np.mean(trend_resid**2, axis=0))
            )

            if self.logger:
                self.logger.info(
                    f"Trend translation at {min_angle:.4f} degrees: "
                    f"({trend_translation[0]:.4f}, {trend_translation[1]:.4f}) +/- "
                    f"({trend_translation_err[0]:.4f}, {trend_translation_err[1]:.4f}) px"
                )

        scan_result = dict(
            angle=float(min_angle),
            angle_err=float(angle_err),
            trend_translation=trend_translation,
            trend_translation_err=trend_translation_err,
            ternary_min_angle=ternary_min_angle,
            coarse_angles=coarse_angles,
            coarse_diffs=coarse_diffs,
            fine_angles=fine_angles,
            fine_diffs=fine_diffs,
            fine_diffs_raw=fine_diffs_raw,
            trend_translations=trend_translations,
            smoothed=trend_coeffs is not None,
            coarse_translations=coarse_translations,
            fine_translations=fine_translations,
            median_translation=median_translation,
            translation_outliers=translation_outliers,
            fit=fit,
        )

        if save_QA or show_QA:
            self.plot_angle_scan(scan_result, obs_name, show=show_QA)

        return scan_result

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
        scan_for_angles=False,
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
            If True, the rotation estimation will be skipped. This takes
            precedence over `scan_for_angles`.

        scan_for_angles : bool, optional
            If True, estimate the rotation with the shared angle scan (coarse
            ternary search, fine scan and parabola fit). If False (default),
            use the method's own direct rotation estimator.

        **kwargs : dict
            Additional keyword arguments specific to the alignment method.

        Returns
        -------
        offsets: tuple
            The calculated offsets (x_offset, y_offset) in pixels.

        rotation: float
            The calculated rotation in degrees: the rotation to apply to the
            input frame to align it with the reference, positive
            counter-clockwise as the frame is displayed with origin upper
            (numpy array as read by astropy.io.fits, scipy.ndimage.rotate
            convention). Applied about the frame centre, before the offsets.

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
        scan_for_angles=False,
        **kwargs,
    ):
        """
        Dummy alignment method that does nothing.

        Returns zero offsets and rotation with no uncertainty. If
        `scan_for_angles=True` and rotation estimation is not skipped, only
        the rotation is scanned (translation stays fixed at zero).
        """

        if self.logger:
            self.logger.info("NoAlignment method called. No alignment performed.")

        scan_for_angles = bool(scan_for_angles) and not skip_rotation_estimation

        rotation = 0.0
        rotation_error = 0.0

        if scan_for_angles:
            # Translation is fixed at zero, but the rotation can still be
            # scanned. This keeps the dummy method usable for testing the
            # shared scan plumbing.
            def _zero_translation(_rotated_input):
                return dict(
                    shift=(0.0, 0.0),
                    shift_err=(0.0, 0.0),
                    success=True,
                )

            def _evaluate_angle(angle):
                diff, _ = self._scan_difference_metric(
                    reference_frame,
                    input_frame,
                    angle,
                    _zero_translation,
                )
                return diff

            scan_result = self._scan_for_best_angle(
                _evaluate_angle,
                obs_name,
                save_QA=save_QA,
                show_QA=show_QA,
            )

            if scan_result is None:
                return self.FAILED_RETURN

            rotation = scan_result["angle"]
            rotation_error = scan_result["angle_err"]

        # plot the difference image for QA testing purposes

        self.plot_difference_image(
            reference_frame,
            input_frame,
            0.0,
            0.0,
            rotation,
            obs_name,
            show=show_QA,
            custom_title="NoAlignment method called. No alignment performed.",
        )

        return (0.0, 0.0), rotation, (0.0, 0.0), rotation_error, True


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
        self,
        source,
        target,
        centre,
        p0=[0.0, 0.0, 0.0],
        skip_rotation_estimation=False,
        fixed_rotation=None,
    ):
        """
        Fit a Euclidean transform, or a translation with a fixed rotation.

        If `skip_rotation_estimation` is True, the rotation is fixed at zero.
        If `fixed_rotation` is given, it is fixed at that value (radians).
        Otherwise the full rotation + translation model is fitted.

        Fixing the rotation is used by the angle scan: the source coordinates
        are rotated by the trial angle and only the translation is fitted.
        """

        if skip_rotation_estimation or fixed_rotation is not None:

            if skip_rotation_estimation:
                theta = 0.0
                if self.logger:
                    self.logger.warning(
                        "Skipping rotation estimation as per user request."
                    )
            else:
                theta = float(fixed_rotation)

            rotated_source = self._rotate_points(source, centre, theta)

            offsets, translation_err, covariance = self.fit_translation(
                rotated_source, target, p0=p0[1:3]
            )

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

    def _fit_translation_at_angle(
        self,
        source,
        target,
        centre,
        angle,
        p0=(0.0, 0.0),
    ):
        """
        Fit a translation for one trial image rotation angle.

        The image rotation uses the scipy convention of `run()`; the matched
        source coordinates are therefore rotated by `-angle` before the
        translation-only fit. Outliers are rejected iteratively with the same
        threshold as the normal PointSources fit.

        Returns a small result dict compatible with the shared scan metric, or
        None if the fit failed.
        """
        theta = -np.radians(angle)

        good_mask = np.ones(len(source), dtype=bool)
        offset = np.asarray(p0, dtype=float)
        offset_err = np.full(2, np.nan)

        for _ in range(self.max_iterations):
            source_fit = source[good_mask]
            target_fit = target[good_mask]

            if len(source_fit) < 3:
                return None

            rotated_source = self._rotate_points(source_fit, centre, theta)

            try:
                offset, offset_err, _ = self.fit_translation(
                    rotated_source,
                    target_fit,
                    p0=offset,
                )
            except Exception:
                return None

            residuals = target_fit - (rotated_source + offset)

            outlier_mask = (
                np.abs(residuals[:, 0]) > self.outlier_threshold
            ) | (np.abs(residuals[:, 1]) > self.outlier_threshold)

            if not np.any(outlier_mask):
                break

            rejected = np.where(good_mask)[0][outlier_mask]
            good_mask[rejected] = False

        source_fit = source[good_mask]
        target_fit = target[good_mask]

        if len(source_fit) < 3:
            return None

        rotated_source = self._rotate_points(source_fit, centre, theta)

        try:
            offset, offset_err, _ = self.fit_translation(
                rotated_source,
                target_fit,
                p0=offset,
            )
        except Exception:
            return None

        return dict(
            shift=(float(offset[0]), float(offset[1])),
            shift_err=(float(offset_err[0]), float(offset_err[1])),
            success=True,
            good_mask=good_mask,
            theta=theta,
        )

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
        scan_for_angles=False,
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
            If True, the rotation estimation will be skipped. This takes
            precedence over `scan_for_angles`.

        scan_for_angles : bool, optional
            If True, estimate the rotation with the shared angle scan instead
            of the direct joint Euclidean fit. Default is False.

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

            # give up (only the QA plot depends on save_QA, the return must
            # always happen, as source_list is not defined here)
            if not recovered:

                if self.logger:
                    self.logger.error("astroalign failed.")

                if save_QA:
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
                    # the fitted theta is the negative of the returned rotation
                    -np.radians(initial_guess_rotation)
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
        # Optional shared angle scan
        # ------------------------------------------------------------------

        scan_for_angles = bool(scan_for_angles) and not skip_rotation_estimation

        scanned_rotation = None
        scanned_rotation_err = 0.0
        fixed_rotation = None

        if scan_for_angles:

            def _evaluate_angle(angle):
                translation = self._fit_translation_at_angle(
                    source_list,
                    target_list,
                    centre,
                    angle,
                    p0=p0[1:3],
                )

                if translation is None:
                    return np.nan

                diff, _ = self._scan_difference_metric(
                    reference_copy,
                    input_copy,
                    angle,
                    lambda _rotated_input: translation,
                )

                return diff

            scan_result = self._scan_for_best_angle(
                _evaluate_angle,
                obs_name,
                save_QA=save_QA,
                show_QA=show_QA,
            )

            if scan_result is None:
                self.plot_failed_aligment(
                    reference_frame,
                    input_frame,
                    obs_name,
                    show=show_QA,
                )
                return self.FAILED_RETURN

            scanned_rotation = scan_result["angle"]
            scanned_rotation_err = scan_result["angle_err"]

            # Keep the matched sources in the original frame and fit only the
            # translation at this rotation below.
            fixed_rotation = -np.radians(scanned_rotation)

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
                    fixed_rotation=fixed_rotation,
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
                fixed_rotation=fixed_rotation,
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

        if scanned_rotation is not None:
            # The sources were pre-rotated by the scanned angle, so the fit
            # above only measured the translation. Report the scanned angle.
            rot = scanned_rotation
            rot_err = scanned_rotation_err

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
        control_tol=0.05,
        snr_min=10.0,
        sigma_start=0.6,
        sigma_end=16,
        sigma_num=126,
        angle_scan_start = -1,
        angle_scan_end = 1,
        angle_scan_steps = 100,
        ternary_eps=0.2,
        window_compare_size = 100,
        fine_scan_halfwidth=0.15,
        fine_scan_points=25,
        fit_outlier_sigma=1.5,
        fit_passes=2,
        fit_min_points=10,
        translation_guess_tol=1.0,
        snr_ratio_min=0.9,
        peak_shape_tol=0.5,
        plot_scan_progression=False,
        scan_blur_sigma=1.0,
        scan_translation_tol=0.3,
        scan_translation_degree=1,
        direct_fit_before_scan=True,
        rotation_guess_tol=0.1,
        scan_sigma_margin=5,
        logger=None,
        QA_path=None,
    ):
        """
        Parameters (new ones only)
        --------------------------
        control_tol : float
            Maximum difference (px) between the recovered and the known control
            offset for a sigma to be accepted (default 0.05).

        fine_scan_halfwidth : float
            Half-width (deg) of the fine angle scan centred on the ternary search
            minimum (default 0.15, i.e. +/- 0.15 deg).

        ternary_eps : float
            Final interval width (deg) of the ternary search (default 0.2).

        fine_scan_points : int
            Number of equally spaced angles in the fine scan (default 25 for
            FFT, since every angle is a full translation estimate).

        fit_outlier_sigma : float
            In the multi-pass parabola fit, points whose residual from the previous fit
            deviates by more than this many robust sigmas (1.4826 * MAD) are
            rejected before the second fit (default 1.5, i.e. fairly aggressive:
            about 13% of purely Gaussian noise points would be rejected).

        fit_passes : int
            Number of fit passes: 1 fit of all points, then (fit_passes - 1)
            reject-and-refit passes (default 2, must be >= 2).

        fit_min_points : int
            Minimum number of finite points needed for the first fit, and of
            surviving points for the second fit (default 10).

        translation_guess_tol : float
            Uncertainty (px) of `initial_guess_translation` when it is used to
            localize the translation peak search (default 1.0).

        snr_ratio_min : float
            A translation peak is only accepted if its SNR is at least this
            fraction of the control peak SNR (default 0.9).

        peak_shape_tol : float
            A translation peak is only accepted if the widths of its fitted
            elliptical gaussian differ from the control peak fit by at most
            this fraction along the principal directions (default 0.5, i.e.
            width ratios within [1/1.5, 1.5]).

        plot_scan_progression : bool
            If True, save a translation progression plot for EVERY angle tried
            in the scan (can be well over a hundred plots). Default False.

        direct_fit_before_scan : bool
            If True (default), run() with `scan_for_angles=True` first does the
            direct (non-scan) fit; if that fails, the alignment is concluded as
            failed without running the (slow) angle scan.

        rotation_guess_tol : float
            With the angle scan: the direct fit "agrees" with the initial guess
            if its translation is within `translation_guess_tol` (px, in x and
            y) and its rotation within `rotation_guess_tol` (degrees, default
            0.1) of the guess. Then the scan only searches the sigmas around the
            ones the direct fit agreed on (see `scan_sigma_margin`).

        scan_sigma_margin : int
            Number of sigma grid steps added on each side of the sigmas the
            direct fit agreed on, when the sigma range of the scan is narrowed
            (default 5, of the `sigma_num` geometric steps).
        """

        self.oversample_factor = oversample_factor
        self.extra_rotation = extra_rotation
        self.control_tol = control_tol
        self.snr_min = snr_min
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.sigma_num = sigma_num

        self.translation_guess_tol = translation_guess_tol
        self.snr_ratio_min = snr_ratio_min
        self.peak_shape_tol = peak_shape_tol
        self.direct_fit_before_scan = direct_fit_before_scan
        self.rotation_guess_tol = rotation_guess_tol
        self.scan_sigma_margin = scan_sigma_margin

        # result of the last final translation estimate of run(), used to
        # narrow the sigma range of the scan after the direct fit
        self._last_translation_result = None

        super().__init__(
            "FFT",
            "0.0.0",
            logger=logger,
            QA_path=QA_path,
            angle_scan_start=angle_scan_start,
            angle_scan_end=angle_scan_end,
            angle_scan_steps=angle_scan_steps,
            ternary_eps=ternary_eps,
            window_compare_size=window_compare_size,
            fine_scan_halfwidth=fine_scan_halfwidth,
            fine_scan_points=fine_scan_points,
            fit_outlier_sigma=fit_outlier_sigma,
            fit_passes=fit_passes,
            fit_min_points=fit_min_points,
            plot_scan_progression=plot_scan_progression,
            scan_blur_sigma=scan_blur_sigma,
            scan_translation_tol=scan_translation_tol,
            scan_translation_degree=scan_translation_degree,
        )

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

    def integer_peak(self, X, expected, halfwidth, return_map=False):
        """

        Helper for translation estimation.

        Integer-lag peak of ifft2(X) inside |lag - expected| <= halfwidth.
        Returns (tx, ty) integer start point and a peak SNR (peak vs MAD).
        If `return_map` is True, the full correlation map is returned as a
        fourth element, so QA plots can reuse it instead of recomputing the
        inverse FFT."""
        Ny, Nx = X.shape
        c = ifft2(X).real
        ly, lx = fftfreq(Ny, 1.0 / Ny), fftfreq(Nx, 1.0 / Nx)  # signed integer lags
        m = (np.abs(ly - expected[1]) <= halfwidth)[:, None] & (
            np.abs(lx - expected[0]) <= halfwidth
        )[None, :]
        iy, ix = np.unravel_index(np.argmax(np.where(m, c, -np.inf)), c.shape)
        med = np.median(c)
        mad = 1.4826 * np.median(np.abs(c - med))
        peak = (lx[ix], ly[iy])
        snr = (c[iy, ix] - med) / mad
        if return_map:
            return peak, snr, c[iy, ix], c
        return peak, snr, c[iy, ix]

    def _peak_map_data(self, correlation, peak, halfwidth):
        """
        Crop and annotate a correlation map around an integer peak for QA
        plotting.

        Parameters
        ----------
        correlation : np.2darray
            Unshifted correlation map, as returned by `integer_peak`.

        peak : tuple
            Integer peak (tx, ty) in pixels.

        halfwidth : float
            Half-width (pixels) of the region to keep around the peak.

        Returns
        -------
        dict
            Cropped correlation map, its lag axes and the matching `imshow`
            extent.
        """
        c = fftshift(correlation)
        ny, nx = c.shape
        lx = fftshift(fftfreq(nx, 1.0 / nx))
        ly = fftshift(fftfreq(ny, 1.0 / ny))

        ix = int(np.argmin(np.abs(lx - peak[0])))
        iy = int(np.argmin(np.abs(ly - peak[1])))
        half = int(np.ceil(max(halfwidth, 1.0)))

        x0, x1 = max(ix - half, 0), min(ix + half + 1, nx)
        y0, y1 = max(iy - half, 0), min(iy + half + 1, ny)

        return dict(
            corr=c[y0:y1, x0:x1].copy(),
            lags_x=lx[x0:x1],
            lags_y=ly[y0:y1],
            extent=(
                lx[x0] - 0.5,
                lx[x1 - 1] + 0.5,
                ly[y0] - 0.5,
                ly[y1 - 1] + 0.5,
            ),
        )

    @staticmethod
    def _elliptical_gaussian(params, x, y):
        """
        Elliptical 2D gaussian without offset.

        params = (amplitude, x0, y0, sigma_major, sigma_minor, theta), where
        theta (radians) is the angle of the major axis from the x axis.
        """
        amplitude, x0, y0, sigma_major, sigma_minor, theta = params

        c, s = np.cos(theta), np.sin(theta)
        dx, dy = x - x0, y - y0

        # coordinates along the major and minor axis
        u = c * dx + s * dy
        v = -s * dx + c * dy

        return amplitude * np.exp(
            -0.5 * ((u / sigma_major) ** 2 + (v / sigma_minor) ** 2)
        )

    @staticmethod
    def _peak_fit_halfwidth(sigma, control_offset):
        """
        Half-width (pixels) of the region used for the peak shape fit.

        Grows with the DoG sigma (the correlation peak widens with it), but is
        capped at half the control offset, so the fit region never reaches the
        zero-lag artefacts.
        """
        cap = max(2, int(np.floor(min(abs(control_offset[0]), abs(control_offset[1])) / 2)))

        return int(np.clip(np.ceil(1.5 * sigma), 2, cap))

    def fit_peak_shape(self, correlation, peak, halfwidth):
        """
        Fit an elliptical gaussian to a correlation peak.

        The global median of the correlation map is subtracted as background,
        and the map is normalised to the peak value before fitting.

        Parameters
        ----------
        correlation : np.2darray
            Unshifted correlation map, as returned by `integer_peak`.

        peak : tuple
            Integer peak (tx, ty) in pixels.

        halfwidth : int
            Half-width (pixels) of the fit region around the peak.

        Returns
        -------
        dict
            success, amplitude (correlation units, above background), x0, y0,
            sigma_major, sigma_minor (pixels), theta (radians, major axis from
            the x axis), covariance (2x2), rms_residual (normalised units) and
            halfwidth.
        """
        failed = dict(
            success=False,
            amplitude=np.nan,
            x0=np.nan,
            y0=np.nan,
            sigma_major=np.nan,
            sigma_minor=np.nan,
            theta=np.nan,
            covariance=np.full((2, 2), np.nan),
            rms_residual=np.nan,
            halfwidth=int(halfwidth),
        )

        background = np.median(correlation)
        data = self._peak_map_data(correlation, peak, halfwidth)

        corr = data["corr"] - background
        x, y = np.meshgrid(data["lags_x"], data["lags_y"])

        peak_value = np.max(corr)

        if not np.isfinite(peak_value) or peak_value <= 0 or corr.size < 7:
            return failed

        z = corr / peak_value

        # starting width from the area above half maximum:
        # area = 2 pi ln(2) sigma^2 for a round gaussian
        n_half = np.sum(z >= 0.5)
        sigma0 = max(0.5, np.sqrt(n_half / (2 * np.pi * np.log(2))))

        p0 = [1.0, peak[0], peak[1], sigma0, sigma0, 0.0]
        lower = [0.0, peak[0] - 1.5, peak[1] - 1.5, 0.1, 0.1, -np.pi]
        upper = [np.inf, peak[0] + 1.5, peak[1] + 1.5, 4.0 * halfwidth, 4.0 * halfwidth, np.pi]

        def _residuals(params):
            return (self._elliptical_gaussian(params, x, y) - z).ravel()

        try:
            fit = least_squares(_residuals, p0, bounds=(lower, upper))
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Elliptical gaussian peak fit failed: {e}")
            return failed

        if not fit.success:
            return failed

        amplitude, x0, y0, sigma_major, sigma_minor, theta = fit.x

        # make sure the major axis is the larger one, and wrap theta to [-pi/2, pi/2)
        if sigma_minor > sigma_major:
            sigma_major, sigma_minor = sigma_minor, sigma_major
            theta = theta + np.pi / 2
        theta = (theta + np.pi / 2) % np.pi - np.pi / 2

        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        covariance = R @ np.diag([sigma_major**2, sigma_minor**2]) @ R.T

        return dict(
            success=True,
            amplitude=float(amplitude * peak_value),
            x0=float(x0),
            y0=float(y0),
            sigma_major=float(sigma_major),
            sigma_minor=float(sigma_minor),
            theta=float(theta),
            covariance=covariance,
            rms_residual=float(np.sqrt(np.mean(fit.fun**2))),
            halfwidth=int(halfwidth),
        )

    @staticmethod
    def _peak_shape_deviation(fit_main, fit_control):
        """
        Shape deviation between the main and the control peak.

        The eigenvalues of inv(cov_control) @ cov_main are the squared width
        ratios (main / control) along the principal directions, so this is
        independent of the orientation of the peaks, and also catches a change
        of orientation of an elongated peak.

        Returns
        -------
        float
            max(r, 1 / r) - 1 over both width ratios r, i.e. 0 for identical
            shapes and 0.5 if the main peak is 1.5 times wider (or narrower)
            than the control along some direction. NaN if a fit failed.
        """
        if not (fit_main["success"] and fit_control["success"]):
            return np.nan

        squared_ratios = np.linalg.eigvals(
            np.linalg.solve(fit_control["covariance"], fit_main["covariance"])
        ).real

        if np.any(squared_ratios <= 0) or not np.all(np.isfinite(squared_ratios)):
            return np.nan

        log_ratios = 0.5 * np.log(squared_ratios)

        return float(np.exp(np.max(np.abs(log_ratios))) - 1.0)

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
        agree_tol=0.01,
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

    def plot_translation_peak(
        self,
        result,
        obs_name,
        show=False,
    ):
        """
        Plot the quality of the FFT translation correlation peak.

        The figure shows the main and control correlation surfaces around the
        peak used for the final translation estimate (with the FWHM ellipses of
        the fitted elliptical gaussians), cuts through the main peak with the
        artificial control offset marked, and a comparison of the fitted peak
        shapes together with the SNR ratio and shape deviation checks.

        Parameters
        ----------
        result : dict
            Output of `estimate_translation`, containing the `peak` entry.

        obs_name : str
            Name of the observation, used for saving QA plots and logging.

        show : bool, optional
            If True, display the plot interactively. Default is False.
        """
        peak = result.get("peak")
        if peak is None:
            if self.logger:
                self.logger.warning(
                    "No translation peak data available for QA plotting."
                )
            return

        main = peak["main"]
        control = peak["control"]

        fig = plt.figure(figsize=(18, 10))
        grid = fig.add_gridspec(2, 3)

        ax_main = fig.add_subplot(grid[0, 0])
        ax_ctrl = fig.add_subplot(grid[1, 0])
        ax_cut = fig.add_subplot(grid[:, 1])
        ax_shape = fig.add_subplot(grid[:, 2])

        # FWHM = 2 sqrt(2 ln 2) sigma
        fwhm_factor = 2.0 * np.sqrt(2.0 * np.log(2.0))

        def _fwhm_ellipse(fit, centre, color, label):
            return Ellipse(
                xy=centre,
                width=fwhm_factor * fit["sigma_major"],
                height=fwhm_factor * fit["sigma_minor"],
                angle=np.degrees(fit["theta"]),
                facecolor="none",
                edgecolor=color,
                linewidth=1.5,
                label=label,
            )

        def _draw_peak(ax, peak_data, title):
            image = ax.imshow(
                peak_data["corr"],
                origin="lower",
                extent=peak_data["extent"],
                cmap="viridis",
                aspect="auto",
            )
            ax.scatter(
                *peak_data["expected"],
                marker="+",
                color="cyan",
                s=120,
                linewidths=1.5,
                label="Expected peak",
                zorder=2,
            )
            ax.scatter(
                *peak_data["integer_peak"],
                marker="x",
                color="red",
                s=90,
                linewidths=1.5,
                label="Integer peak",
                zorder=3,
            )
            ax.scatter(
                *peak_data["refined_peak"],
                marker="o",
                facecolors="none",
                edgecolors="white",
                s=130,
                linewidths=1.5,
                label="Refined peak",
                zorder=4,
            )
            fit = peak_data.get("fit")
            if fit is not None and fit["success"]:
                ax.add_patch(
                    _fwhm_ellipse(
                        fit, (fit["x0"], fit["y0"]), "magenta", "Fitted FWHM"
                    )
                )
            ax.set_title(title)
            ax.set_xlabel("X lag (pixels)")
            ax.set_ylabel("Y lag (pixels)")
            ax.legend(fontsize=8, loc="best")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

        _draw_peak(
            ax_main,
            main,
            f"Main correlation peak\nsigma = {peak['sigma']:.2f} px, "
            f"SNR = {main['snr']:.1f}",
        )

        _draw_peak(
            ax_ctrl,
            control,
            f"Control correlation peak (SNR = {control['snr']:.1f})",
        )

        # ------------------------------------------------------------------
        # Cuts through the main peak, normalised to peak = 1, background = 0
        # ------------------------------------------------------------------

        corr = main["corr"]
        lags_x, lags_y = main["lags_x"], main["lags_y"]

        ix = int(np.argmin(np.abs(lags_x - main["refined_peak"][0])))
        iy = int(np.argmin(np.abs(lags_y - main["refined_peak"][1])))

        background = np.median(corr)
        peak_value = corr[iy, ix]
        scale = peak_value - background

        if np.isfinite(scale) and scale > 0:
            x_cut = (corr[iy, :] - background) / scale
            y_cut = (corr[:, ix] - background) / scale
        else:
            x_cut = corr[iy, :]
            y_cut = corr[:, ix]

        ax_cut.plot(lags_x, x_cut, color="C0", label="X cut")
        ax_cut.plot(lags_y, y_cut, color="C1", label="Y cut")
        ax_cut.axhline(
            0.5,
            color="gray",
            linestyle="--",
            linewidth=0.8,
            label="Half maximum",
        )
        # Mark the artificial control offset (the shift applied to the control
        # image) on the profile. Both cuts share the same lag axis.
        control_x, control_y = peak["control_offset"]

        if np.isclose(control_x, control_y):
            ax_cut.axvline(
                control_x,
                color="cyan",
                linestyle=":",
                linewidth=1.2,
                label=f"Artificial offset ({control_x:.2f} px)",
            )
        else:
            ax_cut.axvline(
                control_x,
                color="cyan",
                linestyle=":",
                linewidth=1.2,
                label=f"Artificial offset x ({control_x:.2f} px)",
            )
            ax_cut.axvline(
                control_y,
                color="cyan",
                linestyle=":",
                linewidth=1.2,
                label=f"Artificial offset y ({control_y:.2f} px)",
            )

        profile_lags = np.concatenate(
            [
                np.asarray(lags_x, float),
                np.asarray(lags_y, float),
                np.asarray([control_x, control_y], float),
            ]
        )
        ax_cut.set_xlim(profile_lags.min() - 0.5, profile_lags.max() + 0.5)

        ax_cut.set_title(
            f"Main peak profile\n"
            f"SNR = {main['snr']:.1f}, "
            f"window delta = {peak['window_delta']:.3f} px, "
            f"search half-width = {peak['search_halfwidth']:.2f} px"
        )
        ax_cut.set_xlabel("Lag (pixels)")
        ax_cut.set_ylabel("Normalised correlation")
        ax_cut.grid(alpha=0.3)
        ax_cut.legend(fontsize=8, loc="best")

        # ------------------------------------------------------------------
        # Peak shape comparison: both fitted FWHM ellipses, centred on zero
        # ------------------------------------------------------------------

        fit_main = main.get("fit")
        fit_ctrl = control.get("fit")

        snr_ratio = peak.get("snr_ratio", np.nan)
        snr_ratio_min = peak.get("snr_ratio_min", np.nan)
        shape_dev = peak.get("shape_deviation", np.nan)
        shape_tol = peak.get("shape_tol", np.nan)

        snr_ok = bool(snr_ratio >= snr_ratio_min)
        shape_ok = bool(shape_dev <= shape_tol)

        extent = 1.0
        lines = []

        for fit, color, name in (
            (fit_ctrl, "cyan", "Control"),
            (fit_main, "magenta", "Main"),
        ):
            if fit is None or not fit["success"]:
                lines.append(f"{name}: fit failed")
                continue

            ax_shape.add_patch(_fwhm_ellipse(fit, (0.0, 0.0), color, f"{name} FWHM"))
            extent = max(extent, 0.6 * fwhm_factor * fit["sigma_major"])

            lines.append(
                f"{name}: sigma = {fit['sigma_major']:.2f} x {fit['sigma_minor']:.2f} px, "
                f"theta = {np.degrees(fit['theta']):.1f} deg, "
                f"rms res. = {fit['rms_residual']:.3f}"
            )

        lines.append(
            f"SNR ratio main/control = {snr_ratio:.2f} "
            f"(min {snr_ratio_min:.2f}): {'PASS' if snr_ok else 'FAIL'}"
        )
        lines.append(
            f"Shape deviation = {shape_dev:.2f} "
            f"(max {shape_tol:.2f}): {'PASS' if shape_ok else 'FAIL'}"
        )

        # extra space at the bottom for the text box
        ax_shape.set_xlim(-extent, extent)
        ax_shape.set_ylim(-1.8 * extent, extent)
        ax_shape.set_aspect("equal")
        ax_shape.axhline(0.0, color="gray", linewidth=0.6)
        ax_shape.axvline(0.0, color="gray", linewidth=0.6)
        ax_shape.grid(alpha=0.3)
        ax_shape.set_xlabel("X lag from peak centre (pixels)")
        ax_shape.set_ylabel("Y lag from peak centre (pixels)")
        ax_shape.set_title(
            f"Fitted peak shapes (fit half-width = "
            f"{(fit_main or fit_ctrl or {}).get('halfwidth', np.nan)} px)",
            color="green" if (snr_ok and shape_ok) else "red",
        )
        ax_shape.text(
            0.02,
            0.02,
            "\n".join(lines),
            transform=ax_shape.transAxes,
            fontsize=8,
            va="bottom",
            ha="left",
            bbox=dict(facecolor="black", alpha=0.6, edgecolor="none"),
        )
        ax_shape.legend(fontsize=8, loc="upper right")

        # ------------------------------------------------------------------
        # Figure title and output
        # ------------------------------------------------------------------

        status = "success" if result["success"] else "FAILED"
        shift = result["shift"]
        shift_err = result["shift_err"]

        fig.suptitle(
            f"FFT translation peak quality for {obs_name}, method: {self.name} "
            f"({status})\n"
            f"shift = ({shift[0]:.3f}, {shift[1]:.3f}) +/- "
            f"({shift_err[0]:.3f}, {shift_err[1]:.3f}) px, "
            f"sigma = {peak['sigma']:.2f} px"
        )
        fig.tight_layout()

        plt.savefig(
            f"{self.QA_path}/translation_peak_{obs_name}_{self.name}.png"
        )

        if self.logger:
            self.logger.info(
                f"Saved translation peak plot to "
                f"{self.QA_path}/translation_peak_{obs_name}_{self.name}.png"
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
        control_offset=(10.1, 10.1),
        sigmas=None,
        ratio=1.6,
        gamma=0.0,
        taper=0.25,
        ctrl_tol=0.1,
        snr_min=20.0,
        search_halfwidth=10.0,
        n_agree=3,
        agree_tol=0.1,
        window_iters=2,
        window_tol=0.02,
        initial_guess=None,
        guess_tol=1.0,
        snr_ratio_min=0.9,
        shape_tol=0.5,
        obs_name=None,
        save_QA=False,
        show_QA=False,
    ):
        """
        Returns dict(shift=(tx, ty), shift_err=(sx, sy), spread=(px, py), success,
        table, raw, n_used, sigmas_used, localized, peak).

        Sign convention: `shift`, `raw`, `initial_guess` and the eps columns of
        `table` all use the same sign as the offsets returned by `run()`, so no
        negation is needed by the caller.

        Loops over `sigmas` in order and stops at the first run of `n_agree`
        CONSECUTIVE sigmas that are all accepted and whose spread
        (max - min) is <= `agree_tol` in BOTH x and y. A rejected or deviating
        sigma breaks the run. No trend test is applied: a smooth, slowly varying
        eps(sigma) is a good result, and drift inside the run is already bounded
        by the spread. The returned shift is the mean of the run.

        shift_err is conservative: max(standard error of the mean, spread / 2),
        where the standard error is std(ddof=1) / sqrt(n). With few points the
        standard error alone is unreliable and ignores systematics.

        control_offset  : known (dx, dy) applied to `image` for the control pair.
                          Must exceed search_halfwidth + peak width so the dx=dy=0
                          artefacts (window, DFT edge effects) stay outside the search.
                          Use dx != dy: equal offsets cannot reveal an x/y swap.
        search_halfwidth: largest |residual| (px) you are willing to accept.
        n_agree         : number of consecutive accepted sigmas that must agree
                          (default 3, must be >= 2 so the std with ddof=1 is defined).
        agree_tol       : max spread (px) allowed in x and in y across ALL points
                          of the run (default 0.1).
        window_iters    : after the first estimate the reference window is moved onto the
                          same sky area as the image window and the estimate is redone.
                          The control cannot see this bias (control pair has identical
                          windows); it grows with the true shift and with sigma.
        window_tol      : max change (px) of the main-pair peak position between the last
                          two window iterations. A sigma whose window iteration has not
                          settled is rejected. Every iteration must also yield a valid
                          peak (ok_m), not only the last one.
        initial_guess   : optional (tx, ty) prior for the residual shift, in the same
                          convention (and sign) as the returned `shift`. None (default) searches the full
                          +/- search_halfwidth. If given, the peak search is localized
                          around the guess, and the reference window starts already
                          displaced by the guess, so `window_iters` can usually be lowered.
        guess_tol       : uncertainty (px) of `initial_guess` in x and y (default 1.0).
                          The localized search half-width is guess_tol, capped at
                          search_halfwidth. The control peak is searched in the same
                          half-width around its known position. A sigma whose main peak
                          ends up outside the localized region is rejected.
        snr_ratio_min   : a sigma is only accepted if snr_main >= snr_ratio_min * snr_ctrl
                          (default 0.9).
        shape_tol       : an elliptical gaussian is fitted to the main and the control
                          peak (see `fit_peak_shape`). A sigma is only accepted if the
                          widths of the main peak along its principal directions differ
                          from the control by at most this fraction, i.e. ratio in
                          [1 / (1 + shape_tol), 1 + shape_tol] (default 0.5, see
                          `_peak_shape_deviation`). A failed fit rejects the sigma.
        table columns   : sigma, tx, ty, ctrl_x, ctrl_y, eps_x, eps_y, snr_main, snr_ctrl,
                          ok, win_delta, snr_ratio, shape_deviation (only the sigmas
                          processed before stopping).
                          tx, ty, ctrl_x, ctrl_y are raw correlation lags (including the
                          control offset); eps_x, eps_y are the sign-flipped residuals.
        raw             : median uncorrected shift (main - nominal offset), same sign as
                          `shift`, for comparison.
        obs_name        : observation name used for QA filenames. Required for any
                          QA plot to be written.
        save_QA         : if True and `obs_name` is given, save a QA plot of the
                          final main and control correlation peaks (default False).
        show_QA         : if True, display the peak QA plot interactively.
        peak            : dict with the final correlation-surface data (main and
                          control peaks, SNRs and search metadata) used by the QA
                          plot, or None if no sigma was evaluated.
        """
        if n_agree < 2:
            raise ValueError("n_agree must be >= 2: the std with ddof=1 needs two points")

        # we take copies before we start altering the frames
        reference = reference.copy()
        image = image.copy()

        # fill nans by interpolation
        reference = interpolate_nans(reference, logger=self.logger)
        image = interpolate_nans(image, logger=self.logger)

        cx, cy = control_offset
        c_vec = np.array([cx, cy], float)

        # optional localized search around a prior estimate of the residual shift
        if initial_guess is None:
            g_vec = np.zeros(2)
            hw = search_halfwidth
        else:
            guess = np.asarray(initial_guess, float)
            if guess.shape != (2,) or not np.all(np.isfinite(guess)):
                raise ValueError("initial_guess must be a finite (tx, ty) pair or None")
            # internally the correlation lag has the opposite sign to the returned shift
            g_vec = -guess
            # the peak is searched (and must end up) within +/- guess_tol of the guess
            hw = min(search_halfwidth, guess_tol)
        center = tuple(c_vec + g_vec)  # where the main peak is expected

        win = tukey2d(reference.shape, taper)
        # zero-pad so the known shift never crops content and the correlation never wraps
        pad = int(np.ceil(max(abs(cx), abs(cy)) + search_halfwidth)) + 2

        if sigmas is None:
            sigmas = np.geomspace(1.0, 4.0, 10)

        rows = []
        group = None

        # Keep the last computed correlation maps so the final peak QA plot can
        # reuse them without computing any extra inverse FFTs.
        c_main = None
        c_ctrl = None
        last_sigma = None
        last_win_delta = np.nan

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
            t0c, snr_c, pk_c, c_ctrl = self.integer_peak(
                Xc, (cx, cy), hw, return_map=True
            )
            tc, ok_c = self.refine_peak(Xc, t0c, pk_c * Xc.size)
            tc = np.asarray(tc, float)

            # main: A vs reference.  Image content at p <-> reference content at p - eps,
            # so the reference window is the image window displaced by -eps.
            if initial_guess is None:
                w_ref = win
            else:  # start with the window already displaced by the guess
                w_ref = ndi_shift(
                    win, (-g_vec[1], -g_vec[0]), order=3, mode="constant", cval=0.0
                )
            ok_m_all = True
            e_hist = []
            for it in range(window_iters + 1):
                R = np.pad(Dref * w_ref, pad)
                Xm = self.cross_spectrum(A, R, gamma)
                t0m, snr_m, pk_m, c_main = self.integer_peak(
                    Xm, center, hw, return_map=True
                )
                tm, ok_m = self.refine_peak(Xm, t0m, pk_m * Xm.size)
                tm = np.asarray(tm, float)
                ok_m_all = ok_m_all and bool(ok_m)
                e = tm - c_vec
                e_hist.append(e)
                if it < window_iters:  # no need to move the window after the last pass
                    w_ref = ndi_shift(
                        win, (-e[1], -e[0]), order=3, mode="constant", cval=0.0
                    )

            # convergence of the window iteration: change between the last two passes
            if len(e_hist) >= 2:
                win_delta = float(np.max(np.abs(e_hist[-1] - e_hist[-2])))
            else:
                win_delta = 0.0

            last_sigma = float(s)
            last_win_delta = float(win_delta)

            # bias-cancelled residual, sign flipped so it matches the offsets that
            # run() reports (the raw correlation lag has the opposite sign)
            eps = -(tm - tc)
            # with a guess, the refined main peak must stay inside the localized region
            in_region = initial_guess is None or bool(
                np.all(np.abs(e_hist[-1] - g_vec) <= hw)
            )

            # the main peak must be (almost) as significant as the control peak
            snr_ratio = float(snr_m / snr_c) if snr_c > 0 else np.nan

            # and have (almost) the same shape: elliptical gaussian fits to both,
            # on the correlation maps of the last window iteration
            fit_halfwidth = self._peak_fit_halfwidth(s, c_vec)
            fit_m = self.fit_peak_shape(c_main, t0m, fit_halfwidth)
            fit_c = self.fit_peak_shape(c_ctrl, t0c, fit_halfwidth)
            shape_dev = self._peak_shape_deviation(fit_m, fit_c)

            # (NaN ratios / deviations fail the comparisons, i.e. reject the sigma)
            ok = (
                in_region
                and ok_m_all
                and ok_c
                and np.all(np.abs(tc - c_vec) < ctrl_tol)
                and snr_m > snr_min
                and snr_c > snr_min
                and win_delta <= window_tol
                and snr_ratio >= snr_ratio_min
                and shape_dev <= shape_tol
            )
            rows.append(
                (
                    s,
                    *tm,
                    *tc,
                    *eps,
                    snr_m,
                    snr_c,
                    float(ok),
                    win_delta,
                    snr_ratio,
                    shape_dev,
                )
            )

            # check the last n_agree sigmas: consecutive, all accepted, and the
            # full spread within tolerance in BOTH x and y
            if len(rows) >= n_agree:
                window = np.array(rows[-n_agree:], float)
                if np.all(window[:, 9] == 1):
                    spread = np.ptp(window[:, 5:7], axis=0)  # max - min per axis
                    if np.all(spread <= agree_tol):
                        group = window
                        break

        table = np.array(rows, float)
        raw = -(np.median(table[:, 1:3], axis=0) - c_vec)  # same sign as `shift`

        # ------------------------------------------------------------------
        # QA data for the final correlation peak
        # ------------------------------------------------------------------

        peak = None
        if (
            save_QA
            and obs_name is not None
            and c_main is not None
            and c_ctrl is not None
        ):
            zoom_halfwidth = max(hw, 5.0)

            peak = dict(
                sigma=last_sigma,
                main=dict(
                    **self._peak_map_data(c_main, t0m, zoom_halfwidth),
                    integer_peak=tuple(float(v) for v in t0m),
                    refined_peak=tuple(float(v) for v in tm),
                    expected=tuple(float(v) for v in center),
                    snr=float(snr_m),
                    peak_value=float(pk_m),
                    ok=bool(ok_m_all),
                    fit=fit_m,
                ),
                control=dict(
                    **self._peak_map_data(c_ctrl, t0c, zoom_halfwidth),
                    integer_peak=tuple(float(v) for v in t0c),
                    refined_peak=tuple(float(v) for v in tc),
                    expected=tuple(float(v) for v in c_vec),
                    snr=float(snr_c),
                    peak_value=float(pk_c),
                    ok=bool(ok_c),
                    fit=fit_c,
                ),
                snr_min=float(snr_min),
                snr_ratio=float(snr_ratio),
                snr_ratio_min=float(snr_ratio_min),
                shape_deviation=float(shape_dev),
                shape_tol=float(shape_tol),
                search_halfwidth=float(hw),
                window_delta=float(last_win_delta),
                control_offset=tuple(float(v) for v in c_vec),
            )

        if group is not None:
            eps_group = group[:, 5:7]
            n = len(group)
            est = eps_group.mean(axis=0)
            spread = np.ptp(eps_group, axis=0)
            sem = eps_group.std(axis=0, ddof=1) / np.sqrt(n)
            err = np.maximum(sem, spread / 2.0)  # conservative: never below half-spread
            result = dict(
                shift=tuple(est),
                shift_err=tuple(err),
                spread=tuple(spread),
                success=True,
                table=table,
                raw=tuple(raw),
                n_used=n,
                sigmas_used=tuple(group[:, 0]),
                localized=initial_guess is not None,
                peak=peak,
            )
        else:
            result = dict(
                shift=(np.nan, np.nan),
                shift_err=(np.nan, np.nan),
                spread=(np.nan, np.nan),
                success=False,
                table=table,
                raw=tuple(raw),
                n_used=0,
                sigmas_used=(),
                localized=initial_guess is not None,
                peak=peak,
            )

        if save_QA and obs_name is not None:
            self.plot_translation_peak(result, obs_name, show=show_QA)

        return result

    def _narrow_scan_sigmas(
        self,
        sigma_grid,
        direct_result,
        initial_guess_translation,
        initial_guess_rotation,
        obs_name,
    ):
        """
        Sigma range for the translation estimates of the angle scan.

        If the direct fit agrees with the initial guess (translation within
        `translation_guess_tol` px in x and y, rotation within
        `rotation_guess_tol` degrees), the scan only searches the sigmas the
        direct fit's translation agreed on (`sigmas_used`), extended by
        `scan_sigma_margin` grid steps on each side: the solution is already
        known to be near the guess, so the scales that worked there are the
        ones worth trying, and every scanned angle gets much cheaper.
        Otherwise (no guess, disagreement, no agreeing sigmas recorded) the
        full sigma grid is returned.
        """
        (tx, ty), rot, _, _, _ = direct_result

        guess_ok = (
            initial_guess_translation is not None
            and all(g is not None for g in initial_guess_translation)
            and initial_guess_rotation is not None
        )

        if not guess_ok:
            return sigma_grid

        dt = np.abs(np.array([tx, ty]) - np.asarray(initial_guess_translation, float))
        drot = abs(rot - float(initial_guess_rotation))

        if np.any(dt > self.translation_guess_tol) or drot > self.rotation_guess_tol:
            if self.logger:
                self.logger.info(
                    f"Direct fit for {obs_name} does not agree with the initial "
                    f"guess (dtx = {dt[0]:.3f}, dty = {dt[1]:.3f} px, drot = "
                    f"{drot:.4f} degrees), scanning the full sigma range"
                )
            return sigma_grid

        result = self._last_translation_result
        sigmas_used = () if result is None else result.get("sigmas_used", ())

        if len(sigmas_used) == 0:
            return sigma_grid

        # grid indices of the agreeing sigmas, extended by the margin
        indices = [int(np.argmin(np.abs(sigma_grid - s))) for s in sigmas_used]
        lo = max(0, min(indices) - int(self.scan_sigma_margin))
        hi = min(len(sigma_grid) - 1, max(indices) + int(self.scan_sigma_margin))

        if self.logger:
            self.logger.info(
                f"Direct fit for {obs_name} agrees with the initial guess, scan "
                f"sigmas narrowed to [{sigma_grid[lo]:.3f}, {sigma_grid[hi]:.3f}] "
                f"({hi - lo + 1} of {len(sigma_grid)})"
            )

        return sigma_grid[lo : hi + 1]

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
        scan_for_angles=False,
        **kwargs,
    ):
        """
        Align the target image to the reference image.

        By default the rotation is estimated directly with
        `estimate_rotation_fixed_scale`. If `scan_for_angles=True`, it is found
        in two stages instead: a ternary search over the full angle range
        locates the approximate minimum of the difference image, then a fine
        scan of `fine_scan_points` equally spaced angles within
        +/- `fine_scan_halfwidth` of it is fitted with a two-pass parabola
        (fit, reject outliers, refit). The translation is the linear trend of
        tx(theta), ty(theta) over the fine scan evaluated at the resulting
        angle (error: rms scatter of the individual translations around the
        trend); only without a trend is it estimated again at that angle.

        With the scan, the direct fit is run first (`direct_fit_before_scan`);
        if it fails, so does the alignment. If it agrees with the initial guess,
        the scan only searches the sigmas around the ones the direct fit
        agreed on (`rotation_guess_tol`, `scan_sigma_margin`).

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
            Initial guess for the translation (x, y) in pixels, in the same
            convention as the returned offsets. If both values are given, the
            translation peak search is localized around the guess (+/-
            `translation_guess_tol`). Default (None, None) searches everywhere.

        initial_guess_rotation : float, optional
            Initial guess for the rotation in degrees. Currently not used.

        skip_rotation_estimation : bool, optional
            If True, the rotation is fixed at zero. This takes precedence over
            `scan_for_angles`.

        scan_for_angles : bool, optional
            If True, use the shared angle scan. If False (default), use the
            direct `estimate_rotation_fixed_scale` estimator.

        **kwargs : dict
            Additional keyword arguments specific to the alignment method.

        Returns
        -------
        offsets: tuple
            The calculated offsets (x_offset, y_offset) in pixels.

        rotation: float
            The calculated rotation in degrees: the rotation to apply to the
            input frame to align it with the reference, positive
            counter-clockwise as the frame is displayed with origin upper
            (numpy array as read by astropy.io.fits, scipy.ndimage.rotate
            convention). Applied about the frame centre, before the offsets.

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

        # Optional localization of the translation search. estimate_translation
        # uses the same sign convention as the offsets returned here.
        fft_guess = None
        if initial_guess_translation is not None and not any(
            g is None for g in initial_guess_translation
        ):
            fft_guess = (
                float(initial_guess_translation[0]),
                float(initial_guess_translation[1]),
            )
            self.logger.info(
                f"Localizing translation search around guess {tuple(initial_guess_translation)} "
                f"+/- {self.translation_guess_tol} px"
            )

        sigma_grid = np.geomspace(self.sigma_start, self.sigma_end, self.sigma_num)

        def _estimate_translation(
            reference,
            rotated_input,
            save_QA=False,
            show_QA=False,
            obs_name=None,
            sigmas=None,
        ):
            return self.estimate_translation(
                reference,
                rotated_input,
                control_offset=(10.2, 10.2),
                sigmas=sigma_grid if sigmas is None else sigmas,
                ratio=1.6,
                gamma=0.0,
                taper=0.25,
                ctrl_tol=self.control_tol,
                snr_min=10.0,
                search_halfwidth=20.0,
                initial_guess=fft_guess,
                guess_tol=self.translation_guess_tol,
                snr_ratio_min=self.snr_ratio_min,
                shape_tol=self.peak_shape_tol,
                obs_name=obs_name,
                save_QA=save_QA,
                show_QA=show_QA,
            )

        # ------------------------------------------------------------------
        # Rotation estimate: shared angle scan, method-specific non-scan
        # estimator, or no rotation at all.
        # ------------------------------------------------------------------

        scan_for_angles = bool(scan_for_angles) and not skip_rotation_estimation

        if skip_rotation_estimation:

            if self.logger:
                self.logger.info(
                    "Skipping rotation estimation as per user request."
                )

            rot = 0.0
            rot_err = 0.0

        elif scan_for_angles:

            # sigmas searched by the translation estimate of every scanned
            # angle; narrowed below if the direct fit agrees with the guess
            scan_sigmas = sigma_grid

            # cheap check first: the direct fit. If even that fails, the slow
            # angle scan is not run and the alignment is concluded as failed
            # (the direct fit already wrote the failed alignment QA plot).
            if self.direct_fit_before_scan:

                if self.logger:
                    self.logger.info(
                        f"Direct FFT fit for {obs_name} before the angle scan ..."
                    )

                direct_result = self.run(
                    reference_frame_backup,
                    input_frame_backup,
                    obs_name,
                    reference_wcs=reference_wcs,
                    input_wcs=input_wcs,
                    initial_guess_translation=initial_guess_translation,
                    initial_guess_rotation=initial_guess_rotation,
                    save_QA=save_QA,
                    show_QA=show_QA,
                    skip_rotation_estimation=False,
                    scan_for_angles=False,
                    **kwargs,
                )

                if not direct_result[-1]:
                    if self.logger:
                        self.logger.warning(
                            f"Failed FFT aligment for {obs_name}: the direct fit "
                            f"failed, angle scan not run"
                        )

                    return self.FAILED_RETURN

                if self.logger:
                    self.logger.info(
                        f"Direct FFT fit for {obs_name} succeeded "
                        f"(rotation {direct_result[1]:.4f} degrees), "
                        f"running the angle scan ..."
                    )

                scan_sigmas = self._narrow_scan_sigmas(
                    sigma_grid,
                    direct_result,
                    initial_guess_translation,
                    initial_guess_rotation,
                    obs_name,
                )

            def _evaluate_angle(angle):
                diff, result = self._scan_difference_metric(
                    reference_frame_backup,
                    input_frame_backup,
                    angle,
                    lambda rotated_input: _estimate_translation(
                        reference_frame_backup, rotated_input, sigmas=scan_sigmas
                    ),
                )

                # one plot per scanned angle is a lot of files, so this is opt-in
                if self.plot_scan_progression and result is not None:
                    self.plot_translation_progression(
                        result,
                        f"{obs_name}_angle_{angle:.4f}",
                        show=show_QA,
                    )

                if result is None or not result.get("success", False):
                    if self.logger:
                        self.logger.warning(
                            f"Failed to estimate translation for {angle} degrees"
                        )
                    return diff

                tx, ty = result["shift"]
                tx_err, ty_err = result["shift_err"]

                if self.logger:
                    self.logger.info(
                        f"Estimated translation: ({tx:.2f}, {ty:.2f}) +/- "
                        f"({tx_err:.2f}, {ty_err:.2f}) for {angle} degrees"
                    )

                return diff

            scan_result = self._scan_for_best_angle(
                _evaluate_angle,
                obs_name,
                save_QA=save_QA,
                show_QA=show_QA,
            )

            if scan_result is None:
                if self.logger:
                    self.logger.warning(
                        f"Failed FFT aligment for {obs_name}: angle scan "
                        f"found no rotation"
                    )

                self.plot_failed_aligment(
                    reference_frame_backup,
                    input_frame_backup,
                    obs_name,
                    show=show_QA,
                )

                return self.FAILED_RETURN

            rot = scan_result["angle"]
            rot_err = scan_result["angle_err"]

            # The final translation is the tx(theta), ty(theta) trend at the
            # final angle: it averages the translation estimates of the whole
            # fine scan instead of relying on one more estimate at one angle,
            # and costs nothing. Error: rms scatter of the individual
            # estimates around the trend. Without a trend (too few usable
            # translations, or smoothing disabled) the translation is estimated
            # at the final angle as in the direct path.
            if scan_result.get("trend_translation") is not None:
                trend_tx, trend_ty = scan_result["trend_translation"]
                trend_tx_err, trend_ty_err = scan_result["trend_translation_err"]

                if self.logger:
                    self.logger.info(
                        f"Final translation for {obs_name} from the trend at "
                        f"{rot:.4f} degrees: ({trend_tx:.4f}, {trend_ty:.4f}) +/- "
                        f"({trend_tx_err:.4f}, {trend_ty_err:.4f}) px"
                    )

                self.plot_difference_image(
                    reference_frame_backup,
                    input_frame_backup,
                    trend_tx,
                    trend_ty,
                    rot,
                    obs_name,
                    show=False,
                )

                return (
                    (trend_tx, trend_ty),
                    rot,
                    (trend_tx_err, trend_ty_err),
                    rot_err,
                    True,
                )

            if self.logger:
                self.logger.warning(
                    f"No translation trend for {obs_name}, estimating the "
                    f"translation at the final angle"
                )

        else:

            # non-scan rotation estimate (the historical FFT rotation path)
            rot_dict = self.estimate_rotation_fixed_scale(
                reference_frame_backup,
                input_frame_backup,
                extra_rotation=self.extra_rotation,
                sigmas=np.geomspace(
                    self.sigma_start, self.sigma_end, self.sigma_num
                ),
                ratio=1.6,
                ctrl_tol=self.control_tol,
                snr_min=10.0,
                search_halfwidth=1.0,
                min_ok=3,
                pad=2,
                upsample=64,
            )

            table = rot_dict["table"]

            if self.logger:
                np.set_printoptions(precision=4, suppress=True)
                self.logger.info(
                    "\nper-sigma table for rotation case: sigma, a_main, a_ctrl, "
                    "eps, snr_main, snr_ctrl, ok\n" + str(table)
                )

            # keep entries where the control rotation is recovered and both
            # peak SNRs pass the threshold, then use the highest main SNR
            filtered_entries = table[
                (np.abs(table[:, 2] - self.extra_rotation) <= self.control_tol)
                & (table[:, 4] > self.snr_min)
                & (table[:, 5] > self.snr_min)
            ]

            if not rot_dict["success"] or len(filtered_entries) == 0:
                if self.logger:
                    self.logger.warning(
                        "No valid entries found in the per-sigma rotation table "
                        "that satisfy the control tolerance and SNR thresholds. "
                        "No rotation estimated - alignment failed."
                    )

                self.plot_failed_aligment(
                    reference_frame_backup,
                    input_frame_backup,
                    obs_name,
                    show=show_QA,
                )

                return self.FAILED_RETURN

            best_entry = filtered_entries[np.argmax(filtered_entries[:, 4])]
            best_sigma = best_entry[0]
            best_eps = best_entry[3]

            rot = -best_eps
            rot_err = 0.0

            if self.logger:
                self.logger.info(
                    f"Non-scan rotation estimate: sigma = {best_sigma:.4f}, "
                    f"eps = {best_eps:.4f}, rotation = {rot:.4f} degrees"
                )

            # QA for the direct rotation fit
            plt.figure(figsize=(8, 6))
            plt.plot(
                filtered_entries[:, 0],
                filtered_entries[:, 3],
                marker="o",
                linestyle="-",
                color="blue",
                label="eps",
            )
            plt.axvline(
                x=best_sigma,
                color="red",
                linestyle="--",
                label=f"Best sigma: {best_sigma:.4f}",
            )
            plt.xlabel("sigma")
            plt.ylabel("eps")
            plt.title(f"eps vs sigma for {obs_name}")
            plt.legend()
            plt.grid()

            if save_QA:
                plt.savefig(f"{self.QA_path}/eps_vs_sigma_{obs_name}.png")
                if self.logger:
                    self.logger.info(
                        f"Saved QA plot of eps vs sigma to "
                        f"{self.QA_path}/eps_vs_sigma_{obs_name}.png"
                    )

            if show_QA:
                plt.show()
            else:
                plt.close()

        # final pass - shift by the modelled angle and estimate tx and ty

        # reset frames
        reference_frame = reference_frame_backup
        input_frame = input_frame_backup

        rotated_input_frame = rotate(
                        input_frame, rot, reshape=False, order=3, cval=median
                    )
        

        # Only the final translation estimate gets the peak QA plot; the many
        # calls made during the angle scans use the default save_QA=False.
        result = _estimate_translation(
            reference_frame,
            rotated_input_frame,
            save_QA=save_QA,
            show_QA=show_QA,
            obs_name=obs_name,
        )
        self._last_translation_result = result

        if  not result["success"]:
            if self.logger:
                self.logger.warning(
                    f"Failed FFT aligment for {obs_name}: final translation "
                    f"estimate failed"
                )

            self.plot_failed_aligment(
                reference_frame, input_frame, obs_name, show=show_QA
            )

            return self.FAILED_RETURN

        self.plot_translation_progression(result, obs_name, show = show_QA)

        tx, ty = result["shift"]
        tx_err, ty_err = result["shift_err"]

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
    """
    Alignment with optical flow.

    spacepylot computes the dense TV-L1 flow field between the input and the
    reference frame. The rigid transform (rotation + translation, or
    translation only) is fitted to a sparse grid of flow vectors here, in
    `fit_flow_transform`, and not with spacepylot's own homography step: that
    step refers the translation to the corner of its cropped frame instead of
    the frame centre, and its translation-only fit is broken with
    skimage >= 0.26 (details in `fit_flow_transform`).

    Uncertainties: the errors reported by `run` come from a spatial block
    bootstrap of the fitted flow vectors, see `bootstrap_uncertainty`.
    """

    def __init__(
        self,
        num_per_dimension=50,
        residual_threshold=0.5,
        bootstrap_tiles=5,
        bootstrap_samples=200,
        bootstrap_seed=1,
        bootstrap_min_tiles=10,
        logger=None,
        QA_path=None,
    ):
        """
        Parameters
        ----------
        num_per_dimension : int
            Number of flow vectors per axis that are sampled from the dense flow
            field and fitted by spacepylot (default 50, spacepylot's default,
            i.e. 50 x 50 = 2500 points).

        residual_threshold : float
            RANSAC inlier threshold in pixels (default 0.5, spacepylot's
            default). Passed to spacepylot, and used again to recover the
            inliers for the bootstrap.

        bootstrap_tiles : int
            The frame is split into bootstrap_tiles x bootstrap_tiles tiles,
            which are the blocks resampled by the bootstrap (default 5). Tiles
            must be larger than the correlation length of the flow field (a few
            PSF widths), and there must be enough of them for a stable
            bootstrap.

        bootstrap_samples : int
            Number of bootstrap resamples (default 200).

        bootstrap_seed : int
            Seed of the bootstrap random generator, so a rerun gives the same
            errors (default 1).

        bootstrap_min_tiles : int
            Minimum number of tiles that contain inliers. Below this the
            bootstrap is not trusted: a warning is logged and the errors are
            set to 0 (unknown), as before the bootstrap existed (default 10).
        """
        super().__init__("OpticalFlow", "0.0.0", logger=logger, QA_path=QA_path)

        self.num_per_dimension = num_per_dimension
        self.residual_threshold = residual_threshold
        self.bootstrap_tiles = bootstrap_tiles
        self.bootstrap_samples = bootstrap_samples
        self.bootstrap_seed = bootstrap_seed
        self.bootstrap_min_tiles = bootstrap_min_tiles

        if self.logger:
            self.logger.info(
                f"Initialized {self.name} alignment method, version {self.version}."
            )

    # ------------------------------------------------------------------
    # Fitting the flow field
    # ------------------------------------------------------------------

    @staticmethod
    def _rigid_fit(src, dst, rotation=True):
        """
        Least-squares fit of dst = R @ src + t to point pairs.

        With rotation=True this is the closed-form (Umeyama / Kabsch) solution
        for a rotation + translation without scaling, i.e. the model of skimage's
        EuclideanTransform. With rotation=False it is a pure translation,
        t = mean(dst - src), which is what spacepylot's TranslationTransform is
        meant to be (but is not with skimage >= 0.26, see `fit_flow_transform`).

        Parameters
        ----------
        src, dst : (N, 2) arrays
            (x, y) coordinates of the points and of where they moved to.

        rotation : bool
            Fit rotation + translation (True) or translation only (False).

        Returns
        -------
        np.ndarray
            3x3 homogeneous matrix M with dst = M @ [x, y, 1].
        """
        src = np.asarray(src, float)
        dst = np.asarray(dst, float)

        params = np.eye(3)

        if not rotation:
            params[:2, 2] = np.mean(dst - src, axis=0)
            return params

        src_mean = src.mean(axis=0)
        dst_mean = dst.mean(axis=0)

        A = (dst - dst_mean).T @ (src - src_mean) / len(src)
        U, _, Vt = np.linalg.svd(A)

        # keep a proper rotation (no reflection)
        D = np.diag([1.0, np.sign(np.linalg.det(U) * np.linalg.det(Vt))])
        R = U @ D @ Vt

        params[:2, :2] = R
        params[:2, 2] = dst_mean - R @ src_mean

        return params

    @staticmethod
    def _correction_parameters(params, centre):
        """
        Convert a fitted flow model into (tx, ty, rot) in the convention of
        `run` (and of the whole pipeline).

        The flow model maps reference pixels to where they are found in the
        input frame (reference -> input). The alignment reported by `run` is the
        correction to apply to the input frame (input -> reference): rotate by
        `rot` about the frame centre, then shift by (tx, ty). So the model is
        inverted and decomposed about the frame centre with
        `geometry.matrix_to_parameters`, the inverse of
        `geometry.transformation_matrix`.
        """
        return tuple(
            float(v) for v in matrix_to_parameters(np.linalg.inv(params), centre)
        )

    def fit_flow_transform(self, op, frame_shape, rotation=True):
        """
        Fit the alignment to the optical-flow field of spacepylot.

        What spacepylot is used for
        ---------------------------
        Only for the dense TV-L1 flow field (`op.optical_flow()`, with its
        filtering: histogram equalisation, high-pass and a border of
        `remove_boundary_pixels` (25) px removed from every side). The rigid
        transform is fitted here instead of with spacepylot's homography step,
        because that step gives wrong parameters for this pipeline in two ways:

        1. Rotation centre. spacepylot fits on the cropped frame, so its
           translation belongs to a rotation about the corner of the cropped
           frame, pixel (25, 25) of the full frame. The pipeline applies the
           rotation about the frame centre. The difference is
           rotation [rad] x ~175 px, e.g. 0.3 px for 0.09 degrees and 0.55 px
           in x and y for 0.25 degrees (tested on synthetic data).
        2. Translation-only fits. With skimage >= 0.26, RANSAC with
           spacepylot's TranslationTransform does not fit a pure translation
           (the fitted matrix has a non-identity linear part of ~1e-3).
           Read about the cropped corner, the translation is off by up to
           several pixels (synthetic pure shift: scatter of 0.8-1.0 px over
           noise realisations, against 0.02 px for a least-squares fit of the
           same vectors). This affected `skip_rotation_estimation` and the
           angle scan.

        Fitting
        -------
        1. Sample the flow field on the same sparse grid as spacepylot
           (`get_sparse_vector_grid`, `num_per_dimension` per axis) and add the
           border back, so the point pairs are in full-frame pixel coordinates
           (x = column, y = row of the numpy array): src = reference grid,
           dst = src + flow = where that reference pixel is found in the input.
        2. Initial inliers. Rotation + translation: RANSAC with skimage's
           EuclideanTransform (min 3 samples, `residual_threshold`, 1000 trials,
           fixed seed), the same robust step spacepylot uses. Translation only:
           the points within `residual_threshold` of the median flow vector.
        3. Refine: least-squares fit of the inliers (`_rigid_fit`), recompute
           the residuals of ALL points, inliers = residual < `residual_threshold`;
           repeat until the inlier set no longer changes (at most 10 times).
           The final fit is a plain least-squares fit of the inliers, which is
           also what the bootstrap refits, so the bootstrap is centred on the
           reported result.
        4. Convert to the correction of the input frame, rotation about the
           frame centre (`_correction_parameters`).

        Parameters
        ----------
        op : spacepylot.alignment.AlignOpticalFlow
            Set up with (input_frame, reference_frame). The flow is computed
            here.

        frame_shape : tuple
            Shape (ny, nx) of the full input / reference frame.

        rotation : bool
            Fit rotation + translation (True) or translation only (False).

        Returns
        -------
        dict or None
            tx, ty, rot (rot = 0 for a translation-only fit), and for the
            bootstrap: src, dst, inliers, centre, rotation, params (the fitted
            reference -> input matrix). None if there are fewer than 3 inliers.
        """
        # 1) flow field and the sparse point pairs, in full-frame coordinates
        op.v, op.u = op.optical_flow()

        border = float(op.filter_params.get("remove_boundary_pixels", 0))

        y_sparse, x_sparse, v_sparse, u_sparse = spacepylot_utils.get_sparse_vector_grid(
            op.v, op.u, self.num_per_dimension
        )

        src = np.column_stack([np.ravel(x_sparse), np.ravel(y_sparse)]) + border
        dst = src + np.column_stack([np.ravel(u_sparse), np.ravel(v_sparse)])

        finite = np.all(np.isfinite(src), axis=1) & np.all(np.isfinite(dst), axis=1)

        if finite.sum() < 3:
            if self.logger:
                self.logger.warning("Optical flow: fewer than 3 finite flow vectors")
            return None

        def residuals(params):
            res = np.full(len(src), np.inf)
            moved = src[finite] @ params[:2, :2].T + params[:2, 2]
            res[finite] = np.linalg.norm(moved - dst[finite], axis=1)
            return res

        # 2) initial inliers
        if rotation:
            _, ransac_inliers = ransac(
                (src[finite], dst[finite]),
                EuclideanTransform,
                min_samples=3,
                residual_threshold=self.residual_threshold,
                max_trials=1000,
                rng=1,
            )
            inliers = np.zeros(len(src), dtype=bool)
            if ransac_inliers is not None:
                inliers[np.flatnonzero(finite)[ransac_inliers]] = True
        else:
            median_flow = np.median(dst[finite] - src[finite], axis=0)
            inliers = finite & (
                np.linalg.norm(dst - src - median_flow, axis=1) < self.residual_threshold
            )

        # 3) refine with least squares until the inlier set is stable
        params = None
        for _ in range(10):
            if inliers.sum() < 3:
                if self.logger:
                    self.logger.warning(
                        f"Optical flow: only {int(inliers.sum())} inliers within "
                        f"{self.residual_threshold} px, no fit"
                    )
                return None

            params = self._rigid_fit(src[inliers], dst[inliers], rotation=rotation)
            new_inliers = residuals(params) < self.residual_threshold

            if np.array_equal(new_inliers, inliers):
                break
            inliers = new_inliers

        if inliers.sum() < 3:
            return None

        params = self._rigid_fit(src[inliers], dst[inliers], rotation=rotation)

        # 4) correction of the input frame, about the frame centre
        centre = ((frame_shape[1] - 1) / 2.0, (frame_shape[0] - 1) / 2.0)
        tx, ty, rot = self._correction_parameters(params, centre)

        if not rotation:
            rot = 0.0

        if self.logger:
            self.logger.info(
                f"Optical flow fit: {int(inliers.sum())}/{int(finite.sum())} inliers, "
                f"tx = {tx:.4f}, ty = {ty:.4f}"
                + (f", rot = {rot:.5f} degrees" if rotation else " (translation only)")
            )

        return dict(
            tx=tx,
            ty=ty,
            rot=rot,
            src=src,
            dst=dst,
            inliers=inliers,
            centre=centre,
            rotation=rotation,
            params=params,
        )

    # ------------------------------------------------------------------
    # Uncertainties: spatial block bootstrap of the inliers
    # ------------------------------------------------------------------

    def bootstrap_uncertainty(
        self,
        fit,
        obs_name=None,
        reference_frame=None,
        save_QA=True,
        show_QA=False,
    ):
        """
        Uncertainty of an optical-flow alignment from a spatial block bootstrap
        of the inliers of `fit_flow_transform`.

        Why a block bootstrap
        ---------------------
        The rigid transform is fitted to ~2500 flow vectors sampled on a regular
        grid. The formal least-squares error of that fit assumes independent
        points, but the TV-L1 flow is smoothed and the grid points are only a
        few pixels apart, so neighbouring vectors are strongly correlated and
        the formal error is several times too small. Resampling whole tiles
        instead of single points keeps the correlation inside a tile, so the
        bootstrap scatter accounts for it without having to model the
        correlation length. It also picks up a region that pulls the fit (a bad
        patch, residual distortion) as extra scatter.

        Steps
        -----
        1. Assign the inliers to `bootstrap_tiles` x `bootstrap_tiles` tiles
           spanning the flow grid (empty tiles are dropped).
        2. `bootstrap_samples` times: draw as many tiles as there are, with
           replacement, pool their inliers (a tile drawn twice counts twice),
           refit with the same least-squares fit as `fit_flow_transform`
           (`_rigid_fit`) and convert with the same `_correction_parameters`,
           so every draw is in the convention of the reported result.
        3. The error of every parameter is half the 16-84 percentile width of
           the draws (the 1 sigma equivalent, robust to a few odd draws).

        The optical flow is not rerun and the inlier set is kept fixed, so this
        takes well under a second.

        What it does and does not include
        ---------------------------------
        Included: the statistical scatter of the flow vectors, correlated on
        scales up to about a tile. Not included: errors common to the whole
        field (e.g. a bias of the TV-L1 regularisation), and the uncertainty
        of the inlier selection, so the errors are somewhat optimistic (about
        0.7x the scatter over noise realisations in synthetic tests).
        Forward/reverse round trips and dither closures (a -> b -> c vs.
        a -> c) are the empirical check of the reported errors.

        Parameters
        ----------
        fit : dict
            Result of `fit_flow_transform`.

        obs_name : str
            Observation name for logging and the QA plot.

        reference_frame : np.2darray, optional
            Shown under the flow vectors in the QA plot.

        save_QA, show_QA : bool
            Save / show the bootstrap QA plot.

        Returns
        -------
        dict
            success (bool), errors (tx_err, ty_err, rot_err; rot_err = 0 for a
            translation-only fit), draws (bootstrap_samples x 3 array of
            tx, ty, rot), median of the draws, n_inliers, n_points, n_tiles.
            With success False the errors are 0 (unknown).
        """
        src = fit["src"]
        dst = fit["dst"]
        inliers = fit["inliers"]
        rotation = fit["rotation"]
        centre = fit["centre"]
        point_estimate = np.array([fit["tx"], fit["ty"], fit["rot"]], float)

        n_points = int(np.sum(np.all(np.isfinite(dst), axis=1)))
        n_inliers = int(inliers.sum())

        # 1) tiles spanning the flow grid
        n_t = int(self.bootstrap_tiles)
        lo = np.nanmin(src, axis=0)
        span = np.nanmax(src, axis=0) - lo + 1e-9

        tile_x = np.clip(((src[:, 0] - lo[0]) / span[0] * n_t).astype(int), 0, n_t - 1)
        tile_y = np.clip(((src[:, 1] - lo[1]) / span[1] * n_t).astype(int), 0, n_t - 1)
        tile = tile_y * n_t + tile_x

        tiles = [np.flatnonzero(inliers & (tile == t)) for t in range(n_t * n_t)]
        tiles = [members for members in tiles if len(members) > 0]

        fail = dict(
            success=False,
            errors=(0.0, 0.0, 0.0),
            draws=None,
            median=None,
            n_inliers=n_inliers,
            n_points=n_points,
            n_tiles=len(tiles),
        )

        if self.logger:
            self.logger.info(
                f"Bootstrap for {obs_name}: {n_inliers}/{n_points} inliers "
                f"in {len(tiles)} of {n_t * n_t} tiles"
            )

        if len(tiles) < self.bootstrap_min_tiles:
            if self.logger:
                self.logger.warning(
                    f"Bootstrap for {obs_name}: only {len(tiles)} tiles contain "
                    f"inliers, need {self.bootstrap_min_tiles}. Errors unknown, "
                    f"setting them to 0."
                )
            return fail

        # 2) resample tiles and refit
        rng = np.random.default_rng(self.bootstrap_seed)

        draws = np.empty((int(self.bootstrap_samples), 3))

        for b in range(len(draws)):
            picked = rng.integers(0, len(tiles), size=len(tiles))
            idx = np.concatenate([tiles[k] for k in picked])

            draws[b] = self._correction_parameters(
                self._rigid_fit(src[idx], dst[idx], rotation=rotation), centre
            )

        if not rotation:
            draws[:, 2] = 0.0

        # 3) half the 16-84 percentile width = 1 sigma equivalent
        p16, p50, p84 = np.percentile(draws, [16, 50, 84], axis=0)
        errors = 0.5 * (p84 - p16)

        if self.logger:
            self.logger.info(
                f"Bootstrap errors for {obs_name} ({len(draws)} resamples): "
                f"tx_err = {errors[0]:.4f}, ty_err = {errors[1]:.4f} px"
                + (f", rot_err = {errors[2]:.5f} degrees" if rotation else "")
            )

            # a median far from the fit means the fit is pulled by a few tiles
            bias = p50 - point_estimate
            if np.any(np.abs(bias) > errors + 1e-12):
                self.logger.warning(
                    f"Bootstrap for {obs_name}: the median of the draws differs "
                    f"from the fit by more than 1 sigma (dtx = {bias[0]:.4f}, "
                    f"dty = {bias[1]:.4f}, drot = {bias[2]:.5f})"
                )

        result = dict(
            success=True,
            errors=tuple(float(e) for e in errors),
            draws=draws,
            median=tuple(float(m) for m in p50),
            n_inliers=n_inliers,
            n_points=n_points,
            n_tiles=len(tiles),
        )

        if (save_QA or show_QA) and obs_name is not None:
            self.plot_bootstrap(
                result,
                point_estimate,
                src,
                dst,
                inliers,
                tile,
                obs_name,
                rotation=rotation,
                reference_frame=reference_frame,
                save=save_QA,
                show=show_QA,
            )

        return result

    def plot_bootstrap(
        self,
        result,
        point_estimate,
        src,
        dst,
        inliers,
        tile,
        obs_name,
        rotation=True,
        reference_frame=None,
        save=True,
        show=False,
    ):
        """
        QA plot of the bootstrap: the flow vectors of the fit (inliers coloured
        by tile, outliers in red) and the distribution of the draws of tx, ty
        (and rot) with the fit, the median and the +/- 1 sigma band.
        """
        n_panels = 4 if rotation else 3
        fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5.5))

        # ---- flow vectors ---------------------------------------------------
        ax = axes[0]
        if reference_frame is not None:
            vmin, vmax = np.nanpercentile(reference_frame, [5, 95])
            ax.imshow(
                reference_frame,
                cmap="gray",
                origin="upper",
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )

        flow = dst - src
        ax.quiver(
            src[inliers, 0],
            src[inliers, 1],
            flow[inliers, 0],
            flow[inliers, 1],
            tile[inliers],
            cmap="tab20",
            angles="xy",
            scale_units="xy",
            scale=0.1,
            width=0.002,
        )
        if np.any(~inliers):
            ax.scatter(
                src[~inliers, 0],
                src[~inliers, 1],
                s=4,
                color="red",
                label="RANSAC outliers",
            )
            ax.legend(loc="upper right", fontsize=8)

        if reference_frame is None:
            # same orientation as the frame shown with origin upper
            ax.invert_yaxis()
        ax.set_aspect("equal")

        ax.set_title(
            f"Flow vectors x10 ({result['n_inliers']}/{result['n_points']} "
            f"inliers, {result['n_tiles']} tiles)"
        )
        ax.set_xlabel("x [px]")
        ax.set_ylabel("y [px]")

        # ---- draws ------------------------------------------------------------
        labels = [("tx", "px"), ("ty", "px"), ("rotation", "degrees")]

        for k, ax in enumerate(axes[1:]):
            values = result["draws"][:, k]
            fit_value = point_estimate[k]
            median = result["median"][k]
            error = result["errors"][k]

            ax.hist(values, bins=30, color="C0", alpha=0.8)
            ax.axvspan(median - error, median + error, color="white", alpha=0.15,
                       label=f"median +/- 1 sigma ({error:.4f})")
            ax.axvline(median, color="white", linestyle="--", label="median of draws")
            ax.axvline(fit_value, color="orange", label=f"fit ({fit_value:.4f})")

            name, unit = labels[k]
            ax.set_title(f"Bootstrap of {name}")
            ax.set_xlabel(f"{name} [{unit}]")
            ax.set_ylabel("draws")
            ax.legend(fontsize=8)

        fig.suptitle(
            f"Optical flow bootstrap for {obs_name} "
            f"({len(result['draws'])} resamples of {self.bootstrap_tiles}x"
            f"{self.bootstrap_tiles} tiles)"
        )
        fig.tight_layout()

        if save:
            filename = f"{self.QA_path}/bootstrap_{obs_name}_{self.name}.png"
            fig.savefig(filename)
            if self.logger:
                self.logger.info(f"Saved bootstrap QA plot to {filename}")

        if show:
            plt.show()
        else:
            plt.close(fig)

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
        scan_for_angles=False,
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
            If True, the rotation is fixed at zero and optical flow is run
            with a translation-only transform. Takes precedence over
            `scan_for_angles`.

        scan_for_angles : bool, optional
            If True, estimate the rotation with the shared angle scan instead
            of the direct optical-flow rotation. Default is False.

        **kwargs : dict
            Additional keyword arguments specific to the alignment method.

        Returns
        -------
        offsets: tuple
            The calculated offsets (x_offset, y_offset) in pixels.

        rotation: float
            The calculated rotation in degrees: the rotation to apply to the
            input frame to align it with the reference, positive
            counter-clockwise as the frame is displayed with origin upper
            (numpy array as read by astropy.io.fits, scipy.ndimage.rotate
            convention). Applied about the frame centre, before the offsets.

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

        scan_for_angles = bool(scan_for_angles) and not skip_rotation_estimation

        # spacepylot only computes the flow field, the transform is fitted by
        # `fit_flow_transform` (see there for why)
        def _fit(frame, rotation):
            op = align.AlignOpticalFlow(frame, reference_frame)

            # (redirect spacepylot prints to logger)
            with redirect_stdout_to_logger(logger):
                return self.fit_flow_transform(
                    op, reference_frame.shape, rotation=rotation
                )

        if scan_for_angles:

            def _translation_only(rotated_input):
                fit = _fit(rotated_input, rotation=False)

                if fit is None:
                    return dict(shift=(np.nan, np.nan), success=False)

                # errors are only bootstrapped for the final translation
                return dict(
                    shift=(fit["tx"], fit["ty"]),
                    shift_err=(0.0, 0.0),
                    success=True,
                    fit=fit,
                )

            def _evaluate_angle(angle):
                diff, _ = self._scan_difference_metric(
                    reference_frame,
                    input_frame,
                    angle,
                    _translation_only,
                )
                return diff

            scan_result = self._scan_for_best_angle(
                _evaluate_angle,
                obs_name,
                save_QA=save_QA,
                show_QA=show_QA,
            )

            if scan_result is None:
                self.plot_failed_aligment(
                    reference_frame, input_frame, obs_name, show=show_QA
                )
                return self.FAILED_RETURN

            rot = scan_result["angle"]
            rot_err = scan_result["angle_err"]

            median = np.nanmedian(input_frame)

            rotated_input = rotate(
                input_frame,
                rot,
                reshape=False,
                order=3,
                cval=median,
            )

            # the final translation is fitted on the input rotated about the
            # frame centre, so it is already the translation that goes with
            # the scanned rotation
            fit = _fit(rotated_input, rotation=False)

            if fit is not None:
                tx, ty = fit["tx"], fit["ty"]

                if self.logger:
                    self.logger.info(
                        f"Estimated translation: ({tx:.3f}, {ty:.3f}), "
                        f"scanned rotation: {rot:.4f} degrees"
                    )

        else:

            if skip_rotation_estimation:
                self.logger.warning("Skipping rotation estimation as per user request.")

            fit = _fit(input_frame, rotation=not skip_rotation_estimation)

            if fit is not None:
                tx, ty, rot = fit["tx"], fit["ty"], fit["rot"]

                # rotation error from the bootstrap below (0 if not fitted)
                rot_err = 0.0

                self.logger.info(
                    f"Estimated translation: ({tx:.4f}, {ty:.4f}), "
                    f"Estimated rotation: {rot:.5f}"
                )

        if fit is None:
            self.logger.warning(
                f"Optical flow fit failed for {obs_name} - setting aligment as failed."
            )
            self.plot_failed_aligment(
                reference_frame, input_frame, obs_name, show=show_QA
            )
            return self.FAILED_RETURN

        if np.abs(tx - 0) < 1e-3 and np.abs(ty - 0) < 1e-3 and np.abs(rot - 0) < 1e-3:
            self.logger.warning(
                "No offset or rotation detected in optical flow - setting aligment as failed."
            )
            self.plot_failed_aligment(
                reference_frame, input_frame, obs_name, show=show_QA
            )
            return self.FAILED_RETURN

        # uncertainties from the spatial block bootstrap of the inliers; for the
        # angle scan only tx, ty (the rotation error comes from the scan)
        bootstrap = self.bootstrap_uncertainty(
            fit,
            obs_name=obs_name,
            reference_frame=reference_frame,
            save_QA=save_QA,
            show_QA=show_QA,
        )

        tx_err, ty_err = bootstrap["errors"][:2]

        if fit["rotation"]:
            rot_err = bootstrap["errors"][2]

        # plot difference image

        self.plot_difference_image(
            reference_frame, input_frame, tx, ty, rot, obs_name, show=show_QA
        )

        success = True

        return (
            (tx, ty),
            rot,
            (tx_err, ty_err),
            rot_err,
            success,
        )
