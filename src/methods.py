from abc import ABC, abstractmethod
from asyncio.log import logger
import os
import io
import functools
import astroalign as aa
import sep
from scipy.spatial import cKDTree
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Ellipse
from matplotlib.colors import LinearSegmentedColormap
import os
from scipy.optimize import least_squares
from scipy.ndimage import shift, rotate, gaussian_filter
import numpy.fft as fft
from scipy.ndimage import maximum_filter
from geometry import (
    reproject_image_to_target_wcs,
    make_window_wcs,
    matrix_to_parameters,
    transformation_matrix,
)
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
from spacepylot.params import pcc_params as spacepylot_filter_params
import logging

from stats import ternary_search, robust_residual_threshold


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


# arrows drawn over grey frames (flow fields, point source displacements and
# residuals): the bright part of viridis only, and a thin black outline
ARROW_CMAP = LinearSegmentedColormap.from_list(
    "viridis_bright", plt.get_cmap("viridis")(np.linspace(0.35, 1.0, 256))
)
ARROW_OUTLINE = dict(edgecolor="black", linewidth=0.6)


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
        scan_max_shift=10,
        scan_shift_slack=3,
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
        angle_fit_model="auto",
        fit_min_contrast=3.0,
        fit_max_extrapolation=1.0,
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

        scan_max_shift : int
            The comparison window of the scan is clipped so that it stays
            covered by the input for translations up to this many pixels
            (default 10). Without this, on frames smaller than
            2 x `window_compare_size` the window is the whole frame and any
            non-zero translation leaves the angle unscored. This is the
            minimum: for larger translations the room is set from the
            expected translation of the scan, see `scan_shift_slack`.

        scan_shift_slack : int
            Extra room (pixels) on top of the expected translation of a scan
            (default 3), for the drift of the translation with the angle.
            The expected translation is the one known before the scan (e.g.
            the direct fit or the initial guess), otherwise the first
            translation measured in the scan. The room is then
            max(`scan_max_shift`, max(|tx|, |ty|) + `scan_shift_slack`), the
            same window for every angle of the scan.

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
            Model fitted to the fine scan to find the minimum: "auto" (default)
            fits both a parabola and a hyperbola, f = sqrt(a (x - x_v)^2 + b) + c,
            and uses the one with the lower AICc; "parabola" or "hyperbola"
            force one model. The difference metric is a rounded V over the fine
            scan: when the scan reaches the straight arms a parabola is biased
            towards the longer side, when it only sees the rounded bottom the
            hyperbola is degenerate and does not converge. See
            `_fit_angle_parabola`.

        fit_min_contrast : float
            Shape check of the fine scan fit: the model must rise from its
            minimum to the higher fine scan edge by more than this many robust
            residual sigmas (default 3), else the curve is flat within the noise
            and the fit is rejected.

        fit_max_extrapolation : float
            Shape check of the fine scan fit: the minimum may lie at most this
            many fine scan widths beyond the edges of the fine scan (default 1).
        """

        if angle_fit_model not in ("auto", "hyperbola", "parabola"):
            raise ValueError(
                f"angle_fit_model must be 'auto', 'hyperbola' or 'parabola', got "
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
        self.scan_max_shift = scan_max_shift
        self.scan_shift_slack = scan_shift_slack
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
        self.fit_min_contrast = fit_min_contrast
        self.fit_max_extrapolation = fit_max_extrapolation

        # translations per scanned angle and the frames of the current scan,
        # filled by _scan_difference_metric
        self._scan_translations = {}
        self._scan_frames = None

        # room (pixels) for the translation in the comparison window of the
        # current scan, see `_scan_shift_room`
        self._scan_room = None

        self.angle_scan_array = np.linspace(
            self.angle_scan_start, self.angle_scan_end, self.angle_scan_steps
        )

        # this is the return value when the alignment fails
        self.FAILED_RETURN = (None, None), None, (None, None), None, False

        # QA plots made during `run` are buffered and only written if the
        # alignment succeeds, see `_buffer_QA_during_run`
        self._QA_buffer = None
        self._QA_depth = 0

    # ------------------------------------------------------------------
    # QA plots of failed alignments
    #
    # A failed alignment gets only the failed difference image
    # (`plot_failed_aligment`). Other QA plots may already have been made by
    # then (angle scan, FFT peak / progression, the direct FFT fit before the
    # scan, ...), so every run() of a subclass is wrapped: all QA figures
    # saved during the run go through `_save_QA_figure`, which keeps them in
    # memory (as rendered images); they are written when the run returns
    # with success, and dropped when it fails or raises.
    # `plot_failed_aligment` writes directly. Nested runs (the FFT direct fit
    # before the angle scan) share the buffer of the outermost run. Plots
    # made outside run() (e.g. the pipeline's composed difference images)
    # are written directly.
    # ------------------------------------------------------------------

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "run" in cls.__dict__:
            cls.run = AligmentMethod._buffer_QA_during_run(cls.__dict__["run"])

    @staticmethod
    def _buffer_QA_during_run(run):
        @functools.wraps(run)
        def wrapper(self, *args, **kwargs):
            outermost = getattr(self, "_QA_depth", 0) == 0
            if outermost:
                self._QA_buffer = []
            self._QA_depth = getattr(self, "_QA_depth", 0) + 1

            try:
                result = run(self, *args, **kwargs)
            except BaseException:
                self._QA_depth -= 1
                if outermost:
                    self._QA_buffer = None
                raise

            self._QA_depth -= 1

            if outermost:
                buffered, self._QA_buffer = self._QA_buffer, None
                success = bool(result[-1]) if isinstance(result, tuple) else False

                if success:
                    for filename, data in buffered:
                        with open(filename, "wb") as f:
                            f.write(data)
                elif buffered and self.logger:
                    self.logger.info(
                        f"Alignment failed: {len(buffered)} QA plot(s) of the "
                        f"attempt not written, only the failed difference image "
                        f"is kept."
                    )

            return result

        return wrapper

    def _save_QA_figure(self, filename, fig=None):
        """
        Save a QA figure (`fig`, default the current pyplot figure): kept in
        memory while a run() is in progress (written only if it succeeds),
        written directly otherwise.
        """
        fig = plt.gcf() if fig is None else fig

        if self._QA_buffer is None:
            fig.savefig(filename)
            return

        data = io.BytesIO()
        fig.savefig(data, format=os.path.splitext(filename)[1].lstrip(".") or "png")
        self._QA_buffer.append((filename, data.getvalue()))

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
        self._save_QA_figure(f"{self.QA_path}/difference_image_{obs_name}_{self.name}.png")

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

        # room for the translation in the comparison window: fixed for the
        # whole scan, from the expected translation or else the first one
        if self._scan_room is None:
            self._scan_room = self._scan_shift_room((tx, ty), source="first measured")

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

        # central comparison window, clipped to the frame size minus room for
        # the blur margin and the translation (`_scan_room`, the same window
        # for every angle)
        ny, nx = reference_data.shape
        cy, cx = ny // 2, nx // 2
        room = int(np.ceil(4 * self.scan_blur_sigma)) + int(self._scan_room)
        half_y = int(max(min(self.window_compare_size, cy - room), min(10, cy)))
        half_x = int(max(min(self.window_compare_size, cx - room), min(10, cx)))

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
        Fit of the fine scan to find its minimum. (The name is historical: with
        the default `angle_fit_model` = "auto" both a parabola and a hyperbola
        are fitted and the better one is used.)

        Models
        ------
        "parabola": f = A x^2 + B x + C, minimum at x = -B / 2A (A > 0).

        "hyperbola": f = sqrt(a (x - x_v)^2 + b) + c, a > 0, b >= 0.
        Close to the minimum (|x - x_v| << sqrt(b / a)) this is a parabola,
        c + sqrt(b) + a / (2 sqrt(b)) (x - x_v)^2; further out it becomes the
        straight arms c + sqrt(a) |x - x_v| of a V (b = 0 is a sharp V). The
        minimum is exactly at x = x_v, a fit parameter. Fitted with
        scipy.optimize.least_squares, started from a parabola fit.

        Why both: the difference metric over the fine scan is a rounded V. When
        the fine scan reaches the straight arms, a parabola fitted to a
        lopsided V is pulled towards the longer side (synthetic test: 0.310
        instead of 0.300 degrees, ~12x the statistical error), and the
        hyperbola is needed. When the fine scan only sees the rounded bottom,
        the scan is a parabola and the hyperbola is degenerate: it can only
        reach the parabola in the limit a, b -> infinity, c -> -infinity (with
        a / (2 sqrt(b)) fixed), so its parameters run away and the optimizer
        never converges, although its x_v is fine. On real data this was the
        cause of most failed fine scan fits (34 of 81 in a hybrid FFT run).

        Model choice ("auto")
        ---------------------
        Each model gets its own multi-pass fit (`_multipass_fit`), then the
        shape check below. If both pass, the one with the lower small-sample
        Akaike information criterion,
            AICc = n ln(RSS / n) + 2k + 2k (k + 1) / (n - k - 1),
        wins, with RSS and n taken over the points kept by BOTH fits (so the
        two are compared on the same data) and k = 3 (parabola) or 4
        (hyperbola). The extra term penalises the extra hyperbola parameter, so
        on a parabolic scan (equal RSS) the parabola wins, and the hyperbola
        only wins if it really fits better. A hyperbola fit that stopped at the
        evaluation limit (the degenerate case above) is still a valid
        candidate; it simply loses to the parabola.

        Multi-pass fit: pass 1 fits all finite points. In each further pass,
        points whose residual (from the previous fit, evaluated for all points)
        deviates from the median residual by more than `fit_outlier_sigma`
        robust sigmas (1.4826 * MAD) are rejected and the survivors are
        refitted. If a pass would leave fewer than `fit_min_points` points, the
        previous pass is kept instead of failing.

        Shape check
        -----------
        Only fits whose parameters describe a shape that cannot give a
        meaningful minimum are removed:
        1. no minimum: parabola opens downwards (A <= 0), or hyperbola with
           flat arms (a <= 0);
        2. the minimum lies outside the full angle search range
           [`angle_scan_start`, `angle_scan_end`];
        3. the minimum is extrapolated more than `fit_max_extrapolation` fine
           scan widths beyond the edges of the fine scan (the position then
           depends only on the assumed model shape, not on the data);
        4. the curve is flat within the noise: the rise of the model from its
           minimum to the higher fine scan edge is less than
           `fit_min_contrast` times the robust scatter of the residuals.
        A minimum somewhat outside the fine scan is allowed (the fine scan is
        centred on the rough ternary search minimum, which can be off).

        Error of the vertex
        -------------------
        The formal least-squares error (covariance of x_v, for the parabola
        propagated to x_v = -B / 2A) assumes independent points, but
        neighbouring fine scan angles compare almost the same pixels and the
        residuals are correlated (they show a smooth wave rather than random
        scatter), so the formal error is too small. The reported error comes
        from a moving-block residual bootstrap of the chosen model instead:

        1. residuals r = d - fit for ALL finite points (in angle order),
           including the ones rejected as outliers;
        2. `angle_bootstrap_samples` times: build a new scan d* = fit + r*,
           where r* is made of randomly chosen contiguous blocks of
           `angle_bootstrap_block` residuals (so their correlation is kept),
           and repeat the WHOLE fit on it, outlier rejection included
           (`_multipass_fit`), taking its vertex (draws without a minimum
           are dropped);
        3. error = half the 16-84 percentile width of the vertices.

        Resampling only the kept residuals and refitting in one pass would
        underestimate the error by about 2x (tested on synthetic scans): the
        aggressive rejection trims the residual distribution, and the rejection
        itself adds scatter to the vertex. With the full refit the error is
        about 0.75-0.8x the true scatter of the vertex over noise realisations
        (white and correlated synthetic noise), i.e. still slightly optimistic,
        as residual bootstraps are. The model choice is not repeated in the
        bootstrap.

        No new image comparisons are needed. The formal error is logged next to
        it for comparison.

        Returns
        -------
        dict or None
            coeffs (model parameters, in x = angle - x0; 3 for the parabola, 4
            for the hyperbola), model (name of the chosen model), x0,
            vertex (deg, the minimum), vertex_err (deg, bootstrap),
            vertex_err_formal (deg), angles, diffs (finite points only), keep
            (bool mask of points used in the final pass), mean_resid (mean
            |residual| of the kept points), aicc (dict model -> AICc of the
            models that passed the shape check, empty if only one was tried),
            or None if no model passed.
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

        if self.angle_fit_model == "auto":
            models = ("parabola", "hyperbola")
        else:
            models = (self.angle_fit_model,)

        # fit every model and keep the ones with a sensible shape
        candidates = {}
        rejected = {}
        for model in models:
            params, keep, cov = self._multipass_fit(x, d, model)

            if params is None:
                rejected[model] = "the least-squares fit failed"
                continue

            reason = self._angle_fit_shape_problem(params, x, d, keep, x0)
            if reason is not None:
                rejected[model] = f"{reason} (params {np.round(params, 6)})"
                continue

            candidates[model] = (params, keep, cov)

        for model, reason in rejected.items():
            if self.logger:
                # with "auto" one rejected model is no problem, the other is used
                level = (
                    self.logger.info
                    if len(candidates) > 0
                    else self.logger.warning
                )
                level(f"Fine scan {model} fit rejected: {reason}")

        if len(candidates) == 0:
            return None

        # choose the model: AICc over the points kept by all candidates
        aicc = {}
        if len(candidates) > 1:
            common = np.logical_and.reduce([c[1] for c in candidates.values()])
            n = int(common.sum())
            for model, (params, _, _) in candidates.items():
                k = len(params)
                rss = float(
                    np.sum((d[common] - self._angle_model_eval(params, x[common])) ** 2)
                )
                if n > k + 1 and rss > 0:
                    aicc[model] = (
                        n * np.log(rss / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1)
                    )
                else:
                    # noise-free or too few common points: prefer fewer parameters
                    aicc[model] = float(k)
            chosen = min(aicc, key=aicc.get)

            if self.logger:
                self.logger.info(
                    "Fine scan model choice (AICc over "
                    f"{n} common points, lower is better): "
                    + ", ".join(f"{m} {v:.2f}" for m, v in aicc.items())
                    + f" -> {chosen}"
                )
        else:
            chosen = next(iter(candidates))

        params, keep, cov = candidates[chosen]
        vertex = x0 + self._angle_model_vertex(params)

        if not (a.min() <= vertex <= a.max()) and self.logger:
            self.logger.info(
                f"Fine scan {chosen} minimum {vertex:.4f} lies "
                f"outside the fine scan [{a.min():.4f}, {a.max():.4f}] (extrapolated)"
            )

        mean_resid = float(
            np.mean(np.abs(d[keep] - self._angle_model_eval(params, x[keep])))
        )

        vertex_err, vertex_err_formal = self._vertex_error(x, d, params, keep, cov)

        if self.logger:
            self.logger.info(
                f"Fine scan {chosen} vertex error: {vertex_err:.5f} degrees (block "
                f"bootstrap), {vertex_err_formal:.5f} degrees (formal, "
                f"independent points)"
            )

        return dict(
            coeffs=params,
            model=chosen,
            x0=x0,
            vertex=float(vertex),
            vertex_err=float(vertex_err),
            vertex_err_formal=float(vertex_err_formal),
            angles=a,
            diffs=d,
            keep=keep,
            mean_resid=mean_resid,
            aicc=aicc,
        )

    def _angle_fit_shape_problem(self, params, x, d, keep, x0):
        """
        Shape check of a fine scan fit (see `_fit_angle_parabola`).

        Returns None if the shape gives a meaningful minimum, else a short
        description of the problem (for the log).
        """
        model = self._angle_model_name(params)
        vertex_x = self._angle_model_vertex(params)

        # 1. the model must have a minimum
        if vertex_x is None:
            if model == "parabola":
                return "no minimum, the parabola opens downwards"
            return "no minimum, the hyperbola arms are flat"

        # 2. inside the full angle search range
        vertex = x0 + vertex_x
        if not (self.angle_scan_start <= vertex <= self.angle_scan_end):
            return (
                f"minimum {vertex:.4f} lies outside the angle search range "
                f"[{self.angle_scan_start:.4f}, {self.angle_scan_end:.4f}]"
            )

        # 3. not extrapolated far beyond the fine scan
        width = float(x.max() - x.min())
        max_out = self.fit_max_extrapolation * width
        if vertex_x < x.min() - max_out or vertex_x > x.max() + max_out:
            return (
                f"minimum {vertex:.4f} is extrapolated more than "
                f"{self.fit_max_extrapolation:g} fine scan widths beyond the "
                f"fine scan [{x0 + x.min():.4f}, {x0 + x.max():.4f}]"
            )

        # 4. the dip must stand out from the noise
        residuals = d[keep] - self._angle_model_eval(params, x[keep])
        scatter = 1.4826 * np.median(np.abs(residuals - np.median(residuals)))
        if scatter <= 0:
            scatter = np.std(residuals)

        bottom = float(self._angle_model_eval(params, np.array([vertex_x]))[0])
        edges = self._angle_model_eval(params, np.array([x.min(), x.max()]))
        rise = float(np.max(edges)) - bottom

        if rise <= self.fit_min_contrast * scatter:
            return (
                f"the curve is flat within the noise (rise {rise:.3g} from the "
                f"minimum to the higher scan edge, residual scatter {scatter:.3g}, "
                f"need > {self.fit_min_contrast:g}x)"
            )

        return None

    # ------------------------------------------------------------------
    # Fine scan models
    #
    #   parabola : params = (A, B, C), f = A x^2 + B x + C, vertex -B / 2A
    #   hyperbola: params = (a, b, c, x_v), f = sqrt(a (x - x_v)^2 + b) + c,
    #              minimum exactly at x_v (a > 0, b >= 0)
    #
    # The model is identified by the number of parameters, so a fit result
    # carries its model with it. x is the angle minus the centre of the fine
    # scan (well conditioned).
    # ------------------------------------------------------------------

    @staticmethod
    def _angle_model_name(params):
        """Name of the fine scan model of `params` (3 = parabola, 4 = hyperbola)."""
        return "parabola" if len(params) == 3 else "hyperbola"

    def _angle_model_eval(self, params, x):
        """Evaluate the fine scan model `params` at x (= angle - x0)."""
        params = np.asarray(params, float)

        if self._angle_model_name(params) == "parabola":
            return np.polyval(params, x)

        a, b, c, x_v = params
        return np.sqrt(a * (x - x_v) ** 2 + b) + c

    def _angle_model_vertex(self, params):
        """Position (in x = angle - x0) of the minimum of the model, or None
        if the model has no minimum."""
        params = np.asarray(params, float)

        if not np.all(np.isfinite(params)):
            return None

        if self._angle_model_name(params) == "parabola":
            A, B, _ = params
            return None if A <= 0 else float(-B / (2 * A))

        # b = 0 is a sharp V, still with a well defined minimum at x_v
        a, b, _, x_v = params
        return None if (a <= 0 or b < 0) else float(x_v)

    def _angle_model_fit(self, x, d, model, p0=None):
        """
        Least-squares fit of the fine scan model `model` ("parabola" or
        "hyperbola") to (x, d).

        Parabola: linear (np.polyfit). Hyperbola: nonlinear
        (scipy.optimize.least_squares with a >= 0, b >= 0), started from `p0`
        if given, else from a parabola fit: its vertex for x_v, the slopes of
        the outer thirds of the scan for the arms (sqrt(a)), and its curvature
        A = a / (2 sqrt(b)) near the minimum for b.

        A hyperbola fit that stops at the evaluation limit is kept (if its
        parameters are finite): that is the degenerate case of a parabolic
        scan (see `_fit_angle_parabola`), where the parameters drift off
        while the shape and x_v are already settled. The model choice then
        prefers the parabola.

        Returns (params, covariance), covariance from the jacobian scaled by
        the residual variance (NaN if it cannot be computed), or (None, None)
        if the fit failed.
        """
        if model == "parabola":
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

        # status 0 = evaluation limit reached: kept (degenerate parabolic case),
        # status -1 = improper input: failed
        if result.status < 0 or not np.all(np.isfinite(result.x)):
            return None, None

        params = result.x

        dof = max(len(x) - len(params), 1)
        sigma2 = float(np.sum(result.fun**2) / dof)
        try:
            cov = sigma2 * np.linalg.inv(result.jac.T @ result.jac)
        except np.linalg.LinAlgError:
            cov = np.full((4, 4), np.nan)

        return params, cov

    def _multipass_fit(self, x, d, model, p0=None):
        """
        The multi-pass fit of the fine scan model `model` used by
        `_fit_angle_parabola` (and repeated by its bootstrap): pass 1 fits all
        points; every further pass rejects points whose residual deviates from
        the median residual by more than `fit_outlier_sigma` robust sigmas and
        refits the survivors. Residuals of ALL points are recomputed from each
        new fit, so a point rejected earlier can be let back in. If a pass
        would leave fewer than `fit_min_points` points, or its refit fails,
        the previous pass is kept (the rejection only refines a fit, it is
        not a reason to lose it).

        Returns (params, keep, covariance of the last kept pass), or
        (None, keep, None) if the first fit fails.
        """
        keep = np.ones(len(x), dtype=bool)
        params, cov = self._angle_model_fit(x, d, model, p0=p0)

        if params is None:
            return None, keep, None

        for _ in range(self.fit_passes - 1):
            r = d - self._angle_model_eval(params, x)
            med = np.median(r)
            sigma = 1.4826 * np.median(np.abs(r - med))
            if sigma <= 0:
                sigma = np.std(r)
            if sigma > 0:
                new_keep = np.abs(r - med) <= self.fit_outlier_sigma * sigma
            else:
                new_keep = np.ones(len(x), dtype=bool)

            if new_keep.sum() < self.fit_min_points:
                break

            new_params, new_cov = self._angle_model_fit(
                x[new_keep], d[new_keep], model, p0=params
            )

            if new_params is None:
                break

            params, cov, keep = new_params, new_cov, new_keep

        return params, keep, cov

    def _vertex_error(self, x, d, params, keep, cov):
        """
        Error of the minimum of the fine scan model `params` fitted to the
        points (x, d) (x = angle - x0, in angle order; `keep` = points used in
        the final pass, `cov` = covariance of that pass).

        Returns (bootstrap error, formal error) in degrees, see
        `_fit_angle_parabola`. NaN if it cannot be estimated.
        """
        model = self._angle_model_name(params)

        # formal error, assuming independent points
        vertex_err_formal = np.nan
        if cov is not None and np.all(np.isfinite(cov)):
            if model == "parabola":
                # propagate (A, B) to x_v = -B / (2A)
                A, B, _ = params
                J = np.array([B / (2 * A**2), -1.0 / (2 * A)])
                vertex_err_formal = float(np.sqrt(J @ cov[:2, :2] @ J))
            elif cov[3, 3] >= 0:
                # x_v is a fit parameter
                vertex_err_formal = float(np.sqrt(cov[3, 3]))

        # moving-block residual bootstrap of ALL points, repeating the full
        # fit (with outlier rejection) for every draw
        fitted = self._angle_model_eval(params, x)
        residuals = d - fitted

        n = len(x)
        block = int(np.clip(self.angle_bootstrap_block, 1, n))
        n_blocks = int(np.ceil(n / block))

        rng = np.random.default_rng(self.angle_bootstrap_seed)

        vertices = []
        for _ in range(int(self.angle_bootstrap_samples)):
            starts = rng.integers(0, n - block + 1, size=n_blocks)
            resampled = np.concatenate([residuals[k : k + block] for k in starts])[:n]

            p, _, _ = self._multipass_fit(x, fitted + resampled, model, p0=params)
            vertex = None if p is None else self._angle_model_vertex(p)
            if vertex is not None:
                vertices.append(vertex)

        if len(vertices) < 0.5 * self.angle_bootstrap_samples:
            if self.logger:
                self.logger.warning(
                    f"Only {len(vertices)}/{self.angle_bootstrap_samples} bootstrap "
                    f"{model} fits have a minimum, vertex error unknown"
                )
            return np.nan, vertex_err_formal

        p16, p84 = np.percentile(vertices, [16, 84])

        return float(0.5 * (p84 - p16)), vertex_err_formal

    # ------------------------------------------------------------------
    # Arrow plots (flow fields, point source displacements and residuals)
    # ------------------------------------------------------------------

    @staticmethod
    def _quiver_subsample(src, mask, max_per_axis=25):
        """
        Mask of a regular subset of the flow grid with at most `max_per_axis`
        vectors per axis (2500 arrows are unreadable), and the spacing (px) of
        that subset.
        """
        xs = np.unique(src[mask, 0])
        ys = np.unique(src[mask, 1])

        step = max(1, int(np.ceil(max(len(xs), len(ys)) / max_per_axis)))

        sub = mask & np.isin(src[:, 0], xs[::step]) & np.isin(src[:, 1], ys[::step])

        spacing = step * float(np.median(np.diff(xs))) if len(xs) > 1 else 1.0

        return sub, spacing

    @staticmethod
    def _arrow_magnification(vectors, spacing):
        """
        Factor that makes the longest arrows (95th percentile of the vector
        lengths) about 1.5 grid spacings long, and a round reference length
        (px) for the arrow key.
        """
        lengths = np.linalg.norm(vectors, axis=1)
        lengths = lengths[np.isfinite(lengths)]

        if len(lengths) == 0 or np.percentile(lengths, 95) <= 0:
            return 1.0, 1.0

        long = float(np.percentile(lengths, 95))
        reference = float(f"{long:.1g}")

        return 1.5 * spacing / long, reference

    def _draw_flow_quiver(
        self, ax, fig, src, vectors, sub, spacing, magnification, reference, label,
        width=0.003,
    ):
        """Arrows of `vectors` at `src[sub]`, `magnification` times longer,
        coloured by their (true) length, with a colour bar and an arrow key.

        The arrows are drawn over grey frames, so only the bright part of
        viridis is used (teal to yellow, the dark purple / blue end disappears
        on grey) and every arrow gets a thin black outline."""
        lengths = np.linalg.norm(vectors[sub], axis=1)

        q = ax.quiver(
            src[sub, 0],
            src[sub, 1],
            vectors[sub, 0] * magnification,
            vectors[sub, 1] * magnification,
            lengths,
            cmap=ARROW_CMAP,
            angles="xy",
            scale_units="xy",
            scale=1,
            width=width,
            **ARROW_OUTLINE,
        )
        ax.quiverkey(
            q,
            0.80,
            1.02,
            reference * magnification,
            f"{reference:g} px",
            labelpos="E",
            coordinates="axes",
            color="white",
            labelcolor="white",
        )
        # constant length (e.g. the model of a pure shift): one colour, not
        # rounding noise stretched over the whole colour map
        if len(lengths) and np.ptp(lengths) <= 1e-3 * max(np.max(lengths), 1e-12):
            q.set_clim(0.9 * np.mean(lengths), 1.1 * np.mean(lengths) + 1e-12)
        cbar = fig.colorbar(q, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(label)

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
                    label=f"{fit['model'].capitalize()} fit ({self.fit_passes}-pass)",
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
                f"{fit['model'].capitalize()} fit residuals (rms of used "
                f"points = {rms:.3g})",
                fontsize=10,
            )
            ax_fit_res.legend(fontsize=8)

            # y range from the used points, rejected points may be far off
            r_max = np.max(np.abs(residuals[keep])) if keep.any() else 1.0
            ax_fit_res.set_ylim(-1.5 * r_max, 1.5 * r_max)
        else:
            ax_fit_res.set_title(
                "Fine scan fit residuals (fit failed)",
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
        self._save_QA_figure(filename, fig)

        if self.logger:
            self.logger.info(f"Saved angle scan plot to {filename}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    def _scan_shift_room(self, expected_shift, source="expected"):
        """
        Room (pixels) for the translation in the comparison window of the
        angle scan: max(`scan_max_shift`, max(|tx|, |ty|) + `scan_shift_slack`).
        Returns None if `expected_shift` is None or not finite (the room is
        then set from the first translation measured in the scan).
        """
        room = int(self.scan_max_shift)

        if expected_shift is None:
            return None

        expected_shift = np.asarray(
            [np.nan if v is None else v for v in expected_shift], dtype=float
        )

        if not np.all(np.isfinite(expected_shift)):
            return None

        needed = int(np.ceil(np.max(np.abs(expected_shift)))) + int(self.scan_shift_slack)

        if needed > room:
            room = needed

            if self.logger:
                self.logger.info(
                    f"Angle scan: comparison window leaves room for translations "
                    f"up to {room} px ({source} translation ({expected_shift[0]:.2f}, "
                    f"{expected_shift[1]:.2f}) px + {self.scan_shift_slack} px)"
                )

        return room

    def _scan_for_best_angle(
        self,
        evaluate_angle,
        obs_name,
        save_QA=True,
        show_QA=False,
        expected_shift=None,
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

        expected_shift : tuple or None
            Translation (tx, ty) expected for the scan (e.g. the direct fit or
            the initial guess). Sets the room for the translation in the
            comparison window (`_scan_shift_room`). If None, the first
            translation measured in the scan is used.

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

        # room for the translation in the comparison window, the same for
        # every angle of this scan (None: set from the first translation)
        self._scan_room = self._scan_shift_room(expected_shift)

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
                expected_shift=(0.0, 0.0),
            )

            if scan_result is None:
                self.plot_failed_aligment(
                    reference_frame, input_frame, obs_name, show=show_QA
                )
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
        NUM_NEAREST_NEIGHBOURS=5,
        PIXEL_TOL=5,
        min_area=10,
        rematch=True,
        rematch_detection_sigma=3.0,
        rematch_min_area=5,
        match_radius=3.0,
        outlier_threshold=1.5,
        outlier_sigma=3.0,
        outlier_threshold_min=0.1,
        robust_f_scale=0.5,
        min_sources=5,
        max_iterations=5,
        QA_path=None,
    ):
        """
        Parameters (matching: see astroalign; outlier rejection below)
        ----------
        NUM_NEAREST_NEIGHBOURS : int
            astroalign NUM_NEAREST_NEIGHBORS (default 5, the astroalign
            default; the setting was previously not applied because of the
            spelling, so 5 is what was effectively used before).

        rematch : bool
            If True (default), astroalign is only used to find the transform:
            all sources are then detected again in both frames (no limit of
            50 control points, `rematch_detection_sigma`, `rematch_min_area`),
            the input sources are projected with the astroalign transform, and
            unambiguous mutual nearest neighbours within `match_radius` are
            matched. astroalign itself only returns the sources that are part
            of its matched triangles, which are usually much fewer than the
            sources available. The astroalign matches are kept if the
            re-matching finds fewer.

        rematch_detection_sigma : float
            Detection threshold (background sigmas) of the re-matching
            (default 3). Noise detections hardly ever coincide in both frames
            within `match_radius`, so this can be lower than for astroalign.

        rematch_min_area : int
            Minimum source area (pixels) of the re-matching (default 5).

        match_radius : float
            Matching radius (pixels) of the re-matching (default 3). Pairs
            with a second candidate within this radius are left out as
            ambiguous. Mismatches inside the radius are removed by the outlier
            rejection.

        outlier_threshold : float
            Upper limit (pixels) of the adaptive outlier threshold (default
            1.5). Only there to always reject clear mismatches, also when the
            scatter is large.

        outlier_sigma : float
            Outlier threshold on the residual length in robust sigmas of the
            residual scatter of the accepted sources (default 3), see
            `stats.robust_residual_threshold`.

        outlier_threshold_min : float
            Lower limit (pixels) of the adaptive threshold (default 0.1).

        robust_f_scale : float
            Scale (pixels) of the soft-L1 loss of the first fit (default 0.5):
            residuals well above it count linearly instead of quadratically, so
            mismatches (astroalign accepts matches within PIXEL_TOL = 5 px) do
            not pull the first fit, whose residuals set the first threshold.

        min_sources : int
            Minimum number of accepted (matched and not rejected) sources for
            the alignment to count as successful (default 5). Also the minimum
            number of astroalign matches before the looser matching settings
            are tried.

        Why these values
        ----------------
        On real MUSE data (white-light dither vs. reference) the centroid
        residuals of correctly matched sources scatter by sigma ~0.2-0.8 px
        per axis (sep barycentres, different PSFs, extended sources), while
        mismatches are off by 1.5-5 px. A fixed 0.5 px cut (the old default)
        rejected up to half of the good sources; the adaptive 3 sigma cut
        follows the actual scatter of each frame pair, and the 1.5 px upper
        limit still removes the mismatches.
        """

        self.MIN_MATCHES_FRACTION = MIN_MATCHES_FRACTION
        self.detection_sigma = detection_sigma
        self.NUM_NEAREST_NEIGHBOURS = NUM_NEAREST_NEIGHBOURS
        self.PIXEL_TOL = PIXEL_TOL
        self.min_area = min_area
        self.rematch = rematch
        self.rematch_detection_sigma = rematch_detection_sigma
        self.rematch_min_area = rematch_min_area
        self.match_radius = match_radius

        self.outlier_threshold = outlier_threshold
        self.outlier_sigma = outlier_sigma
        self.outlier_threshold_min = outlier_threshold_min
        self.robust_f_scale = robust_f_scale
        self.min_sources = int(min_sources)
        if self.min_sources < 3:
            raise ValueError("min_sources must be >= 3 (3 parameters are fitted)")
        self.max_iterations = max_iterations

        super().__init__("PointSources", "0.0.0", logger=logger, QA_path=QA_path)

        if self.logger:
            self.logger.info(
                f"Initialized {self.name} alignment method, version {self.version}."
            )

    def fit_euclidean_transform(
        self, source, target, centre, p0=[0.0, 0.0, 0.0], loss="linear", f_scale=1.0
    ):
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
        loss, f_scale : optional
            Loss of scipy.optimize.least_squares ("linear" = plain least
            squares; "soft_l1" for the robust first fit) and its scale in
            pixels. The errors are only meaningful for "linear".

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
            loss=loss,
            f_scale=f_scale,
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

    def fit_translation(self, source, target, p0=[0.0, 0.0], loss="linear", f_scale=1.0):
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

        loss, f_scale : optional
            As in `fit_euclidean_transform`.

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
            loss=loss,
            f_scale=f_scale,
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
        loss="linear",
        f_scale=1.0,
    ):
        """
        Fit a Euclidean transform, or a translation with a fixed rotation.

        If `skip_rotation_estimation` is True, the rotation is fixed at zero.
        If `fixed_rotation` is given, it is fixed at that value (radians).
        Otherwise the full rotation + translation model is fitted.

        Fixing the rotation is used by the angle scan: the source coordinates
        are rotated by the trial angle and only the translation is fitted.

        `loss` and `f_scale` are passed to the least-squares fit (see
        `fit_euclidean_transform`).
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
                rotated_source, target, p0=p0[1:3], loss=loss, f_scale=f_scale
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
            return self.fit_euclidean_transform(
                source, target, centre, p0=p0, loss=loss, f_scale=f_scale
            )

    def _outlier_threshold(self, residuals, n_parameters):
        """
        Adaptive outlier threshold (pixels) on the residual length of the
        accepted sources `residuals` ((N, 2)) of a fit with `n_parameters`
        parameters, see `stats.robust_residual_threshold`. Returns
        (threshold, sigma).
        """
        return robust_residual_threshold(
            residuals,
            n_sigma=self.outlier_sigma,
            minimum=self.outlier_threshold_min,
            maximum=self.outlier_threshold,
            min_points=self.min_sources,
            n_parameters=n_parameters,
        )

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
        translation-only fit. Outliers are rejected iteratively in the same way
        as in the normal PointSources fit: robust (soft-L1) first fit, then
        least squares with the adaptive threshold (residuals of all sources
        recomputed every pass, so rejected sources can come back). Fewer than
        `min_sources` accepted sources is a failure.

        Returns a small result dict compatible with the shared scan metric, or
        None if the fit failed.
        """
        theta = -np.radians(angle)

        good_mask = np.ones(len(source), dtype=bool)
        offset = np.asarray(p0, dtype=float)
        offset_err = np.full(2, np.nan)

        rotated_all = self._rotate_points(source, centre, theta)

        for iteration in range(self.max_iterations):
            if good_mask.sum() < self.min_sources:
                return None

            # robust first fit, so mismatches do not set the first threshold
            try:
                offset, offset_err, _ = self.fit_translation(
                    rotated_all[good_mask],
                    target[good_mask],
                    p0=offset,
                    loss="soft_l1" if iteration == 0 else "linear",
                    f_scale=self.robust_f_scale,
                )
            except Exception:
                return None

            residuals = target - (rotated_all + offset)

            threshold, _ = self._outlier_threshold(
                residuals[good_mask], n_parameters=2
            )

            new_good_mask = np.linalg.norm(residuals, axis=1) <= threshold

            if np.array_equal(new_good_mask, good_mask):
                break

            good_mask = new_good_mask

        source_fit = source[good_mask]
        target_fit = target[good_mask]

        if len(source_fit) < self.min_sources:
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

    # ------------------------------------------------------------------
    # QA: matched sources and residuals
    # ------------------------------------------------------------------

    def plot_point_sources(
        self,
        reference_frame,
        input_frame,
        source_list,
        target_list,
        obs_name,
        good_mask=None,
        residuals=None,
        threshold=None,
        sigma=None,
        result=None,
        errors=None,
        scan_rotation=None,
        failed_reason=None,
        save=True,
        show=False,
    ):
        """
        QA plot of a point-source alignment, made for every alignment where
        astroalign found matches (also failed ones).

        Panels
        ------
        Top row:
        1. Reference frame with the matched sources (accepted: cyan circles,
           rejected: red crosses), numbered as in the log ("Rejected original
           source indices").
        2. Input frame with the same sources and numbers, to check by eye that
           the pairs are the same objects.
        3. Measured displacement of every pair (reference - input position)
           over the reference, magnified and coloured by length. For a rigid
           offset all arrows are parallel (shift) or turn about one point
           (rotation); a mismatch sticks out.

        Bottom row (only with a fit):
        4. Residuals (reference position - fitted model) at the source
           positions, magnified on their own scale. Random arrows are noise; a
           coherent pattern (e.g. one side pulling one way) is something the
           rigid model does not describe. Rejected sources are drawn in red with
           their length clipped (direction only).
        5. Residual vectors in the (dx, dy) plane with the adaptive threshold
           and the 1 sigma circle; rejected sources beyond the plot range are
           drawn on its edge.
        6. Histogram of the residual lengths with the expected (Rayleigh)
           distribution for the estimated sigma and the threshold. Accepted
           sources should follow the curve; a long tail inside the threshold
           means the scatter is not gaussian (e.g. a few poor centroids).

        Saved to `point_sources_{obs_name}_{name}.png`.
        """
        source_list = np.asarray(source_list, float)
        target_list = np.asarray(target_list, float)
        n = len(source_list)

        if good_mask is None:
            good_mask = np.ones(n, dtype=bool)
            fitted = False
        else:
            good_mask = np.asarray(good_mask, bool)
            fitted = residuals is not None

        bad_mask = ~good_mask

        fig, axes = plt.subplots(2, 3, figsize=(22, 14))
        ax_ref, ax_in, ax_disp = axes[0]
        ax_res, ax_scatter, ax_hist = axes[1]

        def _show(ax, frame, title, alpha=1.0):
            data = np.ma.filled(np.ma.asarray(frame, dtype=float), np.nan)
            finite = data[np.isfinite(data)]
            vmin, vmax = (
                np.percentile(finite, [5, 99]) if len(finite) else (0.0, 1.0)
            )
            ax.imshow(
                data,
                cmap="gray",
                origin="upper",
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
                alpha=alpha,
            )
            ax.set_title(title)
            ax.set_xlabel("x [px]")
            ax.set_ylabel("y [px]")

        def _mark(ax, positions):
            ax.scatter(
                positions[good_mask, 0],
                positions[good_mask, 1],
                s=90,
                facecolors="none",
                edgecolors="cyan",
                linewidths=1.2,
                label=f"accepted ({int(good_mask.sum())})" if fitted else f"matched ({n})",
            )
            if fitted and bad_mask.any():
                ax.scatter(
                    positions[bad_mask, 0],
                    positions[bad_mask, 1],
                    s=90,
                    marker="x",
                    color="red",
                    linewidths=1.5,
                    label=f"rejected ({int(bad_mask.sum())})",
                )
            for k, (x, y) in enumerate(positions):
                ax.annotate(
                    str(k),
                    (x, y),
                    xytext=(6, 6),
                    textcoords="offset points",
                    fontsize=9,
                    color="red" if (fitted and bad_mask[k]) else "yellow",
                    bbox=dict(boxstyle="round,pad=0.1", fc="black", ec="none", alpha=0.6),
                )
            ax.legend(loc="lower right", fontsize=8)

        # ---- 1, 2) frames with the matched sources ------------------------
        _show(ax_ref, reference_frame, "Reference frame, matched sources")
        _mark(ax_ref, target_list)

        _show(ax_in, input_frame, "Input frame, matched sources")
        _mark(ax_in, source_list)

        # few, sparse arrows: the longest ones about 12% of the frame size,
        # drawn thicker than the dense flow fields, with a dot at their base
        frame_size = max(np.shape(reference_frame)[:2])
        spacing = 0.08 * frame_size

        # ---- 3) measured displacement -------------------------------------
        displacement = target_list - source_list

        _show(ax_disp, reference_frame, "", alpha=0.6)
        magnification, reference = self._arrow_magnification(
            displacement[good_mask], spacing
        )
        self._draw_flow_quiver(
            ax_disp, fig, target_list, displacement, good_mask, spacing,
            magnification, reference, "|displacement| [px]", width=0.006,
        )
        ax_disp.scatter(target_list[:, 0], target_list[:, 1], s=8, color="white")
        if fitted and bad_mask.any():
            ax_disp.quiver(
                target_list[bad_mask, 0],
                target_list[bad_mask, 1],
                displacement[bad_mask, 0] * magnification,
                displacement[bad_mask, 1] * magnification,
                color="red",
                angles="xy",
                scale_units="xy",
                scale=1,
                width=0.006,
                **ARROW_OUTLINE,
            )
        ax_disp.set_title(
            f"Measured displacement (reference - input position), arrows "
            f"x{magnification:.3g}\n(rejected in red)",
            pad=22,
        )

        if not fitted:
            for ax in (ax_res, ax_scatter, ax_hist):
                ax.axis("off")
                ax.text(
                    0.5, 0.5, "no fit", ha="center", va="center", fontsize=16,
                    transform=ax.transAxes,
                )
        else:
            lengths = np.linalg.norm(residuals, axis=1)
            good_lengths = lengths[good_mask]
            rms = float(np.sqrt(np.mean(good_lengths**2))) if good_mask.any() else np.nan

            # ---- 4) residual arrows ----------------------------------------
            _show(ax_res, reference_frame, "", alpha=0.6)
            res_mag, res_ref = self._arrow_magnification(residuals[good_mask], spacing)
            self._draw_flow_quiver(
                ax_res, fig, target_list, residuals, good_mask, spacing,
                res_mag, res_ref, "|residual| [px]", width=0.006,
            )
            ax_res.scatter(target_list[:, 0], target_list[:, 1], s=8, color="white")
            if bad_mask.any():
                # direction only: clip the drawn length to 2x the longest arrow
                clip = 2 * 1.5 * spacing
                drawn = residuals[bad_mask] * res_mag
                drawn_len = np.linalg.norm(drawn, axis=1)
                scale = np.where(drawn_len > clip, clip / np.maximum(drawn_len, 1e-12), 1.0)
                drawn = drawn * scale[:, None]
                ax_res.quiver(
                    target_list[bad_mask, 0],
                    target_list[bad_mask, 1],
                    drawn[:, 0],
                    drawn[:, 1],
                    color="red",
                    angles="xy",
                    scale_units="xy",
                    scale=1,
                    width=0.006,
                    **ARROW_OUTLINE,
                )
            ax_res.set_title(
                f"Residuals (reference - model) of the accepted sources, arrows "
                f"x{res_mag:.3g}\nrms {rms:.3f} px (rejected in red, length clipped)",
                pad=22,
            )

            # ---- 5) residual vectors ---------------------------------------
            limit = max(
                1.3 * (threshold if threshold else 0.0),
                1.1 * (np.max(good_lengths) if good_mask.any() else 1.0),
            )

            ax_scatter.scatter(
                residuals[good_mask, 0],
                residuals[good_mask, 1],
                s=30,
                color="cyan",
                label=f"accepted ({int(good_mask.sum())})",
            )
            if bad_mask.any():
                on_scale = bad_mask & (lengths <= 0.97 * limit)
                off_scale = bad_mask & (lengths > 0.97 * limit)
                if on_scale.any():
                    ax_scatter.scatter(
                        residuals[on_scale, 0],
                        residuals[on_scale, 1],
                        s=50,
                        marker="x",
                        color="red",
                        label=f"rejected ({int(on_scale.sum())})",
                    )
                if off_scale.any():
                    edge = residuals[off_scale] * (
                        0.97 * limit / lengths[off_scale]
                    )[:, None]
                    ax_scatter.scatter(
                        edge[:, 0],
                        edge[:, 1],
                        s=50,
                        marker="D",
                        facecolors="none",
                        edgecolors="red",
                        label=(
                            f"rejected, off scale ({int(off_scale.sum())}, "
                            f"up to {lengths[off_scale].max():.2f} px)"
                        ),
                    )
                for k in np.flatnonzero(bad_mask):
                    xy = residuals[k] * min(1.0, 0.97 * limit / max(lengths[k], 1e-12))
                    ax_scatter.annotate(
                        str(k), xy, xytext=(5, 5), textcoords="offset points",
                        fontsize=7, color="red",
                    )

            if threshold is not None:
                ax_scatter.add_patch(
                    plt.Circle(
                        (0, 0), threshold, fill=False, color="yellow",
                        linestyle="--",
                        label=f"threshold {threshold:.3f} px",
                    )
                )
            if sigma is not None and np.isfinite(sigma):
                ax_scatter.add_patch(
                    plt.Circle(
                        (0, 0), sigma, fill=False, color="white", linestyle=":",
                        label=f"1 sigma ({sigma:.3f} px per axis)",
                    )
                )
            ax_scatter.axhline(0, color="gray", linewidth=0.8)
            ax_scatter.axvline(0, color="gray", linewidth=0.8)
            ax_scatter.set_xlim(-limit, limit)
            ax_scatter.set_ylim(-limit, limit)
            ax_scatter.set_aspect("equal")
            ax_scatter.set_xlabel("residual x [px]")
            ax_scatter.set_ylabel("residual y [px]")
            ax_scatter.set_title("Residual vectors")
            ax_scatter.legend(loc="upper right", fontsize=8)

            # ---- 6) residual length histogram ------------------------------
            hist_max = max(
                1.5 * (threshold if threshold else 0.0),
                1.1 * (np.max(good_lengths) if good_mask.any() else 1.0),
            )
            bins = np.linspace(0, hist_max, max(8, min(25, n // 2)))
            clipped = np.minimum(lengths, bins[-1] - 1e-9)

            ax_hist.hist(
                [clipped[good_mask], clipped[bad_mask]],
                bins=bins,
                stacked=True,
                color=["cyan", "red"],
                label=["accepted", "rejected (last bin: off scale)"],
            )
            if sigma is not None and np.isfinite(sigma) and sigma > 0:
                r = np.linspace(0, hist_max, 200)
                rayleigh = r / sigma**2 * np.exp(-(r**2) / (2 * sigma**2))
                ax_hist.plot(
                    r,
                    rayleigh * good_mask.sum() * (bins[1] - bins[0]),
                    color="white",
                    label=f"expected (Rayleigh, sigma {sigma:.3f} px)",
                )
            if threshold is not None:
                ax_hist.axvline(
                    threshold, color="yellow", linestyle="--",
                    label=f"threshold {threshold:.3f} px",
                )
            ax_hist.set_xlabel("|residual| [px]")
            ax_hist.set_ylabel("sources")
            ax_hist.set_title("Residual lengths")
            ax_hist.legend(fontsize=8)

        # ---- title ----------------------------------------------------------
        title = f"Point source alignment for {obs_name}"
        match_info = getattr(self, "_last_match_description", None)
        title += f"\n{n} matched sources" + (
            f" (astroalign, {match_info})" if match_info else ""
        )
        if fitted:
            title += (
                f", {int(good_mask.sum())} accepted, {int(bad_mask.sum())} rejected"
                + (
                    f" (threshold {threshold:.3f} px = {self.outlier_sigma:g} sigma, "
                    f"limits [{self.outlier_threshold_min:g}, {self.outlier_threshold:g}] px)"
                    if threshold is not None
                    else ""
                )
            )

        if failed_reason is not None:
            title += f"\nFAILED: {failed_reason}"
        elif result is not None:
            tx, ty, rot = result
            tx_err, ty_err, rot_err = errors if errors is not None else (np.nan,) * 3
            title += (
                f"\nCorrection of the input: tx = {tx:.3f} +/- {tx_err:.3f} px, "
                f"ty = {ty:.3f} +/- {ty_err:.3f} px, "
                f"rot = {rot:.4f} +/- {rot_err:.4f} deg"
            )

        if scan_rotation is not None:
            title += (
                f"\n(rotation {scan_rotation:.4f} deg from the angle scan, "
                f"translation fitted at that rotation)"
            )

        fig.suptitle(title, fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.95))

        if save:
            filename = f"{self.QA_path}/point_sources_{obs_name}_{self.name}.png"
            self._save_QA_figure(filename, fig)
            if self.logger:
                self.logger.info(f"Saved point source QA plot to {filename}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    def _detect_sources(self, frame, detection_sigma, min_area):
        """
        All sep detections (x, y) in a (masked) frame, brightest first. Same
        detection as astroalign (global background rms, barycentres), but
        without the limit on the number of sources.
        """
        data = np.ascontiguousarray(np.ma.getdata(frame), dtype=np.float32)
        mask = np.ascontiguousarray(np.ma.getmaskarray(frame))

        background = sep.Background(data, mask=mask)
        sources = sep.extract(
            data - background.back(),
            detection_sigma * background.globalrms,
            minarea=min_area,
            mask=mask,
        )
        sources.sort(order="flux")
        sources = sources[::-1]

        return np.column_stack([sources["x"], sources["y"]]).astype(float)

    def _rematch_sources(self, transform, input_frame, reference_frame):
        """
        Match all sources detected in both frames using the astroalign
        transform (input -> reference).

        The input sources are projected into the reference frame, and a pair
        is kept if the two sources are mutual nearest neighbours within
        `match_radius` and no other source of either frame lies within
        `match_radius` (ambiguous pairs are left out).

        Returns (source_list, target_list) (input / reference positions), or
        None if the detection fails.
        """
        try:
            input_sources = self._detect_sources(
                input_frame, self.rematch_detection_sigma, self.rematch_min_area
            )
            reference_sources = self._detect_sources(
                reference_frame, self.rematch_detection_sigma, self.rematch_min_area
            )
        except Exception as e:
            if self.logger:
                self.logger.warning(
                    f"Source detection for the re-matching failed: {e}, "
                    f"using the astroalign matches."
                )
            return None

        if len(input_sources) < 2 or len(reference_sources) < 2:
            return None

        projected = aa.matrix_transform(input_sources, transform.params)

        reference_tree = cKDTree(reference_sources)
        projected_tree = cKDTree(projected)

        # two nearest neighbours each way: the second one tells if a pair is ambiguous
        dist_forward, idx_forward = reference_tree.query(projected, k=2)
        dist_backward, idx_backward = projected_tree.query(reference_sources, k=2)

        n_input = len(projected)
        input_idx = np.arange(n_input)
        reference_idx = idx_forward[:, 0]

        keep = (
            (dist_forward[:, 0] <= self.match_radius)
            & (dist_forward[:, 1] > self.match_radius)
            & (idx_backward[reference_idx, 0] == input_idx)
            & (dist_backward[reference_idx, 1] > self.match_radius)
        )

        if self.logger:
            self.logger.info(
                f"Re-matching: {len(input_sources)} input and "
                f"{len(reference_sources)} reference sources detected "
                f"(detection_sigma={self.rematch_detection_sigma}, "
                f"min_area={self.rematch_min_area}), {int(keep.sum())} matched "
                f"within {self.match_radius} px."
            )

        return input_sources[keep], reference_sources[reference_idx[keep]]

    def _find_matches(self, input_frame, reference_frame):
        """
        Match point sources between the frames with astroalign.

        First with the configured settings; if astroalign fails, or finds
        fewer than `min_sources` matches, the settings are loosened one at a
        time, in this order (only values looser than the configured ones):

        1. MIN_MATCHES_FRACTION: 0.4, 0.3, 0.2
        2. detection_sigma: 4, 3, 2, 1
        3. NUM_NEAREST_NEIGHBOURS: 2 ... 8
        4. PIXEL_TOL: 6, 7
        5. min_area: 2 ... 20

        With `rematch` (default), the matches of each attempt are replaced by
        the re-matching of all detected sources with the astroalign transform
        if that gives more (see `_rematch_sources`).

        The first attempt with at least `min_sources` matches is used. The
        astroalign module settings and its random generator (seeded, as
        astroalign is otherwise not deterministic) are always restored.

        Returns (source_list, target_list) (input / reference positions), or
        None if no attempt gave enough matches.
        """
        attempts = [("configured settings", {})]
        attempts += [
            (f"MIN_MATCHES_FRACTION={v}", dict(MIN_MATCHES_FRACTION=v))
            for v in (0.4, 0.3, 0.2)
            if v < self.MIN_MATCHES_FRACTION
        ]
        attempts += [
            (f"detection_sigma={v}", dict(detection_sigma=v))
            for v in (4, 3, 2, 1)
            if v < self.detection_sigma
        ]
        attempts += [
            (f"NUM_NEAREST_NEIGHBOURS={v}", dict(NUM_NEAREST_NEIGHBOURS=v))
            for v in (2, 3, 4, 5, 6, 7, 8)
            if v != self.NUM_NEAREST_NEIGHBOURS
        ]
        attempts += [
            (f"PIXEL_TOL={v}", dict(PIXEL_TOL=v)) for v in (6, 7) if v > self.PIXEL_TOL
        ]
        attempts += [
            (f"min_area={v}", dict(min_area=v))
            for v in (2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20)
            if v != self.min_area
        ]

        self._last_match_description = None

        original_rng = aa._np.random.default_rng
        aa._np.random.default_rng = self._seeded_default_rng
        original_nearest_neighbours = aa.NUM_NEAREST_NEIGHBORS

        try:
            for description, overrides in attempts:

                aa.MIN_MATCHES_FRACTION = overrides.get(
                    "MIN_MATCHES_FRACTION", self.MIN_MATCHES_FRACTION
                )
                aa.NUM_NEAREST_NEIGHBORS = overrides.get(
                    "NUM_NEAREST_NEIGHBOURS", self.NUM_NEAREST_NEIGHBOURS
                )
                aa.PIXEL_TOL = overrides.get("PIXEL_TOL", self.PIXEL_TOL)

                try:
                    transform, (source_list, target_list) = aa.find_transform(
                        input_frame,
                        reference_frame,
                        detection_sigma=overrides.get(
                            "detection_sigma", self.detection_sigma
                        ),
                        min_area=overrides.get("min_area", self.min_area),
                    )
                except Exception as e:
                    if self.logger:
                        self.logger.warning(
                            f"astroalign failed with {description}: {e}"
                        )
                    continue

                source_list = np.asarray(source_list, float)
                target_list = np.asarray(target_list, float)

                if self.logger:
                    self.logger.info(
                        f"astroalign found {len(source_list)} matches with {description}"
                    )

                # astroalign only returns the sources of its matched triangles:
                # re-match all detected sources with its transform
                if self.rematch:
                    rematched = self._rematch_sources(
                        transform, input_frame, reference_frame
                    )
                    if rematched is not None and len(rematched[0]) > len(source_list):
                        source_list, target_list = rematched
                        description = f"{description}, re-matched"

                if len(source_list) < self.min_sources:
                    if self.logger:
                        self.logger.warning(
                            f"only {len(source_list)} matches with "
                            f"{description}, need {self.min_sources}"
                        )
                    continue

                # for the QA plot
                self._last_match_description = description

                return source_list, target_list

        finally:
            aa._np.random.default_rng = original_rng
            aa.MIN_MATCHES_FRACTION = self.MIN_MATCHES_FRACTION
            aa.NUM_NEAREST_NEIGHBORS = original_nearest_neighbours
            aa.PIXEL_TOL = self.PIXEL_TOL

        return None


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

        After each fit, the residuals of ALL matched sources are recomputed and
        the sources whose residual length exceeds an adaptive threshold are
        left out of the next fit: `outlier_sigma` robust sigmas of the residual
        scatter of the accepted sources, clipped to [`outlier_threshold_min`,
        `outlier_threshold`] (see `stats.robust_residual_threshold`). A source
        rejected earlier can come back. Iteration continues until the set of
        accepted sources no longer changes or max_iterations is reached. The
        first fit uses a robust soft-L1 loss (`robust_f_scale`), so mismatches
        do not set the first threshold; the following fits, and the reported
        errors, are plain least squares on the accepted sources.

        The alignment fails if fewer than `min_sources` (default 5) sources
        are accepted in the end (or matched by astroalign at all).

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
            self.plot_failed_aligment(
                reference_frame, input_frame, obs_name, show=show_QA
            )
            return self.FAILED_RETURN

        # ------------------------------------------------------------------
        # Find matching sources using astroalign
        # ------------------------------------------------------------------\

        matches = self._find_matches(input_frame, reference_frame)

        if matches is None:
            if self.logger:
                self.logger.error(
                    f"astroalign failed to find at least {self.min_sources} "
                    f"matching sources."
                )

            self.plot_failed_aligment(
                reference_frame,
                input_frame,
                obs_name,
                show=show_QA,
            )

            return self.FAILED_RETURN

        source_list, target_list = matches

        if self.logger:
            self.logger.info(
                f"Found {len(source_list)} matching sources for alignment."
            )
            self.logger.info(f"Source coordinates:\n{source_list}")
            self.logger.info(f"Target coordinates:\n{target_list}")

        # ------------------------------------------------------------------
        # QA plot of the matched sources (successful alignments only; a failed
        # alignment only gets the failed difference image)
        # ------------------------------------------------------------------

        def _sources_QA(**plot_kwargs):
            if save_QA or show_QA:
                self.plot_point_sources(
                    reference_frame,
                    input_frame,
                    source_list,
                    target_list,
                    obs_name,
                    save=save_QA,
                    show=show_QA,
                    **plot_kwargs,
                )

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
                expected_shift=initial_guess_translation,
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

        # Adaptive residual threshold, kept for the final QA plot.
        rejection_threshold = self.outlier_threshold
        rejection_sigma = np.nan

        converged = False

        for iteration in range(max_iterations):

            source_fit = source_list[good_mask]
            target_fit = target_list[good_mask]

            # Make sure we still have enough sources.
            if len(source_fit) < self.min_sources:
                if self.logger:
                    self.logger.error(
                        f"Only {len(source_fit)} matched sources remain after "
                        f"outlier rejection, need {self.min_sources}."
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

            # The first fit is robust (soft-L1 loss): mismatches of up to
            # PIXEL_TOL (5 px) would otherwise pull it, and its residuals set
            # the first threshold. Later fits are plain least squares on the
            # accepted sources (their errors are reported).
            try:
                (theta, offset, theta_err, offset_err, _) = self.fit_points(
                    source_fit,
                    target_fit,
                    centre,
                    p0=p0,
                    skip_rotation_estimation=skip_rotation_estimation,
                    fixed_rotation=fixed_rotation,
                    loss="soft_l1" if iteration == 0 else "linear",
                    f_scale=self.robust_f_scale,
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
            # Predicted target positions and residuals of ALL sources, so a
            # source rejected earlier can come back
            # --------------------------------------------------------------

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

            residuals_all = target_list - predicted_all

            # --------------------------------------------------------------
            # Find outliers
            #
            # A source is rejected if the LENGTH of its residual exceeds the
            # adaptive threshold: outlier_sigma robust sigmas of the residual
            # scatter of the currently accepted sources, clipped to
            # [outlier_threshold_min, outlier_threshold].
            # --------------------------------------------------------------

            n_parameters = (
                2 if (skip_rotation_estimation or fixed_rotation is not None) else 3
            )
            rejection_threshold, rejection_sigma = self._outlier_threshold(
                residuals_all[good_mask], n_parameters=n_parameters
            )

            new_good_mask = (
                np.linalg.norm(residuals_all, axis=1) <= rejection_threshold
            )

            rejected_indices = np.flatnonzero(good_mask & ~new_good_mask)
            readmitted_indices = np.flatnonzero(~good_mask & new_good_mask)

            if self.logger:
                self.logger.info(
                    f"Fit iteration {iteration + 1}: "
                    f"{len(rejected_indices)} outliers found, "
                    f"{len(readmitted_indices)} sources re-admitted "
                    f"(threshold = {rejection_threshold:.3f} pixels, "
                    f"{self.outlier_sigma:g} x sigma {rejection_sigma:.3f}, "
                    f"clipped to [{self.outlier_threshold_min:g}, "
                    f"{self.outlier_threshold:g}])."
                )

            # --------------------------------------------------------------
            # Accepted set unchanged -> convergence
            # --------------------------------------------------------------

            if np.array_equal(new_good_mask, good_mask):
                converged = True

                if self.logger:
                    self.logger.info(
                        "Accepted sources unchanged. Iterative fitting converged."
                    )

                break

            good_mask = new_good_mask

            if self.logger and len(rejected_indices) > 0:
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

        if len(source_fit) < self.min_sources:
            if self.logger:
                self.logger.error(
                    f"Only {len(source_fit)} accepted sources remain for the final "
                    f"alignment, need {self.min_sources}. Setting alignment as failed."
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
        # QA plots
        # ------------------------------------------------------------------

        _sources_QA(
            good_mask=good_mask,
            residuals=residuals,
            threshold=rejection_threshold,
            sigma=rejection_sigma,
            result=(tx, ty, rot),
            errors=(tx_err, ty_err, rot_err),
            scan_rotation=scanned_rotation,
        )

        if save_QA:

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
        scan_on_rotation_failure=True,
        fallback_to_direct_fit=True,
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
            If a scan with narrowed sigmas fails, it is redone once with the
            full sigma grid.

        scan_on_rotation_failure : bool
            With the angle scan: if the direct fit fails in its rotation
            estimate (no valid entry in the per-sigma rotation table), a
            translation-only fit at 0 degrees is tried, and if it succeeds the
            angle scan is run anyway, with that translation as the expected
            one (the scan does not need the direct rotation estimate).
            If False, the alignment fails as soon as the direct fit fails, as
            before (default True).

        fallback_to_direct_fit : bool
            With the angle scan: if the scan fails (e.g. the ternary search
            ends at the boundary of the angle range, or no angle could be
            scored) after a successful direct fit, return the direct fit
            instead of failing the alignment. The round-trip and dither checks
            of the pipeline still apply to it (default True).
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
        self.scan_on_rotation_failure = scan_on_rotation_failure
        self.fallback_to_direct_fit = fallback_to_direct_fit

        # result of the last final translation estimate of run(), used to
        # narrow the sigma range of the scan after the direct fit
        self._last_translation_result = None

        # stage at which the last run() failed ("rotation" or "translation"),
        # None after a success. Lets the scan path tell a failed direct
        # rotation estimate from a failed translation
        self._last_failure_stage = None

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

        self._save_QA_figure(
            f"{self.QA_path}/translation_progression_{obs_name}_{self.name}.png", fig
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

        self._save_QA_figure(
            f"{self.QA_path}/translation_peak_{obs_name}_{self.name}.png", fig
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
        rejections      : list of (sigma, [failed criteria]) for every processed
                          sigma (empty list: accepted). If no translation is found
                          and `obs_name` is given, a summary is logged
                          (`_log_translation_rejections`).
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

        # per sigma: the acceptance criteria it failed (empty if accepted), for
        # the log when no translation is found
        rejections = []

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
            criteria = (
                ("main peak outside the guess region", in_region),
                ("main peak refinement failed", ok_m_all),
                ("control peak refinement failed", bool(ok_c)),
                ("control offset not recovered", bool(np.all(np.abs(tc - c_vec) < ctrl_tol))),
                (f"snr_main <= {snr_min:g}", bool(snr_m > snr_min)),
                (f"snr_ctrl <= {snr_min:g}", bool(snr_c > snr_min)),
                (f"window not settled (> {window_tol:g} px)", bool(win_delta <= window_tol)),
                (f"snr_main / snr_ctrl < {snr_ratio_min:g}", bool(snr_ratio >= snr_ratio_min)),
                (f"peak shape deviation > {shape_tol:g}", bool(shape_dev <= shape_tol)),
            )
            failed_criteria = [name for name, passed in criteria if not passed]
            rejections.append((float(s), failed_criteria))
            ok = len(failed_criteria) == 0
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

            # why no translation was found. Only logged with an obs_name (the
            # direct and final estimates), the many estimates of the angle
            # scan would flood the log
            if self.logger and obs_name is not None:
                self._log_translation_rejections(
                    rejections, table, n_agree, agree_tol, obs_name
                )

        result["rejections"] = rejections

        if save_QA and obs_name is not None:
            self.plot_translation_peak(result, obs_name, show=show_QA)

        return result

    def _log_translation_rejections(self, rejections, table, n_agree, agree_tol, obs_name):
        """
        Log why `estimate_translation` found no translation: how many sigmas
        failed each acceptance criterion, and the criteria failed per sigma.
        If enough sigmas were accepted, the failure is the agreement instead
        (no `n_agree` consecutive accepted sigmas within `agree_tol` px).
        """
        n_sigmas = len(rejections)
        n_accepted = sum(1 for _, failed in rejections if len(failed) == 0)

        counts = {}
        for _, failed in rejections:
            for name in failed:
                counts[name] = counts.get(name, 0) + 1

        summary = ", ".join(
            f"{name}: {count}"
            for name, count in sorted(counts.items(), key=lambda item: -item[1])
        )

        lines = []
        for k, (sigma, failed) in enumerate(rejections):
            eps = table[k, 5:7] if k < len(table) else (np.nan, np.nan)
            lines.append(
                f"  sigma {sigma:.3f}: eps = ({eps[0]:.3f}, {eps[1]:.3f}) px, "
                + ("accepted" if len(failed) == 0 else "rejected - " + "; ".join(failed))
            )

        if n_accepted >= n_agree:
            reason = (
                f"{n_accepted} sigmas accepted, but no {n_agree} consecutive ones "
                f"agree within {agree_tol:g} px"
            )
        else:
            reason = f"only {n_accepted} sigmas accepted, need {n_agree} consecutive"

        self.logger.warning(
            f"No translation found for {obs_name}: {reason} ({n_sigmas} sigmas "
            f"processed). Rejections per criterion: {summary or 'none'}\n"
            + "\n".join(lines)
        )

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
        if it fails in the translation, so does the alignment. If only its
        rotation estimate fails, a translation-only fit at 0 degrees is tried
        and, if it works, the scan is run anyway (`scan_on_rotation_failure`).
        If the direct fit agrees with the initial guess, the scan only searches
        the sigmas around the ones the direct fit agreed on
        (`rotation_guess_tol`, `scan_sigma_margin`); if that narrowed scan
        fails, it is redone with the full sigma grid. If the scan fails after a
        successful direct fit, the direct fit is returned
        (`fallback_to_direct_fit`).

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

            # translation expected for the scan (room in its comparison
            # window): the direct fit, else the initial guess
            expected_shift = initial_guess_translation

            # direct fit result, returned if the scan fails
            # (`fallback_to_direct_fit`); None if there is none to fall back on
            direct_result = None

            # cheap check first: the direct fit. If it fails in the translation,
            # the slow angle scan is not run and the alignment is concluded as
            # failed (the direct fit already wrote the failed alignment QA
            # plot). If only its rotation estimate failed, the scan can still
            # find the rotation (`scan_on_rotation_failure`).
            if self.direct_fit_before_scan:

                if self.logger:
                    self.logger.info(
                        f"Direct FFT fit for {obs_name} before the angle scan ..."
                    )

                self._last_failure_stage = None

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

                if direct_result[-1]:

                    if self.logger:
                        self.logger.info(
                            f"Direct FFT fit for {obs_name} succeeded "
                            f"(rotation {direct_result[1]:.4f} degrees), "
                            f"running the angle scan ..."
                        )

                    expected_shift = direct_result[0]

                    scan_sigmas = self._narrow_scan_sigmas(
                        sigma_grid,
                        direct_result,
                        initial_guess_translation,
                        initial_guess_rotation,
                        obs_name,
                    )

                elif (
                    self._last_failure_stage == "rotation"
                    and self.scan_on_rotation_failure
                ):
                    # the direct rotation estimate failed, not necessarily the
                    # translation: check that the translation can be measured
                    # at all (at 0 degrees, cheap), then let the scan find the
                    # rotation, with that translation as the expected one
                    if self.logger:
                        self.logger.info(
                            f"Direct FFT rotation estimate failed for {obs_name}, "
                            f"trying a translation-only fit at 0 degrees before "
                            f"the angle scan ..."
                        )

                    translation_only = self.run(
                        reference_frame_backup,
                        input_frame_backup,
                        obs_name,
                        reference_wcs=reference_wcs,
                        input_wcs=input_wcs,
                        initial_guess_translation=initial_guess_translation,
                        initial_guess_rotation=initial_guess_rotation,
                        save_QA=save_QA,
                        show_QA=show_QA,
                        skip_rotation_estimation=True,
                        scan_for_angles=False,
                        **kwargs,
                    )

                    # no direct result to fall back on: its rotation (0) was
                    # not measured
                    direct_result = None

                    if not translation_only[-1]:
                        if self.logger:
                            self.logger.warning(
                                f"Failed FFT aligment for {obs_name}: the direct "
                                f"rotation estimate and the translation-only fit "
                                f"failed, angle scan not run"
                            )
                        return self.FAILED_RETURN

                    expected_shift = translation_only[0]

                    if self.logger:
                        self.logger.info(
                            f"Translation-only fit for {obs_name} succeeded "
                            f"({expected_shift[0]:.3f}, {expected_shift[1]:.3f}) "
                            f"px, running the angle scan with the full sigma range ..."
                        )

                else:
                    if self.logger:
                        self.logger.warning(
                            f"Failed FFT aligment for {obs_name}: the direct fit "
                            f"failed, angle scan not run"
                        )

                    # the failed difference image was made by the direct fit
                    return self.FAILED_RETURN

            def _evaluate_angle(angle, sigmas):
                diff, result = self._scan_difference_metric(
                    reference_frame_backup,
                    input_frame_backup,
                    angle,
                    lambda rotated_input: _estimate_translation(
                        reference_frame_backup, rotated_input, sigmas=sigmas
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
                lambda angle: _evaluate_angle(angle, scan_sigmas),
                obs_name,
                save_QA=save_QA,
                show_QA=show_QA,
                expected_shift=expected_shift,
            )

            # a narrowed sigma range can miss the scales that work at the other
            # angles (the direct fit agreed on them at its own angle only):
            # redo the scan once with the full sigma grid
            if scan_result is None and len(scan_sigmas) < len(sigma_grid):
                if self.logger:
                    self.logger.warning(
                        f"Angle scan with narrowed sigmas ({len(scan_sigmas)} of "
                        f"{len(sigma_grid)}) failed for {obs_name}, redoing it "
                        f"with the full sigma range ..."
                    )

                scan_result = self._scan_for_best_angle(
                    lambda angle: _evaluate_angle(angle, sigma_grid),
                    obs_name,
                    save_QA=save_QA,
                    show_QA=show_QA,
                    expected_shift=expected_shift,
                )

            if scan_result is None:

                # the direct fit was good, only the scan could not improve on
                # it (degenerate metric, boundary of the angle range): keep it
                if direct_result is not None and self.fallback_to_direct_fit:
                    if self.logger:
                        self.logger.warning(
                            f"Angle scan found no rotation for {obs_name}, "
                            f"falling back to the direct fit: translation "
                            f"({direct_result[0][0]:.4f}, {direct_result[0][1]:.4f}) "
                            f"px, rotation {direct_result[1]:.4f} degrees"
                        )

                    # the difference image of the direct fit
                    self.plot_difference_image(
                        reference_frame_backup,
                        input_frame_backup,
                        direct_result[0][0],
                        direct_result[0][1],
                        direct_result[1],
                        obs_name,
                        show=False,
                    )

                    self._last_failure_stage = None
                    return direct_result

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

                self._last_failure_stage = "rotation"
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

                self._last_failure_stage = "rotation"
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
                self._save_QA_figure(f"{self.QA_path}/eps_vs_sigma_{obs_name}.png")
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

            self._last_failure_stage = "translation"
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

    The filtered frames are scaled to unit standard deviation before TV-L1
    (otherwise the flow can stay frozen at zero), and every result is checked
    with a control-shift test (`_control_shift_check`): if it fails, the
    alignment is set as failed.
    """

    def __init__(
        self,
        num_per_dimension=50,
        residual_threshold=0.5,
        residual_sigma=3.0,
        residual_threshold_min=0.05,
        bootstrap_tiles=5,
        bootstrap_samples=200,
        bootstrap_seed=1,
        bootstrap_min_tiles=10,
        normalise_flow_input=True,
        validate=True,
        control_offset=(1.7, -2.3),
        control_tolerance=0.3,
        logger=None,
        QA_path=None,
    ):
        """
        Parameters
        ----------
        normalise_flow_input : bool
            Scale the filtered frames (after spacepylot's histogram
            equalisation and difference-of-gaussians high-pass) to unit
            standard deviation before TV-L1 (default True). The filtered frames
            of smooth (e.g. galaxy) fields only span ~1e-2, and then the TV-L1
            data term is too weak against the smoothness term: the flow stays
            at its starting value of zero (a synthetic 3.6 px shift came out
            as 0.02 px). The TV-L1 parameters are tuned for unit-scale images.

        validate : bool
            Run the control-shift check on every result (default True), see
            `_control_shift_check`. A result that fails it is set as failed.

        control_offset : tuple
            Known shift (x, y) in px applied to the aligned input for the
            control-shift check (default (1.7, -2.3)).

        control_tolerance : float
            Maximum error (px) of the recovered control shift (default 0.3).
            A flow that does not respond to the image (e.g. frozen at zero)
            fails by the full control shift, 2.9 px.

        num_per_dimension : int
            Number of flow vectors per axis that are sampled from the dense flow
            field and fitted by spacepylot (default 50, spacepylot's default,
            i.e. 50 x 50 = 2500 points).

        residual_threshold : float
            Fixed inlier threshold in pixels of the first, rough step (RANSAC,
            or the distance to the median flow vector for a translation-only
            fit), default 0.5 (spacepylot's RANSAC default). In the refinement
            it is the upper limit of the adaptive threshold.

        residual_sigma : float
            Adaptive inlier threshold of the refinement, in robust sigmas of
            the residual scatter (default 3), see
            `stats.robust_residual_threshold`. Clipped to
            [`residual_threshold_min`, `residual_threshold`].

        residual_threshold_min : float
            Lower limit of the adaptive threshold in pixels (default 0.05).

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
            bootstrap is not trusted: a warning is logged and conservative
            fallback errors are used (the residual scatter of the flow vectors
            over sqrt(number of tiles), see `bootstrap_uncertainty`)
            (default 10).
        """
        super().__init__("OpticalFlow", "0.0.0", logger=logger, QA_path=QA_path)

        self.num_per_dimension = num_per_dimension
        self.residual_threshold = residual_threshold
        self.residual_sigma = residual_sigma
        self.residual_threshold_min = residual_threshold_min
        self.bootstrap_tiles = bootstrap_tiles
        self.bootstrap_samples = bootstrap_samples
        self.bootstrap_seed = bootstrap_seed
        self.bootstrap_min_tiles = bootstrap_min_tiles
        self.normalise_flow_input = normalise_flow_input
        self.validate = validate
        self.control_offset = tuple(float(v) for v in control_offset)
        self.control_tolerance = control_tolerance

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
           the residuals of ALL points, inliers = residual < adaptive
           threshold; repeat until the inlier set no longer changes (at most
           10 times). The threshold is `residual_sigma` robust sigmas of the
           residual lengths of the current inliers
           (`stats.robust_residual_threshold`), clipped to
           [`residual_threshold_min`, `residual_threshold`]: the fixed 0.5 px
           of the rough step is a ~7 sigma cut for typical TV-L1 residuals
           (~0.07 px per axis) and lets vectors with a systematic error of
           several tenths of a pixel into the fit, against reported errors of
           ~0.005 px. Rejected vectors can come back in a later pass.
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
            reference -> input matrix), threshold (final inlier threshold, px)
            and sigma (residual scatter per axis, px). None if there are fewer
            than 3 inliers.
        """
        # 1) flow field and the sparse point pairs, in full-frame coordinates
        op.v, op.u = op.optical_flow()

        border = float(op.filter_params.get("remove_boundary_pixels", 0))

        y_sparse, x_sparse, v_sparse, u_sparse = spacepylot_utils.get_sparse_vector_grid(
            op.v, op.u, self.num_per_dimension
        )

        src = np.column_stack([np.ravel(x_sparse), np.ravel(y_sparse)]) + border
        dst = src + np.column_stack([np.ravel(u_sparse), np.ravel(v_sparse)])

        # kept for the flow field QA plot, also when the fit fails
        self._last_flow = dict(src=src, dst=dst)

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

        # 3) refine with least squares and an adaptive threshold until the
        #    inlier set is stable
        params = None
        threshold, sigma = self.residual_threshold, np.nan
        for _ in range(10):
            if inliers.sum() < 3:
                if self.logger:
                    self.logger.warning(
                        f"Optical flow: only {int(inliers.sum())} inliers within "
                        f"{threshold:.3f} px, no fit"
                    )
                return None

            params = self._rigid_fit(src[inliers], dst[inliers], rotation=rotation)
            res = residuals(params)

            threshold, sigma = robust_residual_threshold(
                res[inliers],
                n_sigma=self.residual_sigma,
                minimum=self.residual_threshold_min,
                maximum=self.residual_threshold,
            )
            new_inliers = res < threshold

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
                f"Optical flow fit: {int(inliers.sum())}/{int(finite.sum())} inliers "
                f"(threshold {threshold:.3f} px = {self.residual_sigma:g} x sigma "
                f"{sigma:.3f} px, clipped to [{self.residual_threshold_min:g}, "
                f"{self.residual_threshold:g}]), "
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
            threshold=threshold,
            sigma=sigma,
        )

    # ------------------------------------------------------------------
    # Flow input and control-shift check
    # ------------------------------------------------------------------

    @staticmethod
    def _highpass_unit_std(image, highpass=None, **highpass_kwargs):
        """
        spacepylot's high-pass filter followed by scaling to unit standard
        deviation (see `normalise_flow_input`).
        """
        if highpass is not None:
            image = highpass(image, **highpass_kwargs)

        std = np.nanstd(image)

        if np.isfinite(std) and std > 0:
            image = (image - np.nanmean(image)) / std

        return image

    def _flow_filter_params(self):
        """
        Filter parameters handed to spacepylot: its defaults, with the
        high-pass followed by the unit-std scaling if `normalise_flow_input`.
        """
        params = dict(spacepylot_filter_params)

        if self.normalise_flow_input:
            params["hpf"] = functools.partial(
                self._highpass_unit_std, highpass=spacepylot_filter_params["hpf"]
            )

        return params

    def _single_flow_fit(self, reference_frame, input_frame, rotation):
        """
        TV-L1 flow field (spacepylot) and its rigid fit
        (`fit_flow_transform`). Returns the fit dict or None.
        """
        op = align.AlignOpticalFlow(
            input_frame, reference_frame, filter_params=self._flow_filter_params()
        )

        # (redirect spacepylot prints to logger)
        with redirect_stdout_to_logger(logger):
            return self.fit_flow_transform(op, reference_frame.shape, rotation=rotation)

    def _control_shift_check(self, reference_frame, input_frame, result):
        """
        Control-shift check of an alignment result (tx, ty, rot).

        The input is aligned with the result and then shifted by the known
        `control_offset` (one interpolation, rotation about the frame centre
        then shift, as in `shift_image`; NaNs filled with the median). A
        translation-only flow fit of this frame must recover the control
        shift: the measured correction must be within `control_tolerance` of
        -`control_offset`. A flow that does not follow the image (e.g. frozen
        at zero) fails this by the full control shift.

        Returns (passed, error), error = distance (px) between the recovered
        and the applied control shift (inf if the fit failed).
        """
        cx, cy = self.control_offset
        centre = ((input_frame.shape[1] - 1) / 2.0, (input_frame.shape[0] - 1) / 2.0)

        tx, ty, rot = matrix_to_parameters(
            transformation_matrix(cx, cy, 0.0, centre)
            @ transformation_matrix(*result, centre),
            centre,
        )

        data = np.ma.filled(np.asanyarray(input_frame, dtype=float), np.nan)
        median = np.nanmedian(data)
        data = np.nan_to_num(data, nan=float(median) if np.isfinite(median) else 0.0)

        if rot != 0:
            data = rotate(data, angle=rot, reshape=False, order=3, mode="nearest")
        controlled = shift(data, shift=(ty, tx), order=3, mode="nearest")

        control_fit = self._single_flow_fit(reference_frame, controlled, rotation=False)

        if control_fit is None:
            error = np.inf
        else:
            error = float(np.hypot(control_fit["tx"] + cx, control_fit["ty"] + cy))

        passed = error <= self.control_tolerance

        if self.logger:
            self.logger.info(
                f"Optical flow control-shift check: control shift ({cx:g}, {cy:g}) px "
                f"recovered within {error:.3f} px (tolerance "
                f"{self.control_tolerance:g} px): {'passed' if passed else 'FAILED'}"
            )

        return passed, error

    # ------------------------------------------------------------------
    # Uncertainties: spatial block bootstrap of the inliers
    # ------------------------------------------------------------------

    def bootstrap_uncertainty(self, fit, obs_name=None):
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
            Observation name for logging.

        Returns
        -------
        dict
            success (bool), errors (tx_err, ty_err, rot_err; rot_err = 0 for a
            translation-only fit), draws (bootstrap_samples x 3 array of
            tx, ty, rot), median of the draws, n_inliers, n_points, n_tiles.
            With success False (inliers in fewer than `bootstrap_min_tiles`
            tiles) the errors are a conservative fallback: the residual scatter
            of the flow vectors (`fit["sigma"]`) over sqrt(number of tiles with
            inliers), each tile counted as one independent measurement, and
            for the rotation that error over the rms lever arm of the inliers.
            NaN if the residual scatter is unknown, never 0.
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

        # fallback errors if the bootstrap cannot be done: the inliers are in
        # too few tiles, so treat every tile as a single independent
        # measurement with the residual scatter of the flow vectors. This is
        # conservative, and never 0 (an error of 0 would look like a perfect
        # measurement downstream)
        sigma = float(fit.get("sigma", np.nan))
        n_independent = max(len(tiles), 1)
        fallback_t_err = sigma / np.sqrt(n_independent)
        fallback_rot_err = 0.0
        if rotation:
            arm = src[inliers] - np.asarray(centre, float)
            r_rms = np.sqrt(np.nanmean(np.sum(arm**2, axis=1))) if n_inliers else np.nan
            fallback_rot_err = (
                float(np.degrees(fallback_t_err / r_rms))
                if np.isfinite(r_rms) and r_rms > 0
                else np.nan
            )
        if not np.isfinite(fallback_t_err) or fallback_t_err <= 0:
            fallback_t_err = np.nan

        fail = dict(
            success=False,
            errors=(fallback_t_err, fallback_t_err, fallback_rot_err),
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
                    f"inliers, need {self.bootstrap_min_tiles}. Using the "
                    f"conservative fallback errors (residual scatter "
                    f"{sigma:.3f} px / sqrt({n_independent} tiles)): tx_err = "
                    f"ty_err = {fallback_t_err:.4f} px"
                    + (f", rot_err = {fallback_rot_err:.5f} degrees" if rotation else "")
                    + "."
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

        return result

    # ------------------------------------------------------------------
    # QA: the flow field
    # ------------------------------------------------------------------

    def plot_flow_field(
        self,
        reference_frame,
        obs_name,
        fit=None,
        result=None,
        errors=None,
        scan_rotation=None,
        failed_reason=None,
        notes=None,
        save=True,
        show=False,
    ):
        """
        QA plot of the optical-flow field behind an alignment, made for every
        alignment (also failed ones).

        Panels
        ------
        1. Measured flow: the flow vectors fitted by `fit_flow_transform`
           (where every reference pixel is found in the input), over the
           reference frame, coloured by length; RANSAC / refit outliers as red
           dots. For a clean rigid offset the arrows are all parallel (pure
           shift) or swirl around one point (rotation).
        2. Fitted model: the flow predicted by the fitted rigid transform at
           the same points, with the same arrow scale as panel 1. The two
           panels should look alike.
        3. Residual flow (measured - model) of the inliers, with its own
           (larger) arrow scale. Random arrows are noise; a coherent pattern
           (a region pulling one way, a radial pattern) shows what the rigid
           model does not describe: a bad region, distortion, a scale change.
        4. Residual vectors of all points in the (dx, dy) plane, with the
           adaptive inlier threshold and its upper limit `residual_threshold`
           as circles.

        Only up to 25 x 25 arrows are drawn (a regular subset of the grid);
        the outliers, the residual scatter and all numbers use all points.
        If the fit failed (`fit` None), only the measured flow of the last
        flow computation is shown, without inlier information.

        Parameters
        ----------
        reference_frame : np.2darray
            Shown under the measured flow.

        obs_name : str
            Observation name for the title and the file name.

        fit : dict, optional
            Result of `fit_flow_transform`.

        result : tuple, optional
            Reported (tx, ty, rot) of the alignment.

        errors : tuple, optional
            Reported (tx_err, ty_err, rot_err).

        scan_rotation : float, optional
            Rotation from the angle scan (degrees): the flow was then measured
            on the input rotated by this angle, so only a translation is left.

        failed_reason : str, optional
            Why the alignment was set as failed (shown in the title).

        notes : list of str, optional
            Extra title lines (pre-warp, iterations, validation checks).

        save, show : bool
            Save to `flow_field_{obs_name}_{name}.png` / show the plot.
        """
        if fit is not None:
            src, dst, inliers = fit["src"], fit["dst"], fit["inliers"]
        elif getattr(self, "_last_flow", None) is not None:
            src, dst, inliers = self._last_flow["src"], self._last_flow["dst"], None
        else:
            if self.logger:
                self.logger.warning(
                    f"No flow field to plot for {obs_name}, skipping flow QA plot."
                )
            return

        flow = dst - src
        finite = np.all(np.isfinite(src), axis=1) & np.all(np.isfinite(flow), axis=1)

        if not finite.any():
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 14))
        ax_flow, ax_model, ax_res, ax_scatter = axes.ravel()

        vmin, vmax = np.nanpercentile(reference_frame, [5, 95])

        def _background(ax):
            ax.imshow(
                reference_frame,
                cmap="gray",
                origin="upper",
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
                alpha=0.6,
            )
            ax.set_xlabel("x [px]")
            ax.set_ylabel("y [px]")

        # ---- 1) measured flow -------------------------------------------
        good = finite if inliers is None else finite & inliers

        sub, spacing = self._quiver_subsample(src, good)
        magnification, reference = self._arrow_magnification(flow[good], spacing)

        _background(ax_flow)
        self._draw_flow_quiver(
            ax_flow, fig, src, flow, sub, spacing, magnification, reference,
            "|flow| [px]",
        )

        if inliers is not None and np.any(finite & ~inliers):
            ax_flow.scatter(
                src[finite & ~inliers, 0],
                src[finite & ~inliers, 1],
                s=4,
                color="red",
                label=f"outliers ({int(np.sum(finite & ~inliers))})",
            )
            ax_flow.legend(loc="lower right", fontsize=8)

        n_text = (
            f"{int(np.sum(finite))} vectors"
            if inliers is None
            else f"{int(np.sum(finite & inliers))}/{int(np.sum(finite))} inliers"
        )
        ax_flow.set_title(
            f"Measured flow, arrows x{magnification:.3g} ({n_text})", pad=22
        )

        if fit is None:
            for ax in (ax_model, ax_res, ax_scatter):
                ax.axis("off")
                ax.text(
                    0.5,
                    0.5,
                    "no fit",
                    ha="center",
                    va="center",
                    fontsize=16,
                    transform=ax.transAxes,
                )
        else:
            params = fit["params"]
            model_flow = src @ params[:2, :2].T + params[:2, 2] - src
            residual = flow - model_flow

            # ---- 2) fitted model, same arrow scale, at all grid points -----
            sub_all, _ = self._quiver_subsample(src, finite)
            _background(ax_model)
            self._draw_flow_quiver(
                ax_model, fig, src, model_flow, sub_all, spacing, magnification,
                reference, "|model flow| [px]",
            )
            ax_model.set_title(
                f"Fitted rigid model flow (reference -> input), arrows "
                f"x{magnification:.3g}\n(same arrow scale as the measured flow)",
                pad=22,
            )

            # ---- 3) residual flow, own scale ------------------------------
            res_mag, res_ref = self._arrow_magnification(residual[good], spacing)
            rms = float(np.sqrt(np.mean(np.sum(residual[good] ** 2, axis=1))))

            _background(ax_res)
            self._draw_flow_quiver(
                ax_res, fig, src, residual, sub, spacing, res_mag, res_ref,
                "|residual| [px]",
            )
            if np.any(finite & ~inliers):
                ax_res.scatter(
                    src[finite & ~inliers, 0],
                    src[finite & ~inliers, 1],
                    s=4,
                    color="red",
                    label="outliers",
                )
                ax_res.legend(loc="lower right", fontsize=8)
            ax_res.set_title(
                f"Residual flow (measured - model) of the inliers, arrows "
                f"x{res_mag:.3g}\nrms {rms:.3f} px",
                pad=22,
            )

            # ---- 4) residual vectors --------------------------------------
            ax_scatter.scatter(
                residual[good, 0],
                residual[good, 1],
                s=3,
                color="cyan",
                alpha=0.5,
                label=f"inliers ({int(np.sum(good))})",
            )
            outl = finite & ~inliers
            if outl.any():
                ax_scatter.scatter(
                    residual[outl, 0],
                    residual[outl, 1],
                    s=3,
                    color="red",
                    alpha=0.5,
                    label=f"outliers ({int(np.sum(outl))})",
                )
            threshold = fit.get("threshold", self.residual_threshold)
            ax_scatter.add_patch(
                plt.Circle(
                    (0, 0),
                    threshold,
                    fill=False,
                    color="white",
                    linestyle="--",
                    label=(
                        f"inlier threshold ({threshold:.3f} px = "
                        f"{self.residual_sigma:g} sigma, {fit.get('sigma', np.nan):.3f} px)"
                    ),
                )
            )
            ax_scatter.add_patch(
                plt.Circle(
                    (0, 0),
                    self.residual_threshold,
                    fill=False,
                    color="gray",
                    linestyle=":",
                    label=f"upper limit ({self.residual_threshold:g} px)",
                )
            )
            ax_scatter.axhline(0, color="gray", linewidth=0.8)
            ax_scatter.axvline(0, color="gray", linewidth=0.8)

            # range: the threshold circle and the inliers, outliers may be far off
            limit = 1.5 * self.residual_threshold
            ax_scatter.set_xlim(-limit, limit)
            ax_scatter.set_ylim(-limit, limit)
            ax_scatter.set_aspect("equal")
            ax_scatter.set_xlabel("residual x [px]")
            ax_scatter.set_ylabel("residual y [px]")
            ax_scatter.set_title(
                f"Residual vectors (median |residual| of the inliers "
                f"{np.median(np.linalg.norm(residual[good], axis=1)):.3f} px)"
            )
            ax_scatter.legend(loc="upper right", fontsize=8)

        # ---- title: the reported result -----------------------------------
        title = f"Optical flow field for {obs_name}"

        if failed_reason is not None:
            title += f"\nFAILED: {failed_reason}"
        elif result is not None:
            tx, ty, rot = result
            tx_err, ty_err, rot_err = errors if errors is not None else (np.nan,) * 3
            title += (
                f"\nCorrection of the input: tx = {tx:.3f} +/- {tx_err:.3f} px, "
                f"ty = {ty:.3f} +/- {ty_err:.3f} px, "
                f"rot = {rot:.4f} +/- {rot_err:.4f} deg"
            )

        if scan_rotation is not None:
            title += (
                f"\n(rotation from the angle scan: flow measured on the input "
                f"rotated by {scan_rotation:.4f} deg, translation only)"
            )

        for note in notes or []:
            title += f"\n{note}"

        fig.suptitle(title, fontsize=13)
        # leave room for the (multi-line) title
        n_lines = title.count("\n") + 1
        fig.tight_layout(rect=(0, 0, 1, 1.0 - 0.017 * max(3, n_lines)))

        if save:
            filename = f"{self.QA_path}/flow_field_{obs_name}_{self.name}.png"
            self._save_QA_figure(filename, fig)
            if self.logger:
                self.logger.info(f"Saved flow field QA plot to {filename}")

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

        # an empty (all-NaN) frame cannot be aligned. It has to be caught
        # before the identity check below: the difference to an empty frame
        # is all NaN, its nansum 0, and the pair would pass as identical
        reference_finite = np.isfinite(
            np.ma.filled(np.asanyarray(reference_frame, dtype=float), np.nan)
        )
        input_finite = np.isfinite(
            np.ma.filled(np.asanyarray(input_frame, dtype=float), np.nan)
        )

        if not (reference_finite & input_finite).any():
            self.logger.warning(
                f"No common finite pixels between the reference and input frame "
                f"for {obs_name} (empty or corrupt frame) - setting aligment as failed."
            )
            return self.FAILED_RETURN

        if np.array_equal(reference_finite, input_finite) and (
            np.nansum(np.abs(reference_frame - input_frame)) < 1e-6
        ):
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

        self._last_flow = None

        def _flow_QA(fit, **plot_kwargs):
            # flow field QA plot (successful alignments only; a failed
            # alignment only gets the failed difference image)
            if save_QA or show_QA:
                self.plot_flow_field(
                    reference_frame,
                    obs_name,
                    fit=fit,
                    scan_rotation=rot if scan_for_angles else None,
                    save=save_QA,
                    show=show_QA,
                    **plot_kwargs,
                )

        # spacepylot only computes the flow field, the transform is fitted by
        # `fit_flow_transform` (see there for why)
        def _fit(frame, rotation):
            return self._single_flow_fit(reference_frame, frame, rotation)

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

        # control-shift check: the flow must recover a known shift of the
        # aligned input, otherwise the result is not trusted
        control_error = None
        if self.validate:
            passed, control_error = self._control_shift_check(
                reference_frame, input_frame, (tx, ty, rot)
            )
            if not passed:
                self.logger.warning(
                    f"Optical flow for {obs_name} failed the control-shift check "
                    f"(control shift recovered within {control_error:.2f} px, "
                    f"tolerance {self.control_tolerance:g} px) - setting aligment "
                    f"as failed."
                )
                self.plot_failed_aligment(
                    reference_frame, input_frame, obs_name, show=show_QA
                )
                return self.FAILED_RETURN

        # uncertainties from the spatial block bootstrap of the inliers; for the
        # angle scan only tx, ty (the rotation error comes from the scan)
        bootstrap = self.bootstrap_uncertainty(fit, obs_name=obs_name)

        tx_err, ty_err = bootstrap["errors"][:2]

        if fit["rotation"]:
            rot_err = bootstrap["errors"][2]

        _flow_QA(
            fit,
            result=(tx, ty, rot),
            errors=(tx_err, ty_err, rot_err),
            notes=(
                [
                    f"Control-shift check passed: control shift recovered within "
                    f"{control_error:.3f} px (tolerance {self.control_tolerance:g} px)"
                ]
                if control_error is not None
                else None
            ),
        )

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
