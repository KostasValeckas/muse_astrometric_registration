from abc import ABC, abstractmethod
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
from geometry import reproject_image_to_target_wcs
from skimage.transform import warp_polar
from skimage.registration import phase_cross_correlation
from filters import remove_nans


from skimage.filters import window, difference_of_gaussians

from scipy.fft import fft2, fftshift


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
        v_lower_per = 15,
        v_upper_per = 85
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


        _, axes = plt.subplots(2, 2, figsize=(12, 12))

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

        for ax in axes.ravel():
            ax.set_aspect("equal")

        if custom_title is not None:
            plt.suptitle(
                custom_title
                + f"\n Offsets: dx={dx:.2f}, dy={dy:.2f}, dtheta={dtheta:.2f} deg"
            )
        else:
            plt.suptitle(
                f"Difference Image for {obs_name}, method: {self.name}\n Offsets: dx={dx:.2f}, dy={dy:.2f}, dtheta={dtheta:.2f} deg"
            )
        plt.savefig(f"{self.QA_path}/difference_image_{obs_name}.png")

        if self.logger:
            self.logger.info(
                f"Saved difference image to {self.QA_path}/difference_image_{obs_name}.png"
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

        good_data = self.check_input(reference_frame, input_frame)

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

                fig = plt.figure(figsize=(10, 5))

                ax1 = plt.subplot(1, 2, 1)
                ax1.imshow(
                    reference_frame,
                    cmap="gray",
                    vmin=np.nanpercentile(reference_frame.compressed(), 5),
                    vmax=np.nanpercentile(reference_frame.compressed(), 95),
                )
                ax1.set_title("Reference Frame")

                ax2 = plt.subplot(1, 2, 2)
                ax2.imshow(
                    input_frame,
                    cmap="gray",
                    vmin=np.nanpercentile(input_frame.compressed(), 5),
                    vmax=np.nanpercentile(input_frame.compressed(), 95),
                )
                ax2.set_title("Input Frame")

                fig.suptitle(
                    f"Failed Alignment for {obs_name}: Could not find matching sources",
                )

                plt.savefig(
                    f"{self.QA_path}/failed_alignment_{obs_name}.png",
                )

                if self.logger:
                    self.logger.info(
                        f"Saved QA plot of failed alignment to "
                        f"{self.QA_path}/failed_alignment_{obs_name}.png"
                    )

                if show_QA:
                    plt.show()
                else:
                    plt.close()

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

            # plot the images that failed for QA
            if save_QA:
                fig = plt.figure(figsize=(10, 5))

                ax1 = plt.subplot(1, 2, 1)
                ax1.imshow(
                    reference_frame,
                    cmap="gray",
                    vmin=np.nanpercentile(reference_frame.compressed(), 5),
                    vmax=np.nanpercentile(reference_frame.compressed(), 95),
                )
                ax1.set_title("Reference Frame")

                ax2 = plt.subplot(1, 2, 2)
                ax2.imshow(
                    input_frame,
                    cmap="gray",
                    vmin=np.nanpercentile(input_frame.compressed(), 5),
                    vmax=np.nanpercentile(input_frame.compressed(), 95),
                )
                ax2.set_title("Input Frame")

                fig.suptitle(
                    f"Failed Alignment for {obs_name}: Not enough matched sources",
                )

                plt.savefig(
                    f"{self.QA_path}/failed_alignment_{obs_name}.png",
                )

                if self.logger:
                    self.logger.info(
                        f"Saved QA plot of failed alignment to "
                        f"{self.QA_path}/failed_alignment_{obs_name}.png"
                    )

                if show_QA:
                    plt.show()
                else:
                    plt.close()

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
                (theta, offset, theta_err, offset_err, _) = (
                    self.fit_euclidean_transform(source_fit, target_fit, centre, p0=p0)
                )
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to fit transformation: {e}")
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
            return self.FAILED_RETURN

        if self.logger:
            self.logger.info(
                f"Performing final fit using " f"{len(source_fit)} accepted sources."
            )

        try:
            (theta, offset, theta_err, offset_err, _) = self.fit_euclidean_transform(
                source_fit, target_fit, centre, p0=p0
            )
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed final transformation fit: {e}")
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
                reference_frame, input_frame, tx, ty, rot, obs_name, show=show_QA
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

    def __init__(self, oversample_factor=4, extra_rotation = 10.0, gaussian_sigma=0.0, polar_radius=10000.0, logger=None, QA_path=None):

        self.oversample_factor = oversample_factor
        self.extra_rotation = extra_rotation
        self.gaussian_sigma = gaussian_sigma
        self.polar_radius = polar_radius

        super().__init__("FFT", "0.0.0", logger=logger, QA_path=QA_path)

        if self.logger:
            self.logger.info(
                f"Initialized {self.name} alignment method, version {self.version}."
            )

    def _upsampled_dft(
        self, data, upsampled_region_size, upsample_factor=1, axis_offsets=None
    ):
        """
        Upsampled DFT by matrix multiplication (Guizar-Sicairos et al. 2008).
        Computes a small, upsampled region of an inverse DFT, without
        zero-padding and IFFT'ing the whole array.
        Args:
            data: 2D array in the frequency domain (e.g. F0 * conj(F1) / |...|)
            upsampled_region_size: (rows, cols) size of the region to
                compute, in *upsampled* pixel units
            upsample_factor: upsampling factor (kappa)
            axis_offsets: (row_offset, col_offset) - where the region is
                centered, in upsampled-pixel units
        Returns:
            2D complex array of shape upsampled_region_size
        """
        
        if np.isscalar(upsampled_region_size):
            upsampled_region_size = (upsampled_region_size,) * data.ndim
        
        if axis_offsets is None:
            axis_offsets = (0,) * data.ndim
        
        im2pi = 1j * 2 * np.pi
        
        out = data
        
        # apply the 1D kernel along each axis, last axis first
        
        for dim in reversed(range(data.ndim)):
            n_items = data.shape[dim]
            ups_size = upsampled_region_size[dim]
            ax_offset = axis_offsets[dim]
            kernel = (np.arange(ups_size) - ax_offset)[:, None] * fft.fftfreq(
                n_items, upsample_factor
            )
            kernel = np.exp(-im2pi * kernel)
            out = np.tensordot(kernel, out, axes=(1, dim))
            out = np.moveaxis(out, 0, dim)
        
        return out

    def phase_correlation(im0, im1, target_x, target_y, cutoff=None):

        window = tukey_window_2d(im0.shape, alpha=0.25)
        im0 = apply_taper(im0, window=window)
        im1 = apply_taper(im1, window=window)
        target_coords = np.array([target_y, target_x])
        f0, f1 = [fft.fft2(arr) for arr in (im0, im1)]
        eps = abs(f1).max() * 1e-15
        num = (f0 * f1.conjugate()) / (abs(f0) * abs(f1) + eps)
        # Remove low spatial frequencies.
        if cutoff is not None:
            fy = fft.fftfreq(im0.shape[0])
            fx = fft.fftfreq(im0.shape[1])
            FX, FY = np.meshgrid(fx, fy)
            R = np.sqrt(FX**2 + FY**2)
            num[R < cutoff] = 0
        cps = abs(fft.ifft2(num))
        scps = fft.fftshift(cps)
        target_coords_shifted = target_coords + np.array(scps.shape) / 2
        return scps, num, target_coords_shifted

    def count_peaks(a, size=2, threshold_fraction=0.2):

        threshold = threshold_fraction * np.max(a)
        local_max = a == maximum_filter(a, size=size)
        local_max &= a >= threshold
        counts = np.count_nonzero(local_max)
        return counts, threshold

    def rotation_only(self, F_ref, F_mov, n_angles=1440, r_min=0.02, r_max=0.35,
                      max_angle=None):
        """
        Rotation of mov relative to ref, assuming scale == 1.
        F_* : unshifted 2D spectra (DC at [0, 0]), ideally high-passed.
        r_min, r_max : radial band in cycles/pixel to use (excludes DC region
                       and noisy high frequencies).
        max_angle : if given (deg), only accept |angle| <= max_angle (e.g. from WCS).
        """
        h, w = F_ref.shape
        radius = min(h, w) // 2

        def angular_profile(F):
            mag = np.log1p(np.abs(fft.fftshift(F)))
            pol = warp_polar(mag, radius=radius, output_shape=(n_angles, radius),
                             scaling="linear")          # (angle, radius)
            # radial band in pixels of the polar map -> cycles/pixel = r / min(h, w)
            r = np.arange(radius) / min(h, w)
            band = (r >= r_min) & (r <= r_max)
            prof = pol[:, band].sum(axis=1)
            return prof - prof.mean()

        p_ref, p_mov = angular_profile(F_ref), angular_profile(F_mov)

        # circular cross-correlation via FFT
        cc = np.real(fft.ifft(fft.fft(p_ref) * np.conj(fft.fft(p_mov))))

        # optionally restrict the search window around 0 deg
        lags = np.arange(n_angles)
        lags = np.where(lags > n_angles // 2, lags - n_angles, lags)   # signed bins
        deg_per_bin = 360.0 / n_angles
        if max_angle is not None:
            cc = np.where(np.abs(lags * deg_per_bin) <= max_angle, cc, -np.inf)

        k = int(np.argmax(cc))

        # sub-bin refinement: parabola through the peak and its neighbours
        y0, y1, y2 = cc[(k - 1) % n_angles], cc[k], cc[(k + 1) % n_angles]
        denom = y0 - 2 * y1 + y2
        delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0

        return (lags[k] + delta) * deg_per_bin

    def rotation_scale(
        self,
        F_ref,
        F_mov,
        upsample=1000,
        n_angles=4096,
        max_angle=5.0,
    ):
        mag_ref = np.abs(fft.fftshift(F_ref))
        mag_mov = np.abs(fft.fftshift(F_mov))

        h, w = mag_ref.shape
        radius = min(h, w) // 2

        # Log-polar transform
        out_shape = (n_angles, radius)

        lp_ref = warp_polar(
            mag_ref,
            radius=radius,
            output_shape=out_shape,
            scaling="log",
        )

        lp_mov = warp_polar(
            mag_mov,
            radius=radius,
            output_shape=out_shape,
            scaling="log",
        )

        eps = 1e-12
    

        cross_power = (
            lp_ref * lp_mov.conjugate()
            / (np.abs(lp_ref) * np.abs(lp_mov) + eps)
        )

        correlation = np.abs(
            fft.fftshift(fft.ifft2(cross_power))
        )

        plt.imshow(np.abs(correlation), cmap="hot")
        plt.show()

        # Angular axis
        deg_per_bin = 360.0 / n_angles

        angle_axis = (
            np.arange(n_angles) - n_angles // 2
        ) * deg_per_bin

        plt.plot(angle_axis, correlation[:, correlation.shape[1] // 2])
        plt.xlabel("Angle (degrees)")
        plt.ylabel("Correlation")
        plt.title("Angular Correlation Profile")
        plt.grid()
        plt.show()

        # Restrict to +/- max_angle
        mask = np.abs(angle_axis) <= max_angle

        restricted = correlation.copy()
        restricted[~mask, :] = -np.inf

        iy, ix = np.unravel_index(
            np.argmax(restricted),
            restricted.shape,
        )

        # Sub-bin angular refinement
        y0 = correlation[(iy - 1) % n_angles, ix]
        y1 = correlation[iy, ix]
        y2 = correlation[(iy + 1) % n_angles, ix]

        denom = y0 - 2.0 * y1 + y2

        delta = (
            0.5 * (y0 - y2) / denom
            if denom != 0
            else 0.0
        )

        angle_deg = (
            angle_axis[iy]
            + delta * deg_per_bin
        )

        return angle_deg

    def estimate_rotation(
        self,
        F_ref,
        F_mov,
        n_angles=4096,
        r_min=0.005,
        r_max=0.08,
        max_angle=5.0,
    ):
        """
        Estimate rotation of F_mov relative to F_ref.

        Parameters
        ----------
        F_ref, F_mov : 2D complex FFT arrays
            Unshifted FFTs (DC at [0, 0]).
        n_angles : int
            Angular sampling of the log-polar transform.
        r_min, r_max : float
            Spatial-frequency range in cycles/pixel.
        max_angle : float
            Search range in degrees.
        """

        h, w = F_ref.shape
        n = min(h, w)
        radius = n // 2

        # Fourier magnitudes
        mag_ref = np.abs(fft.fftshift(F_ref))
        mag_mov = np.abs(fft.fftshift(F_mov))

        # Log-polar transform
        lp_ref = warp_polar(
            mag_ref,
            radius=radius,
            output_shape=(n_angles, radius),
            scaling="log",
        )

        lp_mov = warp_polar(
            mag_mov,
            radius=radius,
            output_shape=(n_angles, radius),
            scaling="log",
        )

        # Convert radial coordinate to cycles/pixel.
        # For an N-pixel image, one Fourier pixel = 1/N cycles/pixel.
        radial_freq = np.arange(radius) / n

        band = (
            (radial_freq >= r_min) &
            (radial_freq <= r_max)
        )

        lp_ref = lp_ref[:, band]
        lp_mov = lp_mov[:, band]

        # Remove the mean along angle
        lp_ref -= lp_ref.mean(axis=0, keepdims=True)
        lp_mov -= lp_mov.mean(axis=0, keepdims=True)

        # 2D phase correlation
        F0 = fft.fft2(lp_ref)
        F1 = fft.fft2(lp_mov)

        cross_power = (
            F0 * F1.conj()
            / (np.abs(F0) * np.abs(F1) + 1e-12)
        )

        correlation = np.abs(
            fft.fftshift(
                fft.ifft2(cross_power)
            )
        )

        # Angular axis
        deg_per_bin = 360.0 / n_angles

        angle_axis = (
            np.arange(n_angles) - n_angles // 2
        ) * deg_per_bin

        # Restrict search to ±max_angle
        valid = np.abs(angle_axis) <= max_angle

        masked = correlation.copy()
        masked[~valid, :] = -np.inf

        # Peak
        iy, ix = np.unravel_index(
            np.argmax(masked),
            masked.shape,
        )

        # Sub-bin angular refinement
        y0 = correlation[(iy - 1) % n_angles, ix]
        y1 = correlation[iy, ix]
        y2 = correlation[(iy + 1) % n_angles, ix]

        denom = y0 - 2.0 * y1 + y2

        if denom != 0:
            delta = 0.5 * (y0 - y2) / denom
        else:
            delta = 0.0

        angle_deg = (
            angle_axis[iy]
            + delta * deg_per_bin
        )

        return angle_deg
    
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
        # just testing 

        #input_frame = rgb2gray(data.retina())

    

        window_clean, (height, width) = estimate_nan_window(input_frame, min_run=100)

        target_wcs, shape_out = make_window_wcs(
            input_wcs, input_frame.shape, window_clean, self.oversample_factor
        )
        print("shape_out:", shape_out)   # should be ~ (294, 295) * oversample_factor

        input_frame, _ = reproject_image_to_target_wcs(
            input_frame, input_wcs, target_wcs, shape_out=shape_out
        )

        reference_frame, _ = reproject_image_to_target_wcs(
            reference_frame, reference_wcs, target_wcs, shape_out=shape_out
        )

        # update the wcs

        input_wcs = target_wcs
        reference_wcs = target_wcs
        
        # reference_frame = input_frame.copy()  # just for testing
        # clean for nans: 


        mask = np.isnan(input_frame) | np.isnan(reference_frame)

        input_frame[mask] = 0.0
        reference_frame[mask] = 0.0


        # now rotate the input frame by the extra rotation angle 

        # take a copy before 

        input_frame_copy_prior_rot = input_frame.copy()

        input_frame = rotate(input_frame, self.extra_rotation, reshape=False, cval=np.nan)

        # after rotation we have nans again -> recrop

        window_clean, (height, width) = estimate_nan_window(input_frame, min_run=100)

        target_wcs, shape_out = make_window_wcs(
            input_wcs, input_frame.shape, window_clean
        )
        print("shape_out:", shape_out)   

        input_frame, _ = reproject_image_to_target_wcs(
            input_frame, input_wcs, target_wcs
        )

        reference_frame, _ = reproject_image_to_target_wcs(
            reference_frame, reference_wcs, target_wcs
        )

        input_frame_prior_rot, _ = reproject_image_to_target_wcs(
            input_frame_copy_prior_rot, input_wcs, target_wcs
        )

        # clean for nans: 

        mask = np.isnan(input_frame) | np.isnan(reference_frame)

        input_frame[mask] = 0.0
        reference_frame[mask] = 0.0

        input_frame_prior_rot[mask] = 0.0

 
        # now window with hann windows

        input_frame = input_frame * window('hann', input_frame.shape)
        reference_frame = reference_frame * window('hann', reference_frame.shape)
        input_frame_prior_rot = input_frame_prior_rot * window('hann', input_frame_prior_rot.shape)

        recovered_angles = []
        recovered_scales = []
        angle_control = []
        scale_control = []
        angle_diff = []
        used_simgas = []

        input_frame_copy = input_frame.copy()
        reference_frame_copy = reference_frame.copy()
        input_frame_prior_rot_copy = input_frame_prior_rot.copy()

      


        # testing = hard coded constants

        SCALE_ACCEPT_ERROR = 0.01

        ANGLE_CONTROL_ACCEPT_ERROR = 1

        ANGLE_DIFF_ACCEPT_ERROR = 0.01

        SIGMA_START = 0.0001

        BANDWITH_SIZE = 0.01

        SIGMA_STOP = 15

        CONVERGED = False


        while not CONVERGED:


            input_frame = input_frame_copy.copy()
            reference_frame = reference_frame_copy.copy()
            input_frame_prior_rot = input_frame_prior_rot_copy.copy()


   
            input_frame = difference_of_gaussians(input_frame, SIGMA_START, SIGMA_START + BANDWITH_SIZE)
            reference_frame = difference_of_gaussians(reference_frame, SIGMA_START, SIGMA_START + BANDWITH_SIZE)
            input_frame_prior_rot = difference_of_gaussians(input_frame_prior_rot, SIGMA_START, SIGMA_START + BANDWITH_SIZE)

            image_fs = np.abs(fftshift(fft2(input_frame)))
            rts_fs = np.abs(fftshift(fft2(reference_frame)))
            warped_image_copy_prior_rot_fs = np.abs(fftshift(fft2(input_frame_prior_rot)))

            # Create log-polar transformed FFT mag images and register
            shape = image_fs.shape
            

            warped_image_fs = warp_polar(
                image_fs, radius=self.polar_radius, output_shape=shape, scaling='log', order=0
            )
            warped_rts_fs = warp_polar(
                rts_fs, radius=self.polar_radius, output_shape=shape, scaling='log', order=0
            )
            warped_image_copy_prior_rot = warp_polar(
                warped_image_copy_prior_rot_fs, radius=self.polar_radius, output_shape=shape, scaling='log', order=0
            )

            warped_image_fs = warped_image_fs[: shape[0] // 2, :]  # only use half of FFT
            warped_rts_fs = warped_rts_fs[: shape[0] // 2, :]
            warped_image_copy_prior_rot = warped_image_copy_prior_rot[: shape[0] // 2, :]


            shifts, error, phasediff = phase_cross_correlation(
                warped_image_fs, warped_rts_fs, upsample_factor=100, normalization=None
            )

            shifts_prior_rot, error_prior_rot, phasediff_prior_rot = phase_cross_correlation(
                warped_image_fs, warped_image_copy_prior_rot, upsample_factor=100, normalization=None
            )

            # Use translation parameters to calculate rotation and scaling parameters
            shiftr, shiftc = shifts[:2]
            recovered_angle = (360 / shape[0]) * shiftr
            klog = shape[1] / np.log(self.polar_radius)
            shift_scale = np.exp(shiftc / klog)

            shiftr_prior_rot, shiftc_prior_rot = shifts_prior_rot[:2]
            recovered_angle_prior_rot = (360 / shape[0]) * shiftr_prior_rot
            klog_prior_rot = shape[1] / np.log(self.polar_radius)
            shift_scale_prior_rot = np.exp(shiftc_prior_rot / klog_prior_rot)  


            recovered_angles.append(recovered_angle)
            recovered_scales.append(shift_scale)
            angle_control.append(recovered_angle_prior_rot)
            scale_control.append(shift_scale_prior_rot)
            angle_diff.append(recovered_angle - recovered_angle_prior_rot)

            used_simgas.append(SIGMA_START)




            if np.abs(recovered_angle - recovered_angle_prior_rot) < ANGLE_DIFF_ACCEPT_ERROR and np.abs(shift_scale - 1.0) < SCALE_ACCEPT_ERROR and np.abs(shift_scale_prior_rot - 1.0) < SCALE_ACCEPT_ERROR and np.abs(recovered_angle_prior_rot + self.extra_rotation) < ANGLE_CONTROL_ACCEPT_ERROR:
                CONVERGED = True

                self.logger.info(f"Converged with sigma = {SIGMA_START:.4f}, recovered_angle = {recovered_angle:.4f}, recovered_scale = {shift_scale:.4f}, recovered_angle_prior_rot = {recovered_angle_prior_rot:.4f}, recovered_scale_prior_rot = {shift_scale_prior_rot:.4f}")

                # now just testing, sep everything else than the angle as null and return

                (tx, ty), rot, (tx_err, ty_err), rot_err, success = self.FAILED_RETURN

                rot = recovered_angle - recovered_angle_prior_rot

                print(f"Recovered rotation: {rot:.4f} degrees")



            else:

                self.logger.info(f"Not converged with sigma = {SIGMA_START:.4f}, recovered_angle = {recovered_angle:.4f}, recovered_scale = {shift_scale:.4f}, recovered_angle_prior_rot = {recovered_angle_prior_rot:.4f}, recovered_scale_prior_rot = {shift_scale_prior_rot:.4f}")

                SIGMA_START += BANDWITH_SIZE

                if SIGMA_START > SIGMA_STOP:
                    self.logger.error(f"Failed to converge after reaching sigma = {SIGMA_START:.4f}")
                    (tx, ty), rot, (tx_err, ty_err), rot_err, success = self.FAILED_RETURN
                    break


        # save all the arrays using observation name as prefix

        np.save(f"{self.QA_path}/{obs_name}_recovered_angles.npy", np.array(recovered_angles))
        np.save(f"{self.QA_path}/{obs_name}_recovered_scales.npy", np.array(recovered_scales))
        np.save(f"{self.QA_path}/{obs_name}_angle_control.npy", np.array(angle_control))
        np.save(f"{self.QA_path}/{obs_name}_scale_control.npy", np.array(scale_control))
        np.save(f"{self.QA_path}/{obs_name}_angle_diff.npy", np.array(angle_diff))
        np.save(f"{self.QA_path}/{obs_name}_used_simgas.npy", np.array(used_simgas))

        # plot and save the plots using observation name as prefix

        plt.figure(figsize=(10, 6))
        plt.plot(used_simgas, label="Used Sigmas")
        plt.plot(used_simgas, label="Used Sigmas")
        plt.plot(used_simgas, recovered_angles, label="Recovered Angles")
        plt.plot(used_simgas, angle_control, label="Angle Control")
        plt.plot(used_simgas, angle_diff, label="Angle Difference")
        plt.xlabel("Iteration")
        plt.ylabel("Angle (degrees)")
        plt.title(f"Recovered Angles for {obs_name}, status: {CONVERGED}")
        plt.legend()
        plt.savefig(f"{self.QA_path}/{obs_name}_recovered_angles.png")
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(used_simgas, recovered_scales, label="Recovered Scales")
        plt.plot(used_simgas, scale_control, label="Scale Control")
        plt.xlabel("Iteration")
        plt.ylabel("Scale")
        plt.title(f"Recovered Scales for {obs_name}, status: {CONVERGED}")
        plt.legend()
        plt.savefig(f"{self.QA_path}/{obs_name}_recovered_scales.png")
        plt.close()



        return (
            (tx, ty),
            rot,
            (tx_err, ty_err),
            rot_err,
            success,
        )
        