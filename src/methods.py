from abc import ABC, abstractmethod
import os
import astroalign as aa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import os
from scipy.optimize import least_squares
from scipy.ndimage import shift, rotate

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

    def check_input(self, reference_frame, input_frame, skip_shape_check=False, skip_type_check=False, skip_nan_check=False):
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

        if not skip_nan_check and (np.isnan(reference_frame).any() or np.isnan(input_frame).any()):
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

        # Shift the image
        shifted_image = shift(image, shift=(dy, dx), mode="nearest")

        # Rotate the image around its center
        rotated_image = rotate(
            shifted_image, angle=dtheta, reshape=False, mode="nearest"
        )

        return rotated_image

    def plot_difference_image(
        self, reference_frame, input_frame, dx, dy, dtheta, obs_name, show=False
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
        """

        # normalize both images to unity

        reference_frame = reference_frame / np.nanmax(reference_frame)
        input_frame = input_frame / np.nanmax(input_frame)

        aligned_frame = self.shift_image(input_frame, dx, dy, dtheta)

        diff_prior = reference_frame - input_frame
        diff_post = reference_frame - aligned_frame

        _, axes = plt.subplots(2, 2, figsize=(12, 12))

        ref_vmin, ref_vmax = np.percentile(reference_frame, 5), np.percentile(reference_frame, 95)
        in_vmin, in_vmax = np.percentile(input_frame, 5), np.percentile(input_frame, 95)

        axes[0, 0].imshow(reference_frame, cmap="Reds", vmin=ref_vmin, vmax=ref_vmax, alpha=0.5)
        axes[0, 0].imshow(input_frame, cmap="Blues", vmin=in_vmin, vmax=in_vmax, alpha=0.5)
        axes[0, 0].set_title("Before Alignment")
        axes[0, 0].set_xlabel("X Pixel")
        axes[0, 0].set_ylabel("Y Pixel")

        axes[0, 1].imshow(reference_frame, cmap="Reds", vmin=ref_vmin, vmax=ref_vmax, alpha=0.5)
        axes[0, 1].imshow(aligned_frame, cmap="Blues", vmin=in_vmin, vmax=in_vmax, alpha=0.5)
        axes[0, 1].set_title("After Alignment")
        axes[0, 1].set_xlabel("X Pixel")
        axes[0, 1].set_ylabel("Y Pixel")

        axes[1, 0].imshow(
            diff_prior,
            cmap="gray",
            vmin=np.percentile(diff_prior, 5),
            vmax=np.percentile(diff_prior, 95),
        )
        axes[1, 0].set_title("Difference Before Alignment")
        axes[1, 0].set_xlabel("X Pixel")
        axes[1, 0].set_ylabel("Y Pixel")

        axes[1, 1].imshow(
            diff_post,
            cmap="gray",
            vmin=np.percentile(diff_post, 5),
            vmax=np.percentile(diff_post, 95),
        )
        axes[1, 1].set_title("Difference After Alignment")
        axes[1, 1].set_xlabel("X Pixel")
        axes[1, 1].set_ylabel("Y Pixel")

        for ax in axes.ravel():
            ax.set_aspect("equal")

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


class PointSources(AligmentMethod):

    def __init__(self, logger=None, outlier_threshold=0.5, max_iterations=5, QA_path=None):

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

    def run(
        self,
        reference_frame,
        input_frame,
        obs_name,
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
        # ------------------------------------------------------------------

        try:
            _, (source_list, target_list) = aa.find_transform(
                input_frame,
                reference_frame,
            )
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to find matching sources: {e}")

            if save_QA:
                fig = plt.figure(figsize=(10, 5))

                ax1 = plt.subplot(1, 2, 1)
                ax1.imshow(reference_frame, cmap="gray", vmin=np.percentile(reference_frame, 5), vmax=np.percentile(reference_frame, 95))
                ax1.set_title("Reference Frame")

                ax2 = plt.subplot(1, 2, 2)
                ax2.imshow(input_frame, cmap="gray", vmin =np.percentile(input_frame, 5), vmax=np.percentile(input_frame, 95))
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

            return self.FAILED_RETURN

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
                ax1.imshow(reference_frame, cmap="gray", vmin=np.percentile(reference_frame, 5), vmax=np.percentile(reference_frame, 95))
                ax1.set_title("Reference Frame")

                ax2 = plt.subplot(1, 2, 2)
                ax2.imshow(input_frame, cmap="gray", vmin=np.percentile(input_frame, 5), vmax=np.percentile(input_frame, 95))
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
            ax1.imshow(reference_frame, cmap="gray", vmin=np.percentile(reference_frame, 5), vmax=np.percentile(reference_frame, 95))
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
            ax2.imshow(input_frame, cmap="gray", vmin=np.percentile(input_frame, 5), vmax=np.percentile(input_frame, 95))
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
