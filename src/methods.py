from abc import ABC, abstractmethod


class AligmentMethod(ABC):
    """
    Abstract base class for all alignment methods.

    This class defines the interface that all alignment methods must implement.
    """

    def __init__(self, name, version, logger=None):
        """
        Initialize the alignment method.

        Parameters
        ----------
        name : str
            Name of the alignment method.

        version :
            Version of the alignment method.

        logger: Optional
            Logger object for logging messages.
        """

        self.name = name
        self.version = version
        self.logger = logger

    @abstractmethod
    def run(
        self,
        reference_frame,
        input_frame,
        initial_guess_translation=(None, None),
        initial_guess_rotation=None,
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
        super().__init__("PointSources", "0.0.0", logger = logger)


    