.. _rotation_convention:

Rotation convention
===================

All rotations in the pipeline (tables, logs, QA plots) follow one convention:
degrees, positive counter-clockwise as the frame is displayed with origin upper,
i.e. as the numpy array is read by ``astropy.io.fits`` (the
``scipy.ndimage.rotate`` convention). The reported values are the correction to
apply to the input frame: rotate by the angle about the frame centre, then shift by
the offsets.

.. note::
    In DS9 (origin lower) the same correction looks clockwise.

All dithers and the reference are reprojected onto the WCS of the first dither of a
pointing, so all alignments are measured on that pixel grid. Since every frame is
reprojected with its own WCS first, the measured values are the residual
corrections of the WCS, not the raw rotations.
