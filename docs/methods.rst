.. _methods:

Alignment methods
=================

The method is chosen with ``--method`` (:ref:`single method run <run>`) or
``--primary`` / ``--secondary`` / ``--tertiary`` (:ref:`hybrid run <hybrid>`).

NoAlignment
-----------

Does nothing, for testing the plumbing (with the :ref:`angle scan <angle_scan>`
it only scans the rotation).

PointSources
------------

Matches point sources with ``astroalign`` and fits a rotation + translation to the
matched pairs (least-squares errors). astroalign only returns the sources of its
matched triangles (from at most its 50 brightest detections), so its transform is
used to re-match all sources detected in both frames (3 sigma, min. area 5 px,
unambiguous mutual nearest neighbours within 3 px); the astroalign matches are kept
if that finds fewer.

Every alignment gets a QA plot ``point_sources_*.png``: the numbered matched
sources in both frames (accepted / rejected), the measured displacements, the fit
residuals in the frame and in the (dx, dy) plane with the threshold, and a
histogram of the residual lengths against the expected Rayleigh distribution.

Outliers are rejected with an adaptive threshold (see :ref:`outliers`), limits
[0.1, 1.5] px. Matched centroids on real MUSE data scatter by 0.2-0.8 px per axis
and mismatches are off by 1.5-5 px. The first fit uses a robust (soft-L1) loss, so
mismatches do not set the first threshold. An alignment needs at least 5 accepted
sources to count as successful; if astroalign finds fewer matches, its settings are
loosened step by step before giving up.

OpticalFlow
-----------

``spacepylot`` computes the dense TV-L1 flow field between the frames, and the rigid
transform is fitted to a grid of flow vectors in this repository (not with
``spacepylot``'s own homography step, which refers the translation to the corner of
its cropped frame and whose translation-only fit is broken with ``skimage`` >= 0.26).
Errors come from a spatial block bootstrap of the flow vectors (5x5 tiles). If the
inliers lie in too few tiles for the bootstrap, a conservative fallback is used
(residual scatter of the flow vectors over sqrt(number of tiles)), never 0.

Every alignment gets a flow field QA plot ``flow_field_*.png``: the measured flow,
the fitted rigid model, the residual flow and the residual vectors.

Because TV-L1 can fail silently (flow frozen at zero, which is perfectly smooth and
passes the forward/reverse consistency check):

* The filtered frames are scaled to unit standard deviation before TV-L1. Without
  this the flow of smooth fields stays at zero (a 3.6 px shift came out as 0.02 px).
* Control-shift check: the input aligned with the result is shifted by a known
  (1.7, -2.3) px, and the flow must recover that shift within 0.3 px. If not, the
  alignment is set as failed (only ``failed_aligment_*.png``); the log states the
  recovered error.

Outliers are rejected with an adaptive threshold (see :ref:`outliers`), limits
[0.05, 0.5] px (TV-L1 residuals scatter by ~0.07 px).

FFT
---

Phase correlation on band-passed (difference of Gaussians) frames, over a range of
filter scales (sigmas), with a known control offset as a self-check. A translation
is accepted when several consecutive sigmas agree, the control offset is recovered
(``control_tol``, 0.05 px), and the peak has at least 90% of the control peak SNR
and a similar shape (elliptical Gaussian fits, shown in the
``translation_peak_*.png`` QA plots). With an initial guess, the peak is only
searched within +/- 1 px of it.

.. _outliers:

Outlier rejection
-----------------

Both ``PointSources`` and ``OpticalFlow`` reject outliers iteratively with an
adaptive threshold on the length of the residual vector: 3 robust sigmas of the
residual scatter (sigma = median residual length / 1.1774, the Rayleigh median, so
robust to up to 50% outliers), clipped to a lower and upper limit. Residuals of all
points are recomputed in every pass, so a rejected point can come back.

TODO: example QA plots for every method (``pictures/``).
