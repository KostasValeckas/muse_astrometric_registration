.. _angle_scan:

Angle scan
==========

Enabled with ``--scan_for_angles`` (:ref:`single method run <run>`), and on by
default for the primary method of the :ref:`hybrid run <hybrid>`.

Instead of each method's own rotation estimator, the rotation can be found by
minimising the difference between the reference and the rotated + shifted input: the mean absolute difference of the blurred frames over a fixed central
window.

Before the difference, the input is put on the flux scale of the reference at every
angle: ``reference = a * input + b`` is fitted (least squares) over the window, and
the metric is the mean absolute residual of that fit, divided by the median of the
reference (the same for every angle). An earlier version divided each frame by the
median of its whole frame instead; that does not match the flux scales, leaves a
scaled copy of the galaxy in the difference and could pull the scan 0.2 - 0.9
degrees off. It is still available as ``scan_flux_match = "median"`` (parameter of
the methods), for comparisons.

The window is clipped so that it stays covered by the input for the translation.
It leaves room for at least ``scan_max_shift`` (10 px), or for the expected
translation + ``scan_shift_slack`` (3 px) if that is larger. The expected
translation is each method's translation-only fit at 0 degrees. If that fit fails, the initial guess is used, and without one, the
first translation measured in the scan.

The scan itself:

#. A ternary search over [-1, 1] degrees finds the rough minimum. It reports failure
   if it ends at the edge of the range or if all evaluations give the same value
   (e.g. every translation estimate failed).
#. A fine scan around it (+/- 0.15 degrees, 25 points)
   estimates the translation at every angle.
#. A linear trend is fitted to tx(angle) and ty(angle), translation outliers are
   rejected, and all angles are re-scored with the smooth trend translation, so the
   jitter of the individual estimates does not leak into the metric.
#. Both a parabola and a hyperbola, ``sqrt(a (x - x0)^2 + b) + c``, are fitted to
   the fine scan with a 2-pass outlier rejection, and the one with the lower AICc
   (compared on the same points) is used. The metric is a rounded V: when the fine
   scan reaches the straight arms the parabola is biased and the hyperbola wins,
   when it only sees the rounded bottom the hyperbola degenerates and the parabola
   wins. Fits are only thrown away if their shape makes no sense: no minimum,
   minimum outside the search range or extrapolated more than one fine scan width
   beyond the scan, or a curve that is flat within the noise. The minimum may lie
   outside the fine scan. The angle error comes from a block bootstrap of the fit
   residuals.

The QA plot ``angle_scan_model_*.png`` shows the scan, the fit and its residuals,
and the translation per angle with its trend and residuals:

.. figure:: pictures/angle_scan_model_example.png
    :width: 100%
    :alt: Angle scan QA plot for one exposure of NGC1672

    Angle scan QA plot (NGC1672, pointing 1, dither 1 to the reference, FFT).
    **Left:** the whole scan, with the ternary search points (grey) and the
    fine scan around the rough minimum. **Middle:** zoom on the fine scan,
    scored with the smooth translation trend. Cyan points are used in the
    fit, red crosses are rejected outliers, the green curve is the chosen
    model (here the hyperbola) and the dashed line is the estimated minimum
    with its error. The panel below shows the fit residuals. **Right:** the
    translation tx (blue) and ty (orange) measured at every fine scan angle,
    with the linear trend (white) and the chosen translation at the final
    angle (stars). The panel below shows the residuals to the trend.

FFT with the angle scan
-----------------------

FFT, as implemented in the pipeline, is considerably slower than other methods, and 
the many iterations using the angle scan quickly become costly. Therefore, special approach 
is taken when using FFT with the angle scan to optimize it:

* The direct fit (rotation + translation) is run first, as a cheap check. If its
  translation fails, the alignment fails without running the slow scan.
* If only its rotation estimate fails (no valid entry in the per-sigma rotation
  table), the shift may still be measurable, and the scan does not need the
  direct rotation. With ``scan_on_rotation_failure`` (default on), a
  translation-only fit at 0 degrees is tried:

  * if it fails, the alignment fails without the scan;
  * if it works, the scan is run with the full sigma range and that translation
    as the expected one. There is no direct fit to fall back on, so if the scan
    fails, the alignment fails.

  With the option off, any failure of the direct fit fails the alignment.
* If the direct fit agrees with the initial guess (translation within 1 px,
  rotation within 0.1 degrees), the scan only searches the sigmas around the ones
  the direct fit agreed on (+/- 5 grid steps), which makes every scanned angle much
  cheaper. If this narrowed scan fails, it is redone once with the full sigma range.
* If the scan fails after a successful direct fit (e.g. the ternary search ends at
  the boundary of the angle range), the direct fit is returned instead
  (``fallback_to_direct_fit`` flag); the round-trip and dither checks still apply.
* If no translation is found, the log lists the acceptance criteria every sigma
  failed (direct and final estimates only, not every scanned angle).
* The final translation is the trend at the final angle (error: rms scatter of the
  individual translations around the trend), marked with stars in the angle-scan QA
  plot. No extra translation estimate is made at the end.

.. warning::
    **Known limitation:** the reported angle error only covers the scatter of the
    scan points around the fit. The noise in the images is the same at every scanned
    angle, so it shifts the whole curve, and this is not included yet (on synthetic
    frames the minimum moves by ~0.01 degrees between noise realisations, much more
    than the reported error).
    See also :ref:`known limitations <known_limitations>`.
