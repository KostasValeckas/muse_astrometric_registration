.. _known_limitations:

Known limitations
=================

The software is in open beta. These are the limitations we know of so far.

.. _known_limitations_errors:

Error estimation
----------------

All error estimation in the pipeline has been implemented with simplifying assumptions, 
and not tested properly on large scale datasets for correctness. Do not use 
these error estimations for scientific analysis.

Rotation angles
---------------

* **The direct angle fits do not agree to high sub-degree precision.** The rotation
  estimators of the individual methods (:ref:`FFT <method_fft>`,
  :ref:`OpticalFlow <method_opticalflow>`, :ref:`PointSources <method_pointsources>`,
  without the :ref:`angle scan <angle_scan>`) can give rotations that differ by ~0.1 degrees
  for the same exposure, much more than their reported errors. In the
  :ref:`worked example <worked_example>`, the rotations of the same dither differ
  by up to ~0.14 degrees between the methods.

* **The angle scan is the most stable approach so far, but not infallible.** For
  this reason it is on by default for the primary method of the
  :ref:`hybrid run <hybrid>`. Check the angle scan QA plots (``angle_scan_model_*.png``) and
  the :ref:`final difference images <final_difference_images>`, especially for exposures with few features or
  strong artifacts.

* **The fix of the angle scan metric is not yet tested on a large dataset.** An
  earlier version of the scan divided the reference and the input each by the
  median of its whole frame. That does not put the two frames on the same flux
  scale, so even at the correct alignment a scaled copy of the galaxy (~5-10 %)
  stayed in the difference, and the scan could be pulled to a wrong angle (it
  reaches the wrong alignment by rotating, since the FFT translation changes with
  the trial angle when the bright structure is off the frame centre). In the big
  test run (18 galaxies, 300 exposures with results from all three methods), the
  FFT result of 19 exposures was more than 0.5 px (at the consistency check point)
  from both ``OpticalFlow`` and ``PointSources``, which agreed with each other; in
  all of them the FFT direct fit agreed with the other methods within 0.1 degrees,
  and the scan moved the angle away by 0.2-0.9 degrees.

  The scan now fits the flux scale of the input to the reference at every angle
  (see :ref:`angle scan <angle_scan>`). On the case that was investigated
  (NGC3596, pointing 2, dither 3) the scan angle moved from ~0.7 to ~0.2 degrees,
  in agreement with the direct fit and the other methods, and the
  :ref:`worked example <worked_example>` changed by less than 0.005 degrees, but
  the big test run has not been repeated. Until then, compare the angle of the
  scan with the direct fit in the log ("Direct FFT fit for ... succeeded
  (rotation ... degrees)") and with the other methods. An error shared by both
  directions is not always caught by the forward / reverse consistency check.

* **The reported angle error of the scan is too small.** It only covers the
  scatter of the scan points around the fit. The noise in the images is the same
  at every scanned angle, so it shifts the whole curve, and this is not included
  yet: on synthetic frames the minimum moves by ~0.01 degrees between noise
  realisations, much more than the reported error. In the worked example, storing
  the reference as float32 instead of float64 alone moved the FFT angle by
  ~0.001 degrees, several times its reported error.

Optical flow outlier rejection
------------------------------

* **OpticalFlow rejects most of its flow vectors on real data.** The adaptive
  :ref:`outlier threshold <outliers>` is 3 robust sigmas of the residuals,
  clipped to an upper limit of 0.5 px (``residual_threshold``), which was tuned on synthetic frames, where
  the TV-L1 residuals scatter by ~0.07 px. On real MUSE data they scatter by
  ~0.26 px, so the upper limit sets the threshold in ~95% of the fits, and the
  rejected vectors are part of one broad residual distribution rather than a
  separate population of outliers. Over ~7400 fits (18 galaxies), the median
  fraction of vectors kept is 26% (22% for science alignments, 30% for dither
  alignments), and 6% in the worst 5% of the fits. The translations still agree
  with the other methods within ~0.2 px, but the bootstrap errors of OpticalFlow,
  which only resample the kept vectors, are likely too small. A higher upper limit
  has not been tested yet.

.. _known_limitations_units:

Reference units (``--scale-ref``)
---------------------------------

* **Wrong units in the reference header.** Some WFI references
  (``*_Rc_flux_nosky.fits``) have ``BUNIT = 'mJy'`` while their data are in
  microJansky (the same header says ``ZUNIT = 'micro Jansky'``): they give
  FLUX_SCALE ~ 1000. A wrong unit makes FLUX_SCALE off by the same factor
  (BACKGROUND, in the unit of the exposure, is not). The alignment itself is not
  affected: all methods are insensitive to a constant factor (the bright-source
  filter of ``PointSources`` is relative to each frame).

  To catch this, FLUX_SCALE is first fitted for every pointing before any alignment
  (dither 1, as reprojected with its WCS):

  #. With the unit of the header (or ``--ref_unit``): used if FLUX_SCALE is within
     a factor 10 of 1.
  #. If not, the known WFI case is fixed: only if the unit came from ``BUNIT``,
     ``BUNIT`` is mJy and ``ZUNIT`` is microJansky, the reference is converted as
     microJansky, and used if FLUX_SCALE is then within a factor 10 of 1 (warning
     in the log). This is decided at the first pointing and used for the whole run.
  #. Otherwise everything about the conversion is logged as an error and the run
     stops (the hybrid run as a whole): it does not run with a wrong unit.

  A run whose median FLUX_SCALE after the alignment is more than a factor 10 from 1
  ends with a warning in the log. If the ``BUNIT`` is wrong in another way, set the
  unit with ``--ref_unit``.

* **Unit strings are read literally.** ``BUNIT = 'A'`` is Ampere, not Angstrom
  (``Angstrom``), so it cannot be converted and stops the run, as do counts and
  surface brightness units. Use ``--ref_unit`` to override it.
