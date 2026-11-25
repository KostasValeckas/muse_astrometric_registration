.. _hybrid:

Hybrid run
==========

The pipeline is designed to support several different :ref:`alignment methods <methods>`
to cover different edge cases in MUSE data. To make sure all of them are
exhausted, the pipeline can be run in hybrid mode, where all of the methods
are deployed and support each other with initial guesses and statistics.
The pipeline automatically produces one best-attempt registration per frame,
but also all the individual per-method alignments.

The hybrid pipeline is configured to run with three methods at once with all 
having a specific priority role in estiamting the final results, and what methods 
are used for what priority can be chosen by the user.

Quickstart
----------

.. code-block:: bash

    python src/hybrid_pipeline.py NGC1365.json ../reference/NGC1365_reference.fits \
        --input_dir ../MUSE/ --output_dir NGC1365_HYBRID --plot_headless

How the methods are combined
----------------------------

#. The secondary (default :ref:`OpticalFlow <method_opticalflow>`) and tertiary
   (default :ref:`PointSources <method_pointsources>`)
   methods run first, as isolated runs of the pipeline, without initial guesses and
   with their direct rotation estimators.
#. The primary method (default :ref:`FFT <method_fft>`) runs with initial guesses read from the
   results on disk: from the secondary method if it has a result, else from the
   tertiary, else without a guess. The :ref:`angle scan <angle_scan>` is on by
   default for the primary.
#. The final table takes, for every exposure, the primary result if there is one,
   else the secondary, else the tertiary, else null, and adds a column
   ``ALIGNMENT_METHOD`` with the method that was used.
#. The :ref:`final difference images <final_difference_images>` of exactly the
   results in the final table are copied to ``final_difference_images/``.

Flags
-----

.. code-block:: text

      --primary PRIMARY     Primary method. Default is FFT.
      --secondary SECONDARY
                            Secondary method (first source of initial guesses).
                            Default is OpticalFlow.
      --tertiary TERTIARY   Tertiary method (second source of initial guesses).
                            Default is PointSources.
                            (Options for all three: NoAlignment, PointSources, FFT,
                            OpticalFlow.)
      --plot_headless       Generate plots without a GUI.
      --distance_threshold DISTANCE_THRESHOLD
                            Round-trip distance (pixels) for the consistency checks.
                            Default is 0.10.
      --skip_rotation_estimation
                            Fix the rotation at zero for all methods.
      --no_primary_angle_scan
                            The primary method uses its direct rotation estimator
                            instead of the angle scan.
      --secondary_angle_scan
                            The secondary method uses the angle scan.
      --tertiary_angle_scan
                            The tertiary method uses the angle scan.
      --primary_dither_mode {full,targeted}
                            Default is 'targeted' (the primary is the slow method).
      --secondary_dither_mode {full,targeted}
                            Default is 'full'.
      --tertiary_dither_mode {full,targeted}
                            Default is 'full'.
      --scale-ref           See the flux scaling page.
      --ref_filter REF_FILTER
                            Default is WFI_BB.
      --ref_unit REF_UNIT   Unit of the reference (e.g. uJy), used instead of its BUNIT.

See :ref:`alignment methods <methods>`, :ref:`angle scan <angle_scan>`,
:ref:`dither alignment <dithers>` (dither modes, ``--distance_threshold``) and
:ref:`reference flux scaling <scale_ref>` (``--scale-ref``, ``--ref_filter``,
``--ref_unit``).

Output directory
----------------

.. code-block:: text

    NGC1365_HYBRID/
        secondary_OpticalFlow/          full output of the secondary run
        tertiary_PointSources/          full output of the tertiary run
        primary_FFT/                    full output of the primary run
        final_difference_images/        difference images of the reported results
        Hybrid_alignment_output_table.fits
        aligment_log_*.log

See :ref:`output <output>` for the contents of every method directory.
