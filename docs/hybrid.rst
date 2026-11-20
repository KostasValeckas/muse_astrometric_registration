.. _hybrid:

Hybrid run
==========

Quickstart
----------

.. code-block:: bash

    python src/hybrid_pipeline.py NGC1365.json ../reference/NGC1365_reference.fits \
        --input_dir ../MUSE/ --output_dir NGC1365_HYBRID --plot_headless

How the methods are combined
----------------------------

#. The secondary (default ``OpticalFlow``) and tertiary (default ``PointSources``)
   methods run first, as isolated runs of the pipeline, without initial guesses and
   with their direct rotation estimators.
#. The primary method (default ``FFT``) runs with initial guesses read from the
   results on disk: from the secondary method if it has a result, else from the
   tertiary, else without a guess. The :ref:`angle scan <angle_scan>` is on by
   default for the primary.
#. The final table takes, for every exposure, the primary result if there is one,
   else the secondary, else the tertiary, else null, and adds a column
   ``ALIGNMENT_METHOD`` with the method that was used.
#. The difference images of exactly the results in the final table are copied to
   ``final_difference_images/``.

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
