.. _run:

Single method run
=================

This mode uses the pipeline structure to perform the registration with one
:ref:`alignment method <methods>` only. It is meant as a supplement to the
:ref:`hybrid run <hybrid>`, for when only one method is wanted, for scientific or
debugging reasons.

Quickstart
----------

.. code-block:: bash

    python src/run.py configuration_file reference_file --input_dir path_to_directory_with_data \
        --output_dir path_to_directory_where_QA_plots_and_results_will_be_stored

For example:

.. code-block:: bash

    python src/run.py NGC0253.json NGC0253_Rc_flux_wcomb.fits --input_dir /NGC0253/MUSE/ \
        --output_dir NGC0253_TEST --method FFT

Flags
-----

.. code-block:: text

      --method METHOD       NoAlignment (default), PointSources, OpticalFlow, FFT, or All
                            (runs PointSources, OpticalFlow and FFT one after the other,
                            mostly for testing).
      --plot_headless       Generate plots without a GUI (use this on servers).
      --distance_threshold DISTANCE_THRESHOLD
                            Round-trip distance (pixels) for the forward/reverse
                            consistency checks. Default is 0.10.
      --skip_rotation_estimation
                            Fix the rotation at zero, only the translation is fitted.
      --scan_for_angles     Estimate the rotation with the shared angle scan instead of
                            each method's own direct rotation estimator.
      --dither_mode {full,targeted}
                            'full' (default) aligns all dither pairs, 'targeted' only the
                            pairs needed to recover failed science alignments.
      --scale-ref           Convert the reference to MUSE units and fit FLUX_SCALE and
                            BACKGROUND (see the flux scaling page).
      --ref_filter REF_FILTER
                            Filter of the reference for the unit conversion. Default is WFI_BB.
      --ref_unit REF_UNIT   Unit of the reference (e.g. uJy), used instead of its BUNIT.

See :ref:`alignment methods <methods>`, :ref:`angle scan <angle_scan>`,
:ref:`reference flux scaling <scale_ref>` and :ref:`dither alignment <dithers>`.
