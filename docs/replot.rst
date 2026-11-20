.. _replot:

Replot final difference images
==============================

Quickstart
----------

.. code-block:: bash

    python dev/replot_final_difference_images.py OUTPUT_DIR CONFIG_FILE

Replots the final difference images of a finished run with the
:ref:`flux normalisation <scale_ref>`, without running the alignment again: the
offsets are read from the output table, ``FLUX_SCALE`` and ``BACKGROUND`` are fitted
for every exposure, and the difference images are written to
``<output_dir>/final_difference_images_flux``, together with
``{table}_alignment_output_table_flux.fits`` and a log.

Flags
-----

.. code-block:: text

      --input_dir           Default: from run_manifest.json of the run.
      --ref_file_path       Default: from run_manifest.json of the run.
      --table               Output table to read the offsets from. Default is Hybrid.
      --ref_filter          Default is WFI_BB.
      --ref_unit            Unit of the reference (e.g. uJy), used instead of its BUNIT.
      --images_dir          Default is <output_dir>/final_difference_images_flux.
