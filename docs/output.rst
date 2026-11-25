.. _output:

Output
======

Every run (and every method directory of a :ref:`hybrid run <hybrid>`) contains an
output table, dither tables, QA plots, a log and a run manifest, described below.

Example
-------

The output of the :ref:`worked example <worked_example>` (a hybrid run of one
pointing with 4 dithers), as shown by ``tree -L 2``:

.. code-block:: text

    $ tree -L 2 NGC1365_short_HYBRID
    NGC1365_short_HYBRID
    ├── aligment_log_2026-11-25_16-00-09.log
    ├── final_difference_images
    │   ├── difference_image_1_dither_1_final_FFT.png
    │   ├── difference_image_1_dither_2_final_FFT.png
    │   ├── difference_image_1_dither_3_final_FFT.png
    │   └── difference_image_1_dither_4_final_FFT.png
    ├── Hybrid_alignment_output_table.fits
    ├── primary_FFT
    │   ├── aligment_log_2026-11-25_15-47-11.log
    │   ├── aligment_QA
    │   ├── FFT_alignment_output_table.fits
    │   ├── rotation_dither_table_1_FFT.csv
    │   ├── rotation_error_dither_table_1_FFT.csv
    │   ├── run_manifest.json
    │   ├── success_dither_table_1_FFT.csv
    │   ├── x_dither_table_1_FFT.csv
    │   ├── x_error_dither_table_1_FFT.csv
    │   ├── y_dither_table_1_FFT.csv
    │   └── y_error_dither_table_1_FFT.csv
    ├── secondary_OpticalFlow
    │   ├── aligment_log_2026-11-25_15-44-33.log
    │   ├── aligment_QA
    │   ├── OpticalFlow_alignment_output_table.fits
    │   ├── ...                         (same files as primary_FFT)
    └── tertiary_PointSources
        ├── aligment_log_2026-11-25_15-46-15.log
        ├── aligment_QA
        ├── PointSources_alignment_output_table.fits
        └── ...                         (same files as primary_FFT)

The QA plots of one alignment (here dither 2 to the reference) in the FFT
directory, as shown by ``tree -P`` (the ``dithers`` directory is empty: in
targeted dither mode no dither alignment was needed):

.. code-block:: text

    $ tree -P '*dither_2_to_reference*|*combined*' --prune NGC1365_short_HYBRID/primary_FFT
    NGC1365_short_HYBRID/primary_FFT
    └── aligment_QA
        ├── combined
        │   └── 1_combined_QA_FFT.png
        └── science
            ├── angle_scan_model_1_dither_2_to_reference_FFT.png
            ├── difference_image_1_dither_2_to_reference_FFT.png
            ├── eps_vs_sigma_1_dither_2_to_reference.png
            ├── translation_peak_1_dither_2_to_reference_FFT.png
            └── translation_progression_1_dither_2_to_reference_FFT.png

Every alignment is also run in the opposite direction
(``..._reference_to_dither_2_...``), and the result that is reported gets a
``difference_image_1_dither_2_final_FFT.png``.

Output table
------------

``{method}_alignment_output_table.fits`` - one row per exposure, in the order of
the :ref:`configuration file <conf>`:

.. code-block:: text

    DATE_OBS, MJD_OBS, TPL_START, IEXPO_OBS, POINTING_OBS,
    RA_OFFSET, RA_OFFSET_ERROR, DEC_OFFSET, DEC_OFFSET_ERROR, ROTANGLE, ROTATION_ERROR

Offsets and their errors are in degrees of RA and Dec; the pixel errors are
propagated through the WCS Jacobian, so they are always positive and correct for any
position angle (but see :ref:`known limitations <known_limitations_errors>` on the
error estimates). Failed and inconsistent alignments have null values. With
``--scale-ref`` there are two more columns, ``FLUX_SCALE`` and ``BACKGROUND`` (see
:ref:`reference flux scaling <scale_ref>`; NaN without a result; in the hybrid table
taken from the same method as the offsets).

.. _rotation_convention:

Rotation convention
^^^^^^^^^^^^^^^^^^^

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

Dither tables
-------------

``{x,y,rotation}[_error]_dither_table_{pointing}_{method}.csv`` and
``success_dither_table_{pointing}_{method}.csv`` - the dither-to-dither tables
(``table[a, b]`` is the alignment of dither a -> dither b; for the error tables see
:ref:`known limitations <known_limitations_errors>`). These are also what the
hybrid run reads for the initial guesses. See :ref:`dither alignment <dithers>`.

.. _final_difference_images:

Final difference images
-----------------------

Every reported result gets a final difference image,
``aligment_QA/science/difference_image_{pointing}_dither_{i}_final_{method}.png``,
always with the exposure shifted onto the reference by the final offsets (whether
they were measured forward, are the inverse of the reverse measurement, or were
composed via other dithers). These are the images copied to
``final_difference_images/`` in the hybrid run.

With ``--scale-ref`` (see :ref:`reference flux scaling <scale_ref>`) the reference is put on the flux scale of the exposure
(``reference / FLUX_SCALE - BACKGROUND``) and the difference is shown in MUSE
units, with a colour scale symmetric around 0, so a remaining flux or background
difference is visible; without it, both images are normalised to their maximum as
in the other QA plots. The difference images made during the measurements are kept
as they are.

QA plots
--------

``aligment_QA/dithers``, ``aligment_QA/science`` and ``aligment_QA/combined`` - the
QA plots: difference images (also for failed, mirrored and composed alignments),
angle scans, FFT peak and progression plots, optical flow fields, point source
matches, and a one-page report per pointing. A failed alignment only gets its failed
difference image (``failed_aligment_*.png``): the other QA plots of an alignment are
held back until it succeeds.

Example QA plots are shown for every :ref:`method <methods>`, the
:ref:`angle scan <angle_scan>`, :ref:`composed alignments and the per-pointing report
<dithers>`, and in the :ref:`worked example <worked_example>`.

Log
---

``aligment_log_*.log`` - the log of the run. The last line gives the total execution
time (for the hybrid run, the top-level log gives the time of all three runs
together). A crashed or interrupted run ends with "ABORTED" and the traceback
instead of "finished" (also when a batch system stops the job with SIGTERM, e.g. at
the SLURM time limit). The log starts with the code version (git commit, and a
warning if the code has local changes), host, package versions, command line, the
pipeline settings and the method parameters; every alignment logs its duration,
every pointing its duration and the peak memory, and the run ends with a summary.

Unexpected errors do not end a run: an alignment that raises is logged with the
traceback and set as failed, a pointing that raises (e.g. an unreadable file) is
written as failed, and in the hybrid run a method that crashes only loses its own
results. Search the logs for "Unexpected error" and check the summary lines at the
end.

Run manifest
------------

``run_manifest.json`` - the reference file and input directory of the run. A run
into an output directory that holds the output of a different reference file is
refused (use a new output directory, or delete the manifest if the old output is not
needed); a different input directory with the same reference only gives a warning.
