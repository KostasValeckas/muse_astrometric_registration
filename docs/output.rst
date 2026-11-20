.. _output:

Output
======

Every run (and every method directory of a :ref:`hybrid run <hybrid>`) contains:

Output table
------------

``{method}_alignment_output_table.fits`` - one row per exposure, in the order of
the :ref:`configuration file <conf>`:

.. code-block:: text

    DATE_OBS, MJD_OBS, TPL_START, IEXPO_OBS, POINTING_OBS,
    RA_OFFSET, RA_OFFSET_ERROR, DEC_OFFSET, DEC_OFFSET_ERROR, ROTANGLE, ROTATION_ERROR

Offsets and their errors are in degrees of RA and Dec; the pixel errors are
propagated through the WCS Jacobian, so they are always positive and correct for any
position angle. Failed and inconsistent alignments have null values. With
``--scale-ref`` there are two more columns, ``FLUX_SCALE`` and ``BACKGROUND`` (see
:ref:`reference flux scaling <scale_ref>`; NaN without a result; in the hybrid table
taken from the same method as the offsets).

The rotation follows the :ref:`rotation convention <rotation_convention>`.

Dither tables
-------------

``{x,y,rotation}[_error]_dither_table_{pointing}_{method}.csv`` and
``success_dither_table_{pointing}_{method}.csv`` - the dither-to-dither tables
(``table[a, b]`` is the alignment of dither a -> dither b). These are also what the
hybrid run reads for the initial guesses. See :ref:`dither alignment <dithers>`.

QA plots
--------

``aligment_QA/dithers``, ``aligment_QA/science`` and ``aligment_QA/combined`` - the
QA plots: difference images (also for failed, mirrored and composed alignments),
angle scans, FFT peak and progression plots, optical flow fields, point source
matches, and a one-page report per pointing. A failed alignment only gets its failed
difference image (``failed_aligment_*.png``): the other QA plots of an alignment are
held back until it succeeds.

TODO: example QA plots (``pictures/``).

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
