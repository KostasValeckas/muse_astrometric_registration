.. _getting_started:

Getting started
===============

.. tip::
    The easiest way to get started, and to test the pipeline on your own machine,
    is the :ref:`worked example <worked_example>`. It runs on a small dataset that
    is shipped with the repository (NGC1365, one pointing with 4 dithers), so
    nothing has to be downloaded or prepared. From the repository root:

    .. code-block:: bash

        python src/hybrid_pipeline.py dev/NGC1365_short.json \
            dev/NGC1365_short/NGC1365_Rc_flux_nosky_cutout_P01.fits \
            --input_dir dev/NGC1365_short --output_dir NGC1365_short_HYBRID --plot_headless

    The :ref:`worked example <worked_example>` page walks through the outputs of
    this run.

For your own data, a run of the pipeline needs three things:

#. A directory with the MUSE images to align (``IMAGE_FOV_...fits``).
#. A reference image (FITS, with calibrated WCS).
#. A :ref:`configuration file <conf>` that lists the exposures per pointing.
   It can be made with ``dev/sorter.py``.

With these, the recommended way to run is the :ref:`hybrid run <hybrid>`:

.. code-block:: bash

    python src/hybrid_pipeline.py NGC1365.json ../reference/NGC1365_reference.fits \
        --input_dir ../MUSE/ --output_dir NGC1365_HYBRID --plot_headless

or, with a single :ref:`alignment method <methods>`, the :ref:`single method run <run>`:

.. code-block:: bash

    python src/run.py NGC1365.json ../reference/NGC1365_reference.fits \
        --input_dir ../MUSE/ --output_dir NGC1365_FFT --method FFT

The results are written to the output directory, see :ref:`output <output>`.

An overview of all the procedures and their products is given in
:ref:`structure <structure>`.
