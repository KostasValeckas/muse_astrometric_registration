.. _getting_started:

Getting started
===============

A run of the pipeline needs three things:

#. A directory with the MUSE images to align (``IMAGE_FOV_...fits``).
#. A reference image (FITS, with a WCS).
#. A :ref:`configuration file <conf>` that lists the exposures per pointing.
   It can be made with ``dev/sorter.py``.

With these, the recommended way to run is the :ref:`hybrid run <hybrid>`:

.. code-block:: bash

    python src/hybrid_pipeline.py NGC1365.json ../reference/NGC1365_reference.fits \
        --input_dir ../MUSE/ --output_dir NGC1365_HYBRID --plot_headless

or, with a single alignment method, the :ref:`single method run <run>`:

.. code-block:: bash

    python src/run.py NGC1365.json ../reference/NGC1365_reference.fits \
        --input_dir ../MUSE/ --output_dir NGC1365_FFT --method FFT

The results are written to the output directory, see :ref:`output <output>`.

An overview of all the procedures and their products is given in
:ref:`structure <structure>`.

TODO: a worked example with a small test dataset.
