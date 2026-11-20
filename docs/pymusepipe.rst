.. _pymusepipe:

Using the results in pymusepipe
===============================

Quickstart
----------

.. code-block:: bash

    python dev/convert_hybrid_table.py results/NGC0628/Hybrid_alignment_output_table.fits \
        --dataset_obs --negate_ra --negate_dec --negate_rot

``dev/convert_hybrid_table.py`` makes a copy of an output table for pymusepipe
(default name ``<input name>_pymusepipe.fits``; the input table is not changed):

* rows without a result are removed (non-finite ``RA_OFFSET``, ``DEC_OFFSET`` or
  ``ROTANGLE``, or ``FLUX_SCALE`` / ``BACKGROUND`` if the table has them),
* ``--dataset_obs`` renames ``POINTING_OBS`` to ``DATASET_OBS``,
* ``--negate_ra``, ``--negate_dec`` and ``--negate_rot`` negate ``RA_OFFSET``,
  ``DEC_OFFSET`` and ``ROTANGLE``, each on its own.

What was done is written to the header of the new table (``NEG_RA``, ``NEG_DEC``,
``NEG_ROT``, ``DATASET``, ``NDROPPED``).

Columns used by pymusepipe
--------------------------

.. list-table::
    :header-rows: 1

    * - Column
      - Use in pymusepipe
    * - ``MJD_OBS``
      - key to match the exposure (exact match)
    * - ``DATE_OBS``
      - checked against the exposure
    * - ``TPL_START``, ``IEXPO_OBS``
      - name of the pixel table
    * - ``DATASET_OBS``
      - dataset of the exposure
    * - ``RA_OFFSET``, ``DEC_OFFSET``
      - offsets in degrees, applied to the WCS
    * - ``ROTANGLE``
      - added to the instrument position angle
    * - ``FLUX_SCALE``
      - passed to the ESO MUSE pipeline, multiplies the exposure
    * - ``BACKGROUND``
      - rescales the sky continuum used for the sky subtraction

.. warning::
    The signs of ``RA_OFFSET`` / ``DEC_OFFSET`` and ``ROTANGLE`` in the pymusepipe
    convention are not yet verified on data. TODO: document the verified signs.
