.. _scale_ref:

Reference flux scaling
======================

Enabled with ``--scale-ref`` (``run.py`` and ``hybrid_pipeline.py``).

The reference image can be in different units, and on a different flux scale and
background, than the MUSE images. With ``--scale-ref`` the method of pymusepipe is
used (borrowed in ``src/flux.py``, MIT license).

Before the alignment: unit conversion
-------------------------------------

#. When the reference is read, it is converted from its unit to the MUSE unit
   (1e-20 erg/s/cm2/A) with the spectral density equivalency at the pivot
   wavelength of ``--ref_filter`` (default ``WFI_BB``, 6483.58 A; the list is
   ``REFERENCE_FILTERS`` in ``flux.py``, the same as pymusepipe's
   ``dict_equivalencies``). The unit is read from the ``BUNIT`` keyword of the
   reference; if there is none, the unit of the filter is assumed (1 microJansky
   for WFI_BB), with a warning. A ``BUNIT`` that cannot be converted (e.g. counts,
   a surface brightness, or ``A`` - which is Ampere, Angstrom is ``Angstrom``)
   stops the run before any alignment. If the ``BUNIT`` is wrong, set the
   unit with ``--ref_unit`` (e.g. ``--ref_unit uJy``), which is used instead.
#. After the reprojection to the frame of a pointing, it is multiplied by the pixel
   area ratio (MUSE / reference).

.. warning::
    Some WFI references have ``BUNIT = 'mJy'`` while their data are in
    microJansky: they give FLUX_SCALE ~ 1000. The alignments are not affected by
    a wrong unit (all methods are insensitive to a constant factor), and neither
    is BACKGROUND (it is in the unit of the exposure), only FLUX_SCALE. A run whose
    median FLUX_SCALE is more than a factor 10 from 1 ends with a warning in the
    log.

After the alignment: flux normalisation
---------------------------------------

For every exposure with a result, ``reference = FLUX_SCALE * (exposure +
BACKGROUND)`` is fitted with the exposure shifted by its final offsets (medians of
10 x 10 pixel chunks of the median filtered images, orthogonal distance regression
with the chunk MAD standard deviations as errors), and written to the output table
as ``FLUX_SCALE`` and ``BACKGROUND``. These have the same meaning as pymusepipe's
OFFSET table columns (see :ref:`pymusepipe <pymusepipe>`).

Exposures without a result, or with a fit that is not usable (fewer than 10 chunks
with signal), get NaN. pymusepipe fits with ``scipy.odr`` (removed from SciPy
1.19); here the same problem is solved with ``scipy.optimize.least_squares``, which
gives the same values to the precision of the fit. The dither alignments are not
affected.

Final difference images
-----------------------

Every reported result gets a final difference image,
``aligment_QA/science/difference_image_{pointing}_dither_{i}_final_{method}.png``,
always with the exposure shifted onto the reference by the final offsets (whether
they were measured forward, are the inverse of the reverse measurement, or were
composed via other dithers). These are the images copied to
``final_difference_images/`` in the hybrid run.

With ``--scale-ref`` the reference is put on the flux scale of the exposure
(``reference / FLUX_SCALE - BACKGROUND``) and the difference is shown in MUSE
units, with a colour scale symmetric around 0, so a remaining flux or background
difference is visible; without it, both images are normalised to their maximum as
in the other QA plots. The difference images made during the measurements are kept
as they are.
