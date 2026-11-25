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

A wrong unit of the reference is caught before any alignment: FLUX_SCALE is first
fitted for every pointing, and a known WFI header error is fixed automatically, see
:ref:`known limitations <known_limitations_units>`.

After the alignment: flux normalisation
---------------------------------------

For every exposure with a result, ``reference = FLUX_SCALE * (exposure +
BACKGROUND)`` is fitted with the exposure shifted by its final offsets (medians of
10 x 10 pixel chunks of the median filtered images, orthogonal distance regression
with the chunk MAD standard deviations as errors), and written to the :ref:`output table <output>`
as ``FLUX_SCALE`` and ``BACKGROUND``. These have the same meaning as pymusepipe's
OFFSET table columns.

Exposures without a result, or with a fit that is not usable (fewer than 10 chunks
with signal), get NaN. pymusepipe fits with ``scipy.odr`` (removed from SciPy
1.19); here the same problem is solved with ``scipy.optimize.least_squares``, which
gives the same values to the precision of the fit. The :ref:`dither alignments <dithers>` are not
affected.
