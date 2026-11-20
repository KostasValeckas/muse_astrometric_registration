.. _index:

muse_astrometric_registration's documentation page
===================================================

muse_astrometric_registration is a Python pipeline for astrometric registration
of astronomical images. It is currently configured for MUSE frames: every
exposure is aligned to a reference image, and the result is an output table
with RA / Dec offsets, rotation angles (and optionally flux normalisations) per
exposure, together with quality-assessment (QA) plots for every alignment.

The code is developed under the ESO SSDF1-26C project.

.. note::
    The software is currently in an **open-beta state** - functionality is not
    fully tested.

**Principles behind the software:**

#. **Transparency.** *Every alignment produces QA plots and a log entry, so a
   result can always be checked by eye.* (TODO)
#. **Robustness.** *An alignment is checked in both directions, and failed
   alignments are recovered through the other dithers of a pointing where
   possible.* (TODO)

-----------------------------------------------

In this documentation, we provide guidance on installation and usage of the software.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   getting_started
   structure
   develop
   contact
