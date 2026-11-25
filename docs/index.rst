.. _index:

muse_astrometric_registration's documentation page
===================================================

.. note::
    The software is currently in an **open-beta state** - functionality is not
    fully tested.

About
-----

muse_astrometric_registration is a Python pipeline for astrometric registration
of astronomical images. It is currently configured for MUSE frames: every
exposure is aligned to a reference image, and the result is an :ref:`output table <output>`
with RA / Dec offsets, rotation angles (and optionally flux normalisations) per
exposure, together with quality-assessment (QA) plots for every alignment.

The code is developed under the ESO SSDF1-26C project.


The problem this code tries to solve is to perform astrometric registration of images
with a small field of view, various artifacts (such as reprojection artifacts and image slicer
imprints from the integral field optical train) and few or no point sources, in
a robust and automatic way. Therefore, the code relies on several methods and extensive
automated quality checks by cross-referencing alignments.

-----------------------------------------------

In this documentation, we provide guidance on :ref:`installation <installation>` and
:ref:`usage <getting_started>` of the software, 
along with practical information such as contact details and development options.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   getting_started
   worked_example
   structure
   known_limitations
   develop
   contact
