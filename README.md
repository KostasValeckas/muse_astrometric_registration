# muse_astrometric_registration

**Documentation: https://muse-astrometric-registration.readthedocs.io**

> **Open beta:** the software is still being tested, so results and interfaces may
> change. See the [known limitations](https://muse-astrometric-registration.readthedocs.io/en/latest/known_limitations.html).

muse_astrometric_registration is a Python pipeline for the astrometric registration
of astronomical images, currently configured for MUSE frames. Every exposure is
aligned to a reference image, and the result is an output table with RA / Dec
offsets, rotation angles (and optionally flux normalisations) per exposure, together
with quality-assessment (QA) plots for every alignment.

The pipeline is built for images with a small field of view, artifacts (such as
reprojection artifacts and image slicer imprints from the integral field optical
train) and few or no point sources. To align them robustly and automatically, it
combines several alignment methods (FFT phase correlation, optical flow and point
source matching) with extensive automated quality checks: every alignment is
measured in both directions, the dithers of a pointing are aligned to each other,
and failed alignments are recovered by composing the alignments through other
dithers.

The code is developed under the ESO SSDF1-26C project.

Installation, a worked example with data shipped in this repository, and the full
description of the pipeline are in the
[documentation](https://muse-astrometric-registration.readthedocs.io).

## License

The code is released under the MIT License, see `LICENSE`.

The data in `dev/NGC1365_short` are ESO data, made available under the Creative
Commons Attribution 4.0 International license
([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)), following the
[ESO data access policy](https://archive.eso.org/cms/eso-data-access-policy.html).
Based on observations collected at the European Southern Observatory under ESO
programme 1100.B-0651. See `dev/NGC1365_short/README.md`.
