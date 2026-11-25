# NGC1365_short: worked example data

A small dataset for the worked example in the documentation (Getting started ->
Worked example). It is used together with the configuration file
`dev/NGC1365_short.json`.

## Files

* `IMAGE_FOV_prealign_P01_WFI_BB_2018-01-10T02-49-23_000{1,2,3,4}.fits` -
  the 4 dithers of pointing 1 of NGC1365, MUSE (VLT UT4), reconstructed images in
  the WFI BB filter (`IMAGE_FOV_prealign` products of the PHANGS-MUSE reduction
  with pymusepipe). Unmodified copies, renamed: the `:` of the original file
  names (`..._2018-01-10T02:49:23_000N.fits`) are replaced by `-`, so the files
  can be checked out on every operating system. Image in extension 1, units
  1e-20 erg/s/cm2/A.
* `NGC1365_Rc_flux_nosky_cutout_P01.fits` - cutout of the WFI R-band reference
  image `NGC1365_Rc_flux_nosky.fits` (5657 x 5340 pixels, 0.2375 arcsec/pixel,
  241 MB). The cutout was made with `astropy.nddata.Cutout2D`: centred on the
  footprint of the 4 dithers (from their WCS), with a margin of 30 arcsec on every
  side, 521 x 518 pixels (~124 x 123 arcsec), stored as float32. The WCS is updated
  for the cutout; the rest of the primary header is kept (including
  `BUNIT = 'mJy'`, while the data are in microJansky, see `ZUNIT`). The origin of
  the cutout in the full image is written in a `HISTORY` card.

The full reference gives the same alignment results as the cutout (checked with
the hybrid run of the worked example): all exposures use FFT in both runs, the
offsets agree within 0.001 arcsec (below their errors) and the rotations within
0.0012 degrees. The reference reprojected to the MUSE frame differs between the
two only at float32 precision (relative 4e-8).

## Data license

The data in `dev/NGC1365_short` are ESO data, made available under the Creative
Commons Attribution 4.0 International license (CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/), following the ESO data access
policy (https://archive.eso.org/cms/eso-data-access-policy.html). Based on
observations collected at the European Southern Observatory under ESO programme
1100.B-0651. The MUSE frames are from the PHANGS-MUSE survey.
