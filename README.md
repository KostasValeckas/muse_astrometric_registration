# muse_astrometric_registration

This repository holds code for a Python pipeline that can be used to perform astrometric registration on astronomical images. Currently configured to support frames from MUSE. This code is developed under the ESO SSDF1-26C project.

### Currently in open-beta state - functionality not fully tested


# Installation

***Tested on Python 3.12.***

### Spacepylot submodule 

The repository relies on the ```spacepylot``` submodule, which is a [fork of the original ```spacepylot``` repository](https://github.com/KostasValeckas/spacepylot/tree/5dfca0975f47b163739fc1d551baf6f2da0d2b15).

If you have cloned the code using git, you need to initialize the submodule with:

```
git submodule update --init --recursive

```

The submodule link is set up to use https, so if you want to use ssh instead, you can change the link in the ```.gitmodules``` file.

-------------------------

### Installing dependencies


Extra dependencies are in the ```requirements.txt``` file. Install them with:

```
pip install -r requirements.txt
```

# Worked example

A small dataset (NGC1365, pointing 1, 4 dithers, with a cutout of the WFI R-band
reference) is shipped in ```dev/NGC1365_short```. From the root of the repository:

```
python src/hybrid_pipeline.py dev/NGC1365_short.json dev/NGC1365_short/NGC1365_Rc_flux_nosky_cutout_P01.fits --input_dir dev/NGC1365_short --output_dir NGC1365_short_HYBRID --plot_headless
```

The run takes ~16 minutes on a laptop. The output is described in the documentation,
page "Worked example" (```docs/worked_example.rst```).

# Running the pipeline

------------------------------------------------------------------------------

### Input

The pipeline input is a ```.json``` file with the following formatting:

```
{
    Pointing number (str): [
        [
            Dither number (int),
            Observation time (str),
            Filename (str)
        ],
        ...
    ],
    ...
    "headers": {
        "date_keyword": fits keyword for date of observation (str),
        "mjd_keyword": fits keyword for MJD of observation (str),
        "tpl_start_keyword": fits keyword for template start (str),
        "exp_keyword": fits keyword for exposure number (str),
        "pointing_keyword": fits keyword for pointing number (str)
    }
}
```

There are several examples in the ```dev``` directory. In practice, the 
input file will look something like this (snippet):

```
{
    "1": [
        [
            1,
            "2021-11-27T03:13:14.863",
            "IMAGE_FOV_prealign_OB001_WFI_BB_2021-11-27T03:07:22_0001.fits"
        ],
        [
            2,
            "2021-11-27T03:18:29.277",
            "IMAGE_FOV_prealign_OB001_WFI_BB_2021-11-27T03:07:22_0002.fits"
        ],
        [
            3,
            "2021-11-27T03:23:43.565",
            "IMAGE_FOV_prealign_OB001_WFI_BB_2021-11-27T03:07:22_0003.fits"
        ],
        [
            4,
            "2021-11-27T03:28:58.097",
            "IMAGE_FOV_prealign_OB001_WFI_BB_2021-11-27T03:07:22_0004.fits"
        ]
    ],
    "2": [
        [
            1,
            "2021-11-30T02:12:44.409",
            "IMAGE_FOV_prealign_OB002_WFI_BB_2021-11-30T02:05:15_0001.fits"
        ],
        ...
    ],
    
    ...
    
    "headers": {
        "date_keyword": "DATE-OBS",
        "mjd_keyword": "MJD-OBS",
        "tpl_start_keyword": "HIERARCH ESO TPL START",
        "exp_keyword": "HIERARCH MUSEPIPE_IEXPO",
        "pointing_keyword": "MUSEPIPE_DATASET"
    }
}
```
------------------------------------------------------------------------------

### Preparing the configuration file

There is a script in the ```dev``` directory called ```sorter.py```. It can 
produce a configuration file from a directory with MUSE data. Three FITS keywords 
control the sorting of the data: the pointing number, the exposure number and the observation time.

In the script, the defaults for these are set to the following:

```
pointing_keyword = "HIERARCH MUSEPIPE_POINTING"
exp_keyword = "HIERARCH MUSEPIPE_IEXPO"
date_keyword = "DATE-OBS"
```

The script can be called with the default headers by simply giving the input directory as the argument:

```
python sorter.py /path/to/directory/with/muse/data
```

From experience, for some datasets, the headers might differ. The following flags allow for that:

```
  --pointing_keyword POINTING_KEYWORD
                        Keyword for the pointing. Default is 'HIERARCH MUSEPIPE_POINTING'.
  --exp_keyword EXP_KEYWORD
                        Keyword for the exposure time. Default is 'HIERARCH MUSEPIPE_IEXPO'.
  --date_keyword DATE_KEYWORD
                        Keyword for the observation date. Default is 'DATE-OBS'.

```

Furthermore, these flags can specify the output directory and filename, and two more headers that are used 
for the output table of the pipeline (not needed for initial sorting):

```
  --output_dir OUTPUT_DIR
                        Directory to save the results. Default is current working directory.
  --output_filename OUTPUT_FILENAME
                        Name of the output JSON file. Default is 'sorted_files.json'.
  --mjd_keyword MJD_KEYWORD
                        Keyword for the observation MJD. Default is 'MJD-OBS'.
  --tpl_start_keyword TPL_START_KEYWORD
                        Keyword for the template start. Default is 'HIERARCH ESO TPL START'.
```

------------------------------------------------------------------------------

### Execution

There are two entry points in the ```src``` directory:

* ```run.py``` runs the pipeline with one alignment method.
* ```hybrid_pipeline.py``` runs three methods and combines their results (see below).

------------------------------------------------------------------------------

#### Single method: ```run.py```

An example call to the script is:

```
python run.py NGC0253.json NGC0253_Rc_flux_wcomb.fits --input_dir /NGC0253/MUSE/ --output_dir NGC0253_TEST --method FFT
```

Where the different parameters are:


```
python run.py configuration_file reference_file --input_dir path_to_directory_with_data --output_dir path_to_directory_where_QA_plots_and_results_will_be_stored
```

The optional flags are:

```
  --method METHOD       NoAlignment (default), PointSources, OpticalFlow, FFT, or All
                        (runs PointSources, OpticalFlow and FFT one after the other,
                        mostly for testing).
  --plot_headless       Generate plots without a GUI (use this on servers).
  --distance_threshold DISTANCE_THRESHOLD
                        Round-trip distance (pixels) for the forward/reverse
                        consistency checks. Default is 0.10.
  --skip_rotation_estimation
                        Fix the rotation at zero, only the translation is fitted.
  --scan_for_angles     Estimate the rotation with the shared angle scan instead of
                        each method's own direct rotation estimator.
  --dither_mode {full,targeted}
                        'full' (default) aligns all dither pairs, 'targeted' only the
                        pairs needed to recover failed science alignments.
```

------------------------------------------------------------------------------

#### Alignment methods

* ```NoAlignment``` - does nothing, for testing the plumbing (with the angle scan it
  only scans the rotation).
* ```PointSources``` - matches point sources with ```astroalign``` and fits a
  rotation + translation to the matched pairs (least-squares errors). astroalign
  only returns the sources of its matched triangles (from at most its 50 brightest
  detections), so its transform is used to re-match all sources detected in both
  frames (3 sigma, min. area 5 px, unambiguous mutual nearest neighbours within
  3 px); the astroalign matches are kept if that finds fewer. Every
  alignment gets a QA plot ```point_sources_*.png```: the numbered matched sources
  in both frames (accepted / rejected), the measured displacements, the fit
  residuals in the frame and in the (dx, dy) plane with the threshold, and a
  histogram of the residual lengths against the expected Rayleigh distribution.
* ```OpticalFlow``` - ```spacepylot``` computes the dense TV-L1 flow field between the
  frames, and the rigid transform is fitted to a grid of flow vectors in this
  repository (not with ```spacepylot```'s own homography step, which refers the
  translation to the corner of its cropped frame and whose translation-only fit is
  broken with ```skimage``` >= 0.26). Errors come from a spatial block bootstrap of
  the flow vectors (5x5 tiles). If the inliers lie in too few tiles for the
  bootstrap, a conservative fallback is used (residual scatter of the flow vectors
  over sqrt(number of tiles)), never 0. Every alignment gets a flow field QA plot
  ```flow_field_*.png```: the measured flow, the fitted rigid model, the residual flow
  and the residual vectors.

  Because TV-L1 can fail silently (flow frozen at zero, which is perfectly smooth and
  passes the forward/reverse consistency check):
  * The filtered frames are scaled to unit standard deviation before TV-L1. Without
    this the flow of smooth fields stays at zero (a 3.6 px shift came out as 0.02 px).
  * Control-shift check: the input aligned with the result is shifted by a known
    (1.7, -2.3) px, and the flow must recover that shift within 0.3 px. If not, the
    alignment is set as failed (only ```failed_aligment_*.png```); the log states the
    recovered error.

Both ```PointSources``` and ```OpticalFlow``` reject outliers iteratively with an
adaptive threshold on the length of the residual vector: 3 robust sigmas of the
residual scatter (sigma = median residual length / 1.1774, the Rayleigh median, so
robust to up to 50% outliers), clipped to a lower and upper limit. Residuals of all
points are recomputed in every pass, so a rejected point can come back.

* ```OpticalFlow```: limits [0.05, 0.5] px (TV-L1 residuals scatter by ~0.07 px).
* ```PointSources```: limits [0.1, 1.5] px. Matched centroids on real MUSE data scatter
  by 0.2-0.8 px per axis and mismatches are off by 1.5-5 px, so a fixed 0.5 px cut
  threw away up to half of the good sources. The first fit uses a robust (soft-L1)
  loss, so mismatches do not set the first threshold. An alignment needs at least
  5 accepted sources to count as successful; if astroalign finds fewer matches, its
  settings are loosened step by step before giving up.
* ```FFT``` - phase correlation on band-passed (difference of Gaussians) frames, over
  a range of filter scales (sigmas), with a known control offset as a self-check. A
  translation is accepted when several consecutive sigmas agree, the control offset is
  recovered (```control_tol```, 0.05 px), and the peak has at least 90% of the control
  peak SNR and a similar shape (elliptical Gaussian fits, shown in the
  ```translation_peak_*.png``` QA plots). With an initial guess, the peak is only
  searched within +/- 1 px of it.

------------------------------------------------------------------------------

#### Rotation convention

All rotations in the pipeline (tables, logs, QA plots) follow one convention: degrees,
positive counter-clockwise as the frame is displayed with origin upper, i.e. as the
numpy array is read by ```astropy.io.fits``` (the ```scipy.ndimage.rotate```
convention). The reported values are the correction to apply to the input frame:
rotate by the angle about the frame centre, then shift by the offsets.

Note: in DS9 (origin lower) the same correction looks clockwise.

All dithers and the reference are reprojected onto the WCS of the first dither of a
pointing, so all alignments are measured on that pixel grid. Since every frame is
reprojected with its own WCS first, the measured values are the residual corrections
of the WCS, not the raw rotations.

------------------------------------------------------------------------------

#### Angle scan (```--scan_for_angles```)

Instead of each method's own rotation estimator, the rotation can be found by
minimising the difference between the reference and the rotated + shifted input: the mean absolute difference of the
blurred frames over a fixed central window.

The window is clipped so that it stays covered by the input for the translation. It leaves room for at least
`scan_max_shift` (10 px), or for the expected translation + `scan_shift_slack` (3 px) if that is larger. The expected
translation is each method's translation-only fit at 0 degrees (the direct fit for FFT). If that fit fails, the initial
guess is used, and without one, the first translation measured in the scan.

The scan itself:

1. A ternary search over [-1, 1] degrees finds the rough minimum. It reports failure
   if it ends at the edge of the range or if all evaluations give the same value
   (e.g. every translation estimate failed).
2. A fine scan around it (+/- 0.15 degrees, 25 points)
   estimates the translation at every angle.
3. A linear trend is fitted to tx(angle) and ty(angle), translation outliers are
   rejected, and all angles are re-scored with the smooth trend translation, so the
   jitter of the individual estimates does not leak into the metric.
4. Both a parabola and a hyperbola, ```sqrt(a (x - x0)^2 + b) + c```, are fitted to
   the fine scan with a 2-pass outlier rejection, and the one with the lower AICc
   (compared on the same points) is used. The metric is a rounded V: when the fine
   scan reaches the straight arms the parabola is biased and the hyperbola wins, when
   it only sees the rounded bottom the hyperbola degenerates and the parabola wins.
   Fits are only thrown away if their shape makes no sense: no minimum, minimum
   outside the search range or extrapolated more than one fine scan width beyond the
   scan, or a curve that is flat within the noise. The minimum may lie outside the
   fine scan. The angle error comes from a block bootstrap of the fit residuals.

The QA plot ```angle_scan_model_*.png``` shows the scan, the fit and its residuals,
and the translation per angle with its trend and residuals.

**FFT with the angle scan** (the default for the primary method of the hybrid run):

* The direct fit (rotation + translation) is run first, as a cheap check. If its
  translation fails, the alignment fails without running the slow scan.
* If only its rotation estimate fails (no valid entry in the per-sigma rotation
  table), the shift may still be measurable, and the scan does not need the
  direct rotation. With `scan_on_rotation_failure` (default on), a
  translation-only fit at 0 degrees is tried:
  * if it fails, the alignment fails without the scan;
  * if it works, the scan is run with the full sigma range and that translation
    as the expected one. There is no direct fit to fall back on, so if the scan
    fails, the alignment fails.

  With the option off, any failure of the direct fit fails the alignment.
* If the direct fit agrees with the initial guess (translation within 1 px, rotation
  within 0.1 degrees), the scan only searches the sigmas around the ones the direct
  fit agreed on (+/- 5 grid steps), which makes every scanned angle much cheaper.
  If this narrowed scan fails, it is redone once with the full sigma range.
* If the scan fails after a successful direct fit (e.g. the ternary search ends at
  the boundary of the angle range), the direct fit is returned instead
  (```fallback_to_direct_fit```); the round-trip and dither checks still apply.
* If no translation is found, the log lists the acceptance criteria every sigma
  failed (direct and final estimates only, not every scanned angle).
* The final translation is the trend at the final angle (error: rms scatter of the
  individual translations around the trend), marked with stars in the angle-scan QA
  plot. No extra translation estimate is made at the end.

**Known limitation:** the reported angle error only covers the scatter of the scan
points around the fit. The noise in the images is the same at every scanned angle,
so it shifts the whole curve, and this is not included yet (on synthetic frames the
minimum moves by ~0.01 degrees between noise realisations, much more than the
reported error).

------------------------------------------------------------------------------

#### Reference flux scaling (```--scale-ref```)

The reference image can be in different units, and on a different flux scale and
background, than the MUSE images. With ```--scale-ref``` (```run.py``` and
```hybrid_pipeline.py```) the method of pymusepipe is used (borrowed in ```src/flux.py```,
MIT license):

Before the alignment, only the unit conversion:

1. when the reference is read, it is converted from its unit to the MUSE unit
   (1e-20 erg/s/cm2/A) with the spectral density equivalency at the pivot wavelength
   of ```--ref_filter``` (default ```WFI_BB```, 6483.58 A; the list is ```REFERENCE_FILTERS```
   in ```flux.py```, the same as pymusepipe's ```dict_equivalencies```). The unit is read
   from the ```BUNIT``` keyword of the reference; if there is none, the unit of the
   filter is assumed (1 microJansky for WFI_BB), with a warning. A ```BUNIT``` that
   cannot be converted (e.g. counts, a surface brightness, or ```A``` - which is
   Ampere, Angstrom is ```Angstrom```) stops the run before any alignment.
   If the ```BUNIT``` of the reference is wrong, set the unit with ```--ref_unit```
   (e.g. ```--ref_unit uJy```), which is used instead. Some WFI references have
   ```BUNIT = 'mJy'``` while their data are in microJansky: they give FLUX_SCALE ~ 1000.
   A wrong unit makes FLUX_SCALE off by the same factor (BACKGROUND, in the unit
   of the exposure, is not). The alignment itself is not affected: all methods are
   insensitive to a constant factor (the bright-source filter of ```PointSources```
   is relative to each frame). To catch this, FLUX_SCALE is first fitted for every pointing
   before any alignment (dither 1, as reprojected with its WCS):
   1. With the unit of the header (or ```--ref_unit```): used if FLUX_SCALE is within
      a factor 10 of 1.
   2. If not, the known WFI case is fixed: some WFI references
      (```*_Rc_flux_nosky.fits```) have ```BUNIT = 'mJy'``` while the same header says
      ```ZUNIT = 'micro Jansky'```, and the data are in microJansky. Only if the unit
      came from ```BUNIT```, ```BUNIT``` is mJy and ```ZUNIT``` is microJansky, the
      reference is converted as microJansky, and used if FLUX_SCALE is then within a
      factor 10 of 1 (warning in the log). This is decided at the first pointing
      and used for the whole run.
   3. Otherwise everything about the conversion is logged as an error (file, header
      units, conversion and area factors, image medians, the fit, why the fix did
      not apply) and the run stops (the hybrid run as a whole): it does not run with
      a wrong unit. A run whose median FLUX_SCALE after the alignment is more than a factor
   10 from 1 ends with a warning in the log.
2. after the reprojection to the frame of a pointing, it is multiplied by the pixel
   area ratio (MUSE / reference).

After the alignment, for every exposure with a result,
```reference = FLUX_SCALE * (exposure + BACKGROUND)``` is fitted with the exposure
shifted by its final offsets (medians of 10 x 10 pixel chunks of the median filtered
images, orthogonal distance regression with the chunk MAD standard deviations as
errors), and written to the output table as ```FLUX_SCALE``` and ```BACKGROUND```. These
have the same meaning as pymusepipe's OFFSET table columns. Exposures without a
result, or with a fit that is not usable (fewer than 10 chunks with signal), get
NaN. pymusepipe fits with ```scipy.odr``` (removed from SciPy 1.19); here the same
problem is solved with ```scipy.optimize.least_squares```, which gives the same values
to the precision of the fit. The dither alignments are not affected.

**Final difference images.** Every reported result gets a final difference image,
```aligment_QA/science/difference_image_{pointing}_dither_{i}_final_{method}.png```,
always with the exposure shifted onto the reference by the final offsets (whether
they were measured forward, are the inverse of the reverse measurement, or were
composed via other dithers). These are the images copied to
```final_difference_images/``` in the hybrid run. With ```--scale-ref``` the reference is put
on the flux scale of the exposure (```reference / FLUX_SCALE - BACKGROUND```) and the
difference is shown in MUSE units, with a colour scale symmetric around 0, so a
remaining flux or background difference is visible; without it, both images are
normalised to their maximum as in the other QA plots. The difference images made
during the measurements are kept as they are.

------------------------------------------------------------------------------

#### Dither alignment and status codes

Every dither is aligned to the reference in both directions (reference -> dither and
dither -> reference), and the two results are checked for consistency (round trip
within ```--distance_threshold```). Dithers are also aligned to each other, which is
used to recover the science alignments that failed:

* ```--dither_mode full``` aligns all dither pairs before the science alignment.
* ```--dither_mode targeted``` aligns the science frames first, and then only the
  dither pairs needed to recover the ones that failed. Every link is aligned in both
  directions and checked for consistency, like the science alignment; a link is only
  used if both directions work and agree (a one-sided or disagreeing link is marked
  inconsistent). The candidate paths are tried one at a time, shortest first: e.g. to
  recover dither 3 from clean dither 1 it tries 3 - 1, then 3 - 2 - 1, then 3 - 4 - 1,
  and so on, skipping every path through a link that already failed. Paths that reuse
  already verified links and links the guess methods measured go first. Every link is
  aligned at most once. If all science alignments are clean, no dither alignment is
  run at all.

A failed science alignment is replaced by the shortest composition through direct
measurements (reference -> clean dither -> ... -> failed dither). The path length is
the number of direct measurements; a mirrored measurement (the inverse of the
opposite direction) counts as direct, and there is no limit on the length. Errors of
the links are added in quadrature.

The status codes in the tables and QA plots are:

```
  1     direct measurement, consistent
  0.5   composed (or mirrored)
  0.7   inconsistent - never written to the output table (null values)
  0     failed
 -1     not attempted (targeted dither mode, light gray in the QA plots)
```

------------------------------------------------------------------------------

#### Hybrid run: ```hybrid_pipeline.py```

The hybrid run combines three methods:

1. The secondary (default ```OpticalFlow```) and tertiary (default ```PointSources```)
   methods run first, as isolated runs of the pipeline, without initial guesses and
   with their direct rotation estimators.
2. The primary method (default ```FFT```) runs with initial guesses read from the
   results on disk: from the secondary method if it has a result, else from the
   tertiary, else without a guess. The angle scan is on by default for the primary.
3. The final table takes, for every exposure, the primary result if there is one,
   else the secondary, else the tertiary, else null, and adds a column
   ```ALIGNMENT_METHOD``` with the method that was used.
4. The difference images of exactly the results in the final table are copied to
   ```final_difference_images/```.

An example call is:

```
python hybrid_pipeline.py NGC1365.json ../reference/NGC1365_reference.fits --input_dir ../MUSE/ --output_dir NGC1365_HYBRID --plot_headless
```

The optional flags are:

```
  --primary PRIMARY     Primary method. Default is FFT.
  --secondary SECONDARY
                        Secondary method (first source of initial guesses).
                        Default is OpticalFlow.
  --tertiary TERTIARY   Tertiary method (second source of initial guesses).
                        Default is PointSources.
                        (Options for all three: NoAlignment, PointSources, FFT,
                        OpticalFlow.)
  --plot_headless       Generate plots without a GUI.
  --distance_threshold DISTANCE_THRESHOLD
                        Round-trip distance (pixels) for the consistency checks.
                        Default is 0.10.
  --skip_rotation_estimation
                        Fix the rotation at zero for all methods.
  --no_primary_angle_scan
                        The primary method uses its direct rotation estimator
                        instead of the angle scan.
  --secondary_angle_scan
                        The secondary method uses the angle scan.
  --tertiary_angle_scan
                        The tertiary method uses the angle scan.
  --primary_dither_mode {full,targeted}
                        Default is 'targeted' (the primary is the slow method).
  --secondary_dither_mode {full,targeted}
                        Default is 'full'.
  --tertiary_dither_mode {full,targeted}
                        Default is 'full'.
```

The output directory of a hybrid run looks like this:

```
NGC1365_HYBRID/
    secondary_OpticalFlow/          full output of the secondary run
    tertiary_PointSources/          full output of the tertiary run
    primary_FFT/                    full output of the primary run
    final_difference_images/        difference images of the reported results
    Hybrid_alignment_output_table.fits
    aligment_log_*.log
```

------------------------------------------------------------------------------

#### Output

Every run (and every method directory of a hybrid run) contains:

* ```{method}_alignment_output_table.fits``` - one row per exposure, in the order of
  the configuration file: ```DATE_OBS, MJD_OBS, TPL_START, IEXPO_OBS, POINTING_OBS,
  RA_OFFSET, RA_OFFSET_ERROR, DEC_OFFSET, DEC_OFFSET_ERROR, ROTANGLE, ROTATION_ERROR```.
  Offsets and their errors are in degrees of RA and Dec; the pixel errors are
  propagated through the WCS Jacobian, so they are always positive and correct for
  any position angle. Failed and inconsistent alignments have null values.
  With ```--scale-ref``` there are two more columns, ```FLUX_SCALE``` and ```BACKGROUND```
  (see above; NaN without a result; in the hybrid table taken from the same method
  as the offsets).
* ```{x,y,rotation}[_error]_dither_table_{pointing}_{method}.csv``` and
  ```success_dither_table_{pointing}_{method}.csv``` - the dither-to-dither
  tables (```table[a, b]``` is the alignment of dither a -> dither b). These are
  also what the hybrid run reads for the initial guesses.
* ```aligment_QA/dithers```, ```aligment_QA/science``` and ```aligment_QA/combined``` -
  the QA plots: difference images (also for failed, mirrored and composed
  alignments), angle scans, FFT peak and progression plots, optical flow fields, point source matches,
  and a one-page report per pointing. A failed alignment only gets its failed
  difference image (```failed_aligment_*.png```): the other QA plots of an alignment
  are held back until it succeeds.
* ```aligment_log_*.log``` - the log of the run. The last line gives the total execution time (for the hybrid run, the top-level log gives the time of all three runs together). A crashed or interrupted run ends with "ABORTED" and the traceback instead of "finished" (also when a batch system stops the job with SIGTERM, e.g. at the SLURM time limit). The log starts with the code version (git commit, and a warning if the code has local changes), host, package versions, command line, the pipeline settings and the method parameters; every alignment logs its duration, every pointing its duration and the peak memory, and the run ends with a summary.
* Unexpected errors do not end a run: an alignment that raises is logged with the traceback and set as failed, a pointing that raises (e.g. an unreadable file) is written as failed, and in the hybrid run a method that crashes only loses its own results. Search the logs for "Unexpected error" and check the summary lines at the end.
* ```run_manifest.json``` - the reference file and input directory of the run. A run
  into an output directory that holds the output of a different reference file is
  refused (use a new output directory, or delete the manifest if the old output is
  not needed); a different input directory with the same reference only gives a
  warning.

------------------------------------------------------------------------------

# Data license

The data in ```dev/NGC1365_short``` are ESO data, made available under the Creative
Commons Attribution 4.0 International license
([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)), following the
[ESO data access policy](https://archive.eso.org/cms/eso-data-access-policy.html).
Based on observations collected at the European Southern Observatory under ESO
programme 1100.B-0651. See ```dev/NGC1365_short/README.md```.
