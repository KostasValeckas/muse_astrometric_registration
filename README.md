# muse_astrometric_registration

### DEVELOPMENT GROUNDS ###

# KEEP PRIVATE #

# Installation

***Tested on Python 3.12.***

### Spacepylot submodule 

The repository relies on the ```spacepylot``` submodule, which is a [fork of the original ```spacepylot``` repository](https://github.com/KostasValeckas/spacepylot/tree/5dfca0975f47b163739fc1d551baf6f2da0d2b15).

If you have cloned the code using git, you need to initialize the submodule with:

```
git submodule update --init --recursive

```

The submodule link is setup to use https, so if you want to use ssh instead, you can change the link in the ```.gitmodules``` file.

-------------------------

### Installing dependencies


Extra dependencies are in the ```requirements.txt``` file. Install them with:

```
pip install -r requirements.txt
```

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

Furthermore, these flags can specify output filename, and two more headers that are used 
for the output table of the pipeline (not needed for initial sorting):

  --output_dir OUTPUT_DIR
                        Directory to save the results. Default is current working directory.
  --output_filename OUTPUT_FILENAME
                        Name of the output JSON file. Default is 'sorted_files.json'.
  --mjd_keyword MJD_KEYWORD
                        Keyword for the observation MJD. Default is 'MJD-OBS'.
  --tpl_start_keyword TPL_START_KEYWORD
                        Keyword for the template start. Default is 'HIERARCH ESO TPL START'.



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
  rotation + translation to the matched pairs (least-squares errors).
* ```OpticalFlow``` - ```spacepylot``` computes the dense TV-L1 flow field between the
  frames, and the rigid transform is fitted to a grid of flow vectors in this
  repository (not with ```spacepylot```'s own homography step, which refers the
  translation to the corner of its cropped frame and whose translation-only fit is
  broken with ```skimage``` >= 0.26). Errors come from a spatial block bootstrap of
  the flow vectors (5x5 tiles, QA plot ```bootstrap_*.png```).
* ```FFT``` - phase correlation on band-passed (difference of gaussians) frames, over
  a range of filter scales (sigmas), with a known control offset as a self-check. A
  translation is accepted when several consecutive sigmas agree, the control offset is
  recovered (```control_tol```, 0.05 px), and the peak has at least 90% of the control
  peak SNR and a similar shape (elliptical gaussian fits, shown in the
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
minimising the difference between the reference and the rotated + shifted input
(mean absolute difference of the blurred frames, over a fixed central window):

1. A ternary search over [-1, 1] degrees finds the rough minimum. It reports failure
   if it ends at the edge of the range or if all evaluations give the same value
   (e.g. every translation estimate failed).
2. A fine scan around it (+/- 0.15 degrees, 25 points for FFT, 50 otherwise)
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

* The direct fit is run first. If it fails, the alignment fails without running the
  (slow) scan.
* If the direct fit agrees with the initial guess (translation within 1 px, rotation
  within 0.1 degrees), the scan only searches the sigmas around the ones the direct
  fit agreed on (+/- 5 grid steps), which makes every scanned angle much cheaper.
* The final translation is the trend at the final angle (error: rms scatter of the
  individual translations around the trend), marked with stars in the angle-scan QA
  plot. No extra translation estimate is made at the end.

**Known limitation:** the reported angle error only covers the scatter of the scan
points around the fit. The noise in the images is the same at every scanned angle,
so it shifts the whole curve, and this is not included yet (on synthetic frames the
minimum moves by ~0.01 degrees between noise realisations, much more than the
reported error).

------------------------------------------------------------------------------

#### Dither alignment and status codes

Every dither is aligned to the reference in both directions (reference -> dither and
dither -> reference), and the two results are checked for consistency (round trip
within ```--distance_threshold```). Dithers are also aligned to each other, which is
used to recover the science alignments that failed:

* ```--dither_mode full``` aligns all dither pairs before the science alignment.
* ```--dither_mode targeted``` aligns the science frames first, and then only the
  dither pairs needed to recover the ones that failed. It searches for the shortest
  path one length at a time, trying the links that the guess methods measured first,
  and aligns every link at most once. If all science alignments are clean, no dither
  alignment is run at all. Only one direction of every link is aligned, so the links
  get no consistency check.

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
* ```{x,y,rotation}[_error]_dither_table_{pointing}_{method}.csv``` and
  ```success_dither_table_{pointing}_{method}.csv``` - the dither-to-dither
  tables (```table[a, b]``` is the alignment of dither a -> dither b). These are
  also what the hybrid run reads for the initial guesses.
* ```aligment_QA/dithers```, ```aligment_QA/science``` and ```aligment_QA/combined```
  - the QA plots: difference images (also for failed, mirrored and composed
  alignments), angle scans, FFT peak and progression plots, optical flow bootstraps,
  and a one-page report per pointing.
* ```aligment_log_*.log``` - the log of the run. The last line gives the total execution time (for the hybrid run, the top-level log gives the time of all three runs together).

------------------------------------------------------------------------------


### Repository structure

![Alignment Repository Sketch](./docs/alignment_repo_sketch.png)


