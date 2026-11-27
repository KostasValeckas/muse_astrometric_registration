# CLAUDE.md - context for AI-driven development

Context for AI coding assistants working on `muse_astrometric_registration`.
Read this before changing the code. The user documentation is in `docs/` (Sphinx,
published on Read the Docs); `docs/develop.rst` describes the architecture.

## Version control: hands off

- **Never run git commands in this repository**: no commits, pushes, pulls,
  branches, checkouts, merges, rebases, resets, stashes or tags. The maintainers
  do all version control themselves.
- **Never set yourself as a co-author or credit yourself**: no `Co-Authored-By`
  lines, no "Generated with ..." notes, in commits, pull requests or files.
- When a change is ready, say which files changed and leave committing to the
  maintainers.

## What the code does

Astrometric registration of MUSE images (small field of view, slicer / reprojection
artifacts, few or no point sources): every exposure is aligned to a reference image
(e.g. a WFI R-band image), and the result is an output table with RA / Dec offsets,
a rotation angle (and optionally FLUX_SCALE / BACKGROUND) per exposure, plus QA
plots for every alignment. Robustness comes from several alignment methods and
automated cross-checks (forward / reverse round trips, dither - dither alignments,
composition of transforms to recover failed alignments).

## Repository map

The code is the package `muse_astrometric_registration/` (installable with pip,
`setup.cfg`; internal imports are absolute: `from muse_astrometric_registration.X
import ...`). Its three entry points are commands (`console_scripts`), each
calling `main()` of its module:

- `muse_astrometric_registration_single` -> `run.py`
- `muse_astrometric_registration_hybrid` -> `hybrid_pipeline.py`
- `muse_astrometric_registration_manual` -> `manual_alignment.py`

Modules (in `muse_astrometric_registration/`):

- `run.py` - single method run (`--method`), one `AlignmentPipeline`.
- `hybrid_pipeline.py` - hybrid run: secondary (OpticalFlow) and tertiary
  (PointSources) pipelines first, then the primary (FFT, angle scan on, targeted
  dither mode) with initial guesses read from their tables; merges the tables into
  `Hybrid_alignment_output_table.fits` and copies the final difference images.
- `pipeline.py` - `AlignmentPipeline`: per pointing, reproject dithers onto the
  WCS of dither 1, crop NaN edges, reproject the reference; dither alignment
  (`align_dithers` full / `align_dithers_targeted`); science alignment both ways
  (`_science_forward_reverse`); recovery by composition
  (`_recover_science_alignments`); QA report; output table, flux normalisation,
  final difference images.
- `methods.py` - `AligmentMethod` (abstract base: angle scan, QA helpers, QA
  buffering, error handling) and `NoAlignment`, `PointSources`, `FFT`,
  `OpticalFlow`. By far the largest file.
- `geometry.py` - reprojection, 3x3 transform matrices
  (`transformation_matrix`, `compose_transformations`, `inverse_transformation`,
  `matrix_to_parameters`), pixel <-> sky conversions.
- `flux.py` - `--scale-ref` (borrowed from pymusepipe, MIT): unit conversion,
  FLUX_SCALE / BACKGROUND fit, WFI unit fix.
- `IO.py`, `filters.py`, `stats.py`, `logger.py` - I/O, filters,
  robust statistics / ternary search, logging and run environment.
- `manual_alignment.py` - interactive manual alignment of one exposure.
- Every run writes `run_manifest.json` (reference file and input directory) to its
  output directory; `manual_alignment.py` and the `dev/` replot scripts read it.
- `_vendor/spacepylot/` - bundled copy of the spacepylot fork
  (https://github.com/KostasValeckas/spacepylot, MIT, not on PyPI, so it cannot be
  a dependency). Only its TV-L1 optical flow field is used; the rigid fit is done
  in `OpticalFlow.fit_flow_transform`. Do not edit it here: changes go into the
  fork and are copied in. The local changes to the copy are listed in
  `_vendor/__init__.py` (e.g. the `scipy.odr` import, deprecated and removed in
  SciPy 1.19, is made only inside the never-called `regress_odr`); keep them when
  copying again.
- `dev/` - helper scripts, configuration files, and the worked example data
  (`NGC1365_short.json`, `NGC1365_short/`, ESO data under CC BY 4.0, see
  `dev/NGC1365_short/README.md`):
  - `sorter.py` - makes the JSON configuration file from a directory of exposures.
  - `replot_final_difference_images.py` - redraws the final difference images of a
    finished run with the flux normalisation, without aligning again.
  - `convert_hybrid_table.py` - copy of an output table for pymusepipe (drops rows
    without a result, optional column renaming and sign flips). The sign
    convention relative to pymusepipe is not verified yet; its docs page is a
    draft.
  - `stripe_orientation.py` - experiment: orientation of the MUSE slicer pattern
    from the residual stripes in the difference images. Not part of the pipeline.
- `docs/` - Sphinx documentation; `docs/pictures/` holds its figures.
- `tests/` - only an import smoke test (`pytest tests`, against the installed
  package).

## Running

Install the package in editable mode from the repository root (`pip install -e .`;
`requirements.txt` holds the exact tested versions), then, from the repository
root:

```bash
# worked example / smoke test (~19 min on a laptop, mostly the FFT stage)
muse_astrometric_registration_hybrid dev/NGC1365_short.json \
    dev/NGC1365_short/NGC1365_Rc_flux_nosky_cutout_P01.fits \
    --input_dir dev/NGC1365_short --output_dir NGC1365_short_HYBRID --plot_headless

# single method run
muse_astrometric_registration_single CONFIG.json REFERENCE.fits --input_dir DIR --output_dir OUT --method FFT
```

- Pipeline runs are slow (minutes to hours). Do not start long runs on your own:
  give the maintainers the command to run.
- `pytest tests` only checks that the modules import; the worked example is the
  end-to-end check.
  Its expected results (all 4 exposures aligned by FFT, translations within
  ~0.2 px of OpticalFlow / PointSources) are in `docs/worked_example.rst`.
- Before analysing any results, check the `Code version:` line at the top of the
  log: results from a stale checkout look plausible but are not from the current
  code.
- Write run outputs outside the repository (they are large and not tracked).
- Docs: `pip install -r docs/requirements.txt`, then
  `sphinx-build -b html docs <out_dir>`. The build should have no warnings. Pages
  marked as drafts are excluded in `docs/conf.py` (`exclude_patterns`) and must not
  be linked.

## Coding conventions (follow the existing style)

- black-like formatting, multi-line calls with trailing commas.
- numpy-style docstrings (Parameters / Returns with `-----` underlines).
- Logging with f-strings. In the methods (`methods.py`) the logger is optional:
  `if self.logger: self.logger.info(...)`. `AlignmentPipeline` always creates one
  (`get_logger`) and calls `self.logger.info(...)` directly.
- Short lowercase inline comments; section banners `# ---` / `# Title` / `# ---`;
  `# TODO:` for known shortcuts.
- Keep the existing identifier spellings, even the misspelled ones
  (`AligmentMethod`, `aligment_method`, `plot_failed_aligment`, `aligment_QA`,
  `aligment_log_*.log`, `failed_aligment_*.png`, `succes_table` / `sucess_table`):
  they are part of file names and the API.
- QA plots go to `self.QA_path`, with `{obs_name}_{self.name}` in the file name.
- In `run.py` / `hybrid_pipeline.py` / `manual_alignment.py`, the command line
  code is in `main()`, and the heavy imports come after the matplotlib backend is
  chosen (`--plot_headless`).
- A new command needs a `main()` and an entry in `console_scripts` in
  `setup.cfg`; dependencies go into `install_requires` there (minimum versions).

## Invariants that are easy to break

- **`run()` contract**: every method returns
  `((tx, ty), rotation, (tx_err, ty_err), rotation_err, success)`, or
  `self.FAILED_RETURN`. The pipeline creates methods as
  `Method(QA_path=..., logger=...)`; all other constructor arguments need
  defaults. New methods must be registered in the `switcher` dicts of `run.py` and
  `hybrid_pipeline.py`.
- **Rotation convention**: degrees, positive counter-clockwise with origin upper
  (numpy array as read by astropy, `scipy.ndimage.rotate`). The reported values are
  the correction of the input: rotate about the frame centre, then shift. In
  `geometry.transformation_matrix`, theta = -rotation_deg. Any sign error here
  silently corrupts composed alignments. The user-facing description is in
  `docs/output.rst` (label `rotation_convention`).
- **Composition**: `compose_transformations(T1, T2)` = `T2 @ T1` (T1 first); a
  mirrored measurement is the matrix inverse. Dither tables: `table[a, b]` is the
  alignment a -> b. A dither aligned to itself is the identity and is not run.
- **QA buffering**: `__init_subclass__` wraps every subclass `run()`: QA figures
  saved with `_save_QA_figure` during a run are only written if it succeeds;
  exceptions in `run()` are logged and turned into a failed alignment (the run goes
  on). Nested runs (FFT direct fit before the scan) share the buffer.
- **Status codes** (tables and QA): 1 direct and consistent, 0.5 composed or
  mirrored, 0.7 inconsistent (never written to the output table), 0 failed,
  -1 not attempted (targeted dither mode).
- **Sky conversion** happens only when the output table is written, with the WCS
  of the pointing grid (dither 1); errors go through the WCS Jacobian.
- **Wrong reference units stop the run**: `flux.FluxScaleError` (raised by the
  FLUX_SCALE check before the alignment, `--scale-ref`) must not be swallowed. The
  per-pointing `except Exception` blocks (pipeline, replot script) re-raise it
  first, and `run.py` / `hybrid_pipeline.py` turn it into `SystemExit`. Keep this
  when adding new error handling. `--ref_unit` overrides the `BUNIT` of the
  reference.
- **Angle scan**: the comparison window is sized from the expected translation
  (each method's translation-only fit at 0 degrees, else the initial guess, else
  the first measured translation). Fine scan: 25 points, +/- 0.15 degrees.

## Known limitations (do not "fix" silently, see docs/known_limitations.rst)

- Error estimates use simplifying assumptions and are not validated; the angle
  scan error is too small (image noise shifts the whole scan curve).
- Direct rotation estimators of the methods disagree by ~0.1 degrees; the angle
  scan is the most stable, but can be pulled 0.2-0.9 degrees off.
- Angle scan metric (`_scan_difference_metric`): the old metric divided each frame
  by its own whole-frame median, which does not match their flux scales, so a
  scaled copy of the galaxy stayed in the difference and the minimum moved to a
  wrong alignment (reached by rotating, via the angle dependence of the
  translation). In the big run 19 of 300 FFT results; the direct FFT fit was right
  in all of them. Now `scan_flux_match = "linear"` (default) fits
  reference = a * input + b over the window at every angle; `"median"` keeps the
  old metric for comparisons. Do not go back to per-frame normalisation. Verified
  on NGC3596 pointing 2 and the worked example only; the big run has not been
  repeated with the fix.
- OpticalFlow keeps only ~26% of its flow vectors on real data: the outlier
  threshold is capped at 0.5 px (`residual_threshold`), tuned on synthetic data
  (residual scatter ~0.07 px), while real data scatter by ~0.26 px.
- Some WFI references have `BUNIT = 'mJy'` with data in microJansky
  (`ZUNIT = 'micro Jansky'`); `--scale-ref` checks FLUX_SCALE before the alignment
  and fixes this known case, otherwise stops. The alignment itself is unit
  independent (PointSources' bright-source filter is relative to each frame,
  `filters.BRIGHT_SOURCE_SIGMA`).
- File names with `:` (raw MUSE products) cannot be checked out on Windows; the
  shipped example files use `-` instead.
