.. _developer:

Interested in developing?
=========================

The project is still in beta and contributions are welcome.

If you are interested in contributing to the project, :ref:`contact us <contact>`
directly.

If you are interested in adapting the pipeline to your own instrument, feel free
to fork the repository and work on a local copy. If you do so, please reach out
so we can explore collaboration options. The only input contract is the
:ref:`configuration file <conf>`, see :ref:`adapting to other instruments
<adapting_instruments>`.

Pipeline architecture
---------------------

The code is the package ``muse_astrometric_registration/``. It is organised in
three layers: the entry points, one generic pipeline class, and interchangeable
:ref:`alignment methods <methods>`.

.. code-block:: text

    run.py ───────────────► AlignmentPipeline ──► AligmentMethod.run()
    hybrid_pipeline.py ──►  (pipeline.py)         (methods.py: NoAlignment,
      3 x AlignmentPipeline      │                 PointSources, FFT, OpticalFlow)
      + merge of the tables      │
                                 ├── IO.py        FITS / WCS reading, output tables
                                 ├── geometry.py  reprojection, transforms, WCS
                                 ├── filters.py   NaN handling, band-pass filters
                                 ├── stats.py     robust statistics, ternary search
                                 ├── flux.py      reference flux scaling
                                 └── logger.py    logging, run environment

**Entry points.**

The entry points are installed as commands (``console_scripts`` in
``setup.cfg``), each calling the ``main()`` function of its module:
``muse_astrometric_registration_single`` (``run.py``),
``muse_astrometric_registration_hybrid`` (``hybrid_pipeline.py``) and
``muse_astrometric_registration_manual`` (``manual_alignment.py``). For
development, install the package in editable mode (``pip install -e .`` from the
repository root), so changes to the code are used by the commands right away.

* ``run.py`` (:ref:`single method run <run>`) reads the :ref:`configuration file
  <conf>`, picks the alignment method class from ``--method`` and runs one
  ``AlignmentPipeline``.
* ``hybrid_pipeline.py`` (:ref:`hybrid run <hybrid>`) runs three
  ``AlignmentPipeline`` instances one after the other (secondary, tertiary, then
  the primary with initial guesses read from the tables of the first two), and
  merges their :ref:`output tables <output>` into ``Hybrid_alignment_output_table.fits``.
* ``manual_alignment.py`` is an interactive tool to align one exposure by hand,
  set up from the output of a run so that its offsets follow the same convention
  as the output table.
* ``_vendor/spacepylot/`` is a bundled copy of a `fork of spacepylot
  <https://github.com/KostasValeckas/spacepylot>`_ (MIT license; not on PyPI),
  whose TV-L1 optical flow field is used by ``OpticalFlow``. Changes to it are
  made in the fork and copied here.
* ``dev/`` holds helper scripts, e.g. the :ref:`configuration file sorter
  <sorter>`, and the data of the :ref:`worked example <worked_example>`.

**The pipeline** (``AlignmentPipeline`` in ``pipeline.py``) is independent of the
alignment method. For every pointing it:

#. reads the dithers and reprojects them onto the WCS of the first dither, crops
   the common NaN edges, and reprojects the reference onto the same grid (with
   ``--scale-ref``, also converts its unit, see :ref:`reference flux scaling
   <scale_ref>`);
#. in ``full`` dither mode, aligns all dither pairs (``align_dithers``);
#. aligns every dither to the reference in both directions and checks the round
   trip (``_science_forward_reverse``);
#. in ``targeted`` dither mode, aligns only the dither pairs needed to recover the
   failed science alignments (``align_dithers_targeted``);
#. recovers failed science alignments by composing transforms through the other
   dithers (``_recover_science_alignments``, see :ref:`dither alignment
   <dithers>`);
#. writes the QA plots, the one-page report of the pointing and the dither tables.

At the end it fits the flux normalisation (``--scale-ref``), makes the
:ref:`final difference images <final_difference_images>`, and writes the
:ref:`output <output>` table and the log. An unexpected error in one pointing is
logged and the pointing is written as failed, so a long run does not stop.

**The alignment methods** (``methods.py``) are subclasses of ``AligmentMethod``.
The pipeline only calls their ``run()`` method, which aligns one input frame to
one reference frame and returns the translation, the rotation, their errors and
a success flag. The base class owns everything shared between the methods: the
:ref:`angle scan <angle_scan>`, the difference image and failed alignment QA
plots, the buffering of QA plots (only written if the alignment succeeds) and the
error handling around ``run()``. A new method therefore only has to implement
``run()``, see :ref:`implementing a new method <new_method>`.

**Conventions shared by all layers.** All rotations follow the :ref:`rotation
convention <rotation_convention>`. Alignments are combined as 3x3 homogeneous
matrices (``geometry.py``, see :ref:`composing transformations
<composing_transformations>`). All alignments of a pointing are measured
in pixels on the grid of its first dither; they are converted to RA / Dec offsets
with that WCS only when the output table is written.

AI-driven development
---------------------

The file ``dev/CLAUDE.md`` is provided for AI-driven development. It is a context
file for AI coding assistants (such as Claude Code) with an overview of the
repository, how to run and test it, the coding conventions, and the invariants and
pitfalls of the code that are easy to break. Point your assistant to it (or copy it
to the repository root) before asking it to change the code. Keep it up to date when
the architecture changes.
