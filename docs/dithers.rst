.. _dithers:

Dither alignment and status codes
=================================

Every dither is aligned to the reference in both directions (reference -> dither and
dither -> reference), and the two results are checked for consistency (round trip
within ``--distance_threshold``). Dithers are also aligned to each other, which is
used to recover the science alignments that failed:

* ``--dither_mode full`` aligns all dither pairs before the science alignment.
* ``--dither_mode targeted`` aligns the science frames first, and then only the
  dither pairs needed to recover the ones that failed. Every link is aligned in both
  directions and checked for consistency, like the science alignment; a link is only
  used if both directions work and agree (a one-sided or disagreeing link is marked
  inconsistent). The candidate paths are tried one at a time, shortest first: e.g.
  to recover dither 3 from clean dither 1 it tries 3 - 1, then 3 - 2 - 1, then
  3 - 4 - 1, and so on, skipping every path through a link that already failed.
  Paths that reuse already verified links and links the guess methods measured go
  first. Every link is aligned at most once. If all science alignments are clean,
  no dither alignment is run at all.

A failed science alignment is replaced by the shortest composition through direct
measurements (reference -> clean dither -> ... -> failed dither). The path length is
the number of direct measurements; a mirrored measurement (the inverse of the
opposite direction) counts as direct, and there is no limit on the length. Errors of
the links are added in quadrature.

Status codes
------------

The status codes in the tables and QA plots are:

.. list-table::
    :header-rows: 1

    * - Code
      - Meaning
    * - 1
      - direct measurement, consistent
    * - 0.5
      - composed (or mirrored)
    * - 0.7
      - inconsistent - never written to the output table (null values)
    * - 0
      - failed
    * - -1
      - not attempted (targeted dither mode, light gray in the QA plots)
