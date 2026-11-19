"""
Make a copy of an alignment output table (e.g. Hybrid_alignment_output_table.fits)
for pymusepipe:

- rows without a result are removed (non-finite RA_OFFSET, DEC_OFFSET or
  ROTANGLE, or FLUX_SCALE / BACKGROUND if the table has them),
- optionally, POINTING_OBS is renamed to DATASET_OBS (--dataset_obs),
- optionally, RA_OFFSET, DEC_OFFSET and ROTANGLE are negated (--negate_ra,
  --negate_dec, --negate_rot, each on its own).

What was done is written to the header of the new table (NEG_RA, NEG_DEC,
NEG_ROT, DATASET, NDROPPED). The input table is not changed.

Example::

    python dev/convert_hybrid_table.py results/NGC0628/Hybrid_alignment_output_table.fits \\
        --dataset_obs --negate_ra --negate_dec --negate_rot
"""

import argparse
import os

import numpy as np
from astropy.table import Table


# a row is kept only if these columns (those present) are finite
RESULT_COLUMNS = ["RA_OFFSET", "DEC_OFFSET", "ROTANGLE", "FLUX_SCALE", "BACKGROUND"]


def convert_table(
    table,
    dataset_obs=False,
    negate_ra=False,
    negate_dec=False,
    negate_rot=False,
):
    """
    Convert an alignment output table, see the module docstring.

    Parameters
    ----------
    table : astropy.table.Table
        The output table of the pipeline.

    dataset_obs : bool, optional
        Rename POINTING_OBS to DATASET_OBS.

    negate_ra, negate_dec, negate_rot : bool, optional
        Negate RA_OFFSET, DEC_OFFSET, ROTANGLE.

    Returns
    -------
    new_table : astropy.table.Table
        The converted table.

    dropped : astropy.table.Table
        The rows that were removed.
    """

    for column in ("RA_OFFSET", "DEC_OFFSET", "ROTANGLE"):
        if column not in table.colnames:
            raise ValueError(f"The table has no {column} column.")

    present = [c for c in RESULT_COLUMNS if c in table.colnames]
    keep = np.all(
        [np.isfinite(np.asarray(table[c], dtype=np.float64)) for c in present], axis=0
    )

    new_table = table[keep]
    dropped = table[~keep]

    if dataset_obs:
        if "POINTING_OBS" not in new_table.colnames:
            raise ValueError("The table has no POINTING_OBS column to rename.")
        if "DATASET_OBS" in new_table.colnames:
            raise ValueError("The table already has a DATASET_OBS column.")
        new_table.rename_column("POINTING_OBS", "DATASET_OBS")

    for flag, column in (
        (negate_ra, "RA_OFFSET"),
        (negate_dec, "DEC_OFFSET"),
        (negate_rot, "ROTANGLE"),
    ):
        if flag:
            new_table[column] = -np.asarray(new_table[column], dtype=np.float64)

    # what was done, for whoever reads the table later
    new_table.meta["NEG_RA"] = bool(negate_ra)
    new_table.meta["NEG_DEC"] = bool(negate_dec)
    new_table.meta["NEG_ROT"] = bool(negate_rot)
    new_table.meta["DATASET"] = bool(dataset_obs)
    new_table.meta["NDROPPED"] = int(np.sum(~keep))

    return new_table, dropped


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Copy of an alignment output table without the rows without a "
        "result, optionally with DATASET_OBS and negated offsets / rotation."
    )

    parser.add_argument(
        "input_table",
        type=str,
        help="The output table (e.g. Hybrid_alignment_output_table.fits).",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Name of the new table. Default: <input name>_pymusepipe.fits next to "
        "the input.",
    )

    parser.add_argument(
        "--dataset_obs",
        action="store_true",
        help="Rename the column POINTING_OBS to DATASET_OBS.",
    )

    parser.add_argument(
        "--negate_ra", action="store_true", help="Negate RA_OFFSET."
    )

    parser.add_argument(
        "--negate_dec", action="store_true", help="Negate DEC_OFFSET."
    )

    parser.add_argument(
        "--negate_rot", action="store_true", help="Negate ROTANGLE."
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output table if it exists.",
    )

    args = parser.parse_args()

    output = args.output
    if output is None:
        root, _ = os.path.splitext(args.input_table)
        output = f"{root}_pymusepipe.fits"

    if os.path.abspath(output) == os.path.abspath(args.input_table):
        raise ValueError("The output table would overwrite the input table.")

    table = Table.read(args.input_table, mask_invalid=False)

    new_table, dropped = convert_table(
        table,
        dataset_obs=args.dataset_obs,
        negate_ra=args.negate_ra,
        negate_dec=args.negate_dec,
        negate_rot=args.negate_rot,
    )

    new_table.write(output, format="fits", overwrite=args.overwrite)

    print(f"Read {len(table)} rows from {args.input_table}")
    print(f"Removed {len(dropped)} rows without a result:")
    for row in dropped:
        print(f"    {row['DATE_OBS']}")
    print(
        f"Wrote {len(new_table)} rows to {output} (DATASET_OBS: {args.dataset_obs}, "
        f"negated RA: {args.negate_ra}, DEC: {args.negate_dec}, "
        f"ROTANGLE: {args.negate_rot})"
    )
