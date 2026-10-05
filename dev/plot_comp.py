import numpy as np
from astropy.io import fits
from astropy.table import Table

import matplotlib.pyplot as plt

path_reference = "NGC1365_offset_table_DR2.1.fits"
path_pipeline = "output_table_NGC1365.fits"


# reference table

with fits.open(path_reference) as hdul:
    reference_table = Table(hdul[1].data)

# pipeline table
with fits.open(path_pipeline) as hdul:
    pipeline_table = Table(hdul[1].data)
    print(pipeline_table.colnames)

# leave out last 8 obs
date_obs_reference = reference_table["DATE_OBS"][:-8]
ra_offset_reference = reference_table["RA_OFFSET"][:-8]
dec_offset_reference = reference_table["DEC_OFFSET"][:-8]
rot_offset_reference = reference_table["ROTANGLE"][:-8]



date_obs_pipeline = pipeline_table["DATE_OBS"]
ra_offset_pipeline = pipeline_table["RA_OFFSET"]
dec_offset_pipeline = pipeline_table["DEC_OFFSET"]
rot_offset_pipeline = pipeline_table["ROTANGLE"]

# build nan mask buold on pipeline data
mask = ~np.isnan(ra_offset_pipeline) & ~np.isnan(dec_offset_pipeline) & ~np.isnan(rot_offset_pipeline)

# apply the mask to all the reference data
ra_offset_reference = ra_offset_reference[mask]
dec_offset_reference = dec_offset_reference[mask]
rot_offset_reference = rot_offset_reference[mask]

# and to the pipeline data
ra_offset_pipeline = ra_offset_pipeline[mask]
dec_offset_pipeline = dec_offset_pipeline[mask]
rot_offset_pipeline = rot_offset_pipeline[mask]



fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# Plot RA for comparison
ax1.plot(ra_offset_reference, label="Hand-alignment", marker="o")
ax1.plot(ra_offset_pipeline, label="Pipeline", marker="x")
ax1.set_xlabel("OBS")
ax1.set_ylabel("RA Offset")
ax1.set_title("RA Offset Comparison")
ax1.legend()

# Plot Dec for comparison
ax2.plot(dec_offset_reference, label="Hand-alignment", marker="o")
ax2.plot(dec_offset_pipeline, label="Pipeline", marker="x")
ax2.set_xlabel("OBS")
ax2.set_ylabel("Dec Offset")
ax2.set_title("Dec Offset Comparison")
ax2.legend()

# Plot Rotation for comparison
ax3.plot(rot_offset_reference, label="Hand-alignment", marker="o")
ax3.plot(rot_offset_pipeline, label="Pipeline", marker="x")
ax3.set_xlabel("OBS")
ax3.set_ylabel("Rotation Offset")
ax3.set_title("Rotation Offset Comparison")
ax3.legend()
plt.show()

