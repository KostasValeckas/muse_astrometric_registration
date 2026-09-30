import numpy as np
from astropy.io import fits
from astropy.table import Table

import matplotlib.pyplot as plt

path_reference = "NGC1365_offset_table_DR2.1.fits"
path_pipeline = "output_table.fits"


# reference table

with fits.open(path_reference) as hdul:
    reference_table = Table(hdul[1].data)

# pipeline table
with fits.open(path_pipeline) as hdul:
    pipeline_table = Table(hdul[1].data)
    print(pipeline_table.colnames)

date_obs_reference = reference_table["DATE_OBS"]
ra_offset_reference = reference_table["RA_OFFSET"]
dec_offset_reference = reference_table["DEC_OFFSET"]
rot_offset_reference = reference_table["ROTANGLE"]



date_obs_pipeline = pipeline_table["DATE_OBS"]
ra_offset_pipeline = pipeline_table["RA_OFFSET"]
dec_offset_pipeline = pipeline_table["DEC_OFFSET"]
rot_offset_pipeline = pipeline_table["ROTANGLE"]



fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# Plot RA for comparison
ax1.plot(ra_offset_reference, label="Target", marker="o")
ax1.plot(ra_offset_pipeline, label="Pipeline", marker="x")
ax1.set_xlabel("DATE_OBS")
ax1.set_ylabel("RA Offset")
ax1.set_title("RA Offset Comparison")
ax1.legend()

# Plot Dec for comparison
ax2.plot(dec_offset_reference, label="Target", marker="o")
ax2.plot(dec_offset_pipeline, label="Pipeline", marker="x")
ax2.set_xlabel("DATE_OBS")
ax2.set_ylabel("Dec Offset")
ax2.set_title("Dec Offset Comparison")
ax2.legend()

# Plot Rotation for comparison
ax3.plot(rot_offset_reference, label="Target", marker="o")
ax3.plot(rot_offset_pipeline, label="Pipeline", marker="x")
ax3.set_xlabel("DATE_OBS")
ax3.set_ylabel("Rotation Offset")
ax3.set_title("Rotation Offset Comparison")
ax3.legend()
plt.show()

