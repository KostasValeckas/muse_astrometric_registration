import numpy as np
from astropy.io import fits
from astropy.table import Table

import matplotlib.pyplot as plt

path_pipeline = "output_table_NGC1365.fits"


path_all_rots = (
    "/home/kostas/Documents/ESO/muse_astrometric_registration/dev/all_rots.npy"
)

path_new_all_rots = "/home/kostas/Documents/ESO/muse_astrometric_registration/dev/test_claude_rot/all_rotations.npy"

# pipeline table
with fits.open(path_pipeline) as hdul:
    pipeline_table = Table(hdul[1].data)
    print(pipeline_table.colnames)


date_obs_pipeline = pipeline_table["DATE_OBS"]
ra_offset_pipeline = pipeline_table["RA_OFFSET"]
dec_offset_pipeline = pipeline_table["DEC_OFFSET"]
rot_offset_pipeline = pipeline_table["ROTANGLE"]


all_rots = np.load(path_new_all_rots, allow_pickle=True)[:-8]

all_rots = all_rots.astype(float)


# build nan mask buold on pipeline data
mask = (
    ~np.isnan(ra_offset_pipeline)
    & ~np.isnan(dec_offset_pipeline)
    & ~np.isnan(rot_offset_pipeline)
)


# and to the pipeline data
ra_offset_pipeline = ra_offset_pipeline[mask]
dec_offset_pipeline = dec_offset_pipeline[mask]
rot_offset_pipeline = rot_offset_pipeline[mask]


all_rots = all_rots[mask]


plt.plot(rot_offset_pipeline, "-o", label="Pipeline Rotations")
plt.plot(all_rots, "-o", label="FFT Rotations")
plt.xlabel("Index")
plt.ylabel("Rotation (degrees)")
plt.title("Comparison of Pipeline and FFT Rotations")
plt.legend()
plt.show()

plt.plot(rot_offset_pipeline - all_rots, "-o", label="Difference (Pipeline - FFT)")
plt.xlabel("Index")
plt.ylabel("Rotation Difference (degrees)")
plt.title("Difference between Pipeline and FFT Rotations")
plt.axhline(0, color="r", linestyle="--", label="Zero Difference")
plt.legend()
plt.show()
