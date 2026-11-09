#!/usr/bin/env python3
"""
1. Reproject the reference FITS image onto the WCS / pixel grid of the input
   FITS image.
2. Apply a known translation (pixels) and rotation (degrees, counter-clockwise
   about the image centre) to the input image.
3. Plot the difference image against the reprojected reference before and
   after the shift/rotation, together with histograms of the difference
   images.
 
Both images are read from HDU index 1.
 
Requirements
------------
    pip install numpy scipy matplotlib astropy reproject
"""
 
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp  # swap for reproject_exact if flux conservation matters
from scipy.ndimage import shift, rotate
 
# ---------------------------------------------------------------------------
# HARDCODE YOUR INPUTS HERE
# ---------------------------------------------------------------------------
REFERENCE_PATH = "/home/kostas/Documents/micro1/astro_align/environment/data/IMAGE_FOV_prealign_P01_WFI_BB_2018-01-10T02:49:23_0001.fits"  # reprojected onto the input's WCS/grid
INPUT_PATH = "/home/kostas/Documents/micro1/astro_align/environment/data/IMAGE_FOV_prealign_P01_WFI_BB_2018-01-10T02:49:23_0002.fits"          # the image that gets shifted/rotated
HDU_INDEX = 1
 
DX = -6.75       # shift in x (pixels)
DY = 3.63       # shift in y (pixels)
DTHETA = 0.12   # rotation (degrees, counter-clockwise about the image centre)
 
OUTPUT_PNG = "difference_image.png"
SHOW = True    # also display interactively
# ---------------------------------------------------------------------------
 
 
def load_image(path, hdu_index=HDU_INDEX):
    """Load a 2D image and its celestial WCS from the given HDU."""
    with fits.open(path) as hdul:
        hdu = hdul[hdu_index]
        data = np.squeeze(np.array(hdu.data, dtype=float))
        wcs = WCS(hdu.header).celestial
    if data.ndim != 2:
        raise ValueError(
            f"{path}: expected a 2D image in HDU {hdu_index}, got shape {data.shape}"
        )
    return data, wcs
 
 
def reproject_reference(reference, reference_wcs, input_wcs, input_shape):
    """Reproject the reference image onto the input image's WCS / pixel grid."""
    reprojected, footprint = reproject_interp(
        (reference, reference_wcs), input_wcs, shape_out=input_shape
    )
    reprojected[footprint == 0] = np.nan  # outside the reference's coverage
    return reprojected
 
 
def shift_image(image, dx, dy, dtheta):
    """
    Simple image shift and rotation.
 
    Uses scipy.ndimage.shift and scipy.ndimage.rotate.
 
    Parameters
    ----------
    image : np.2darray
        The image to be shifted and rotated.
    dx, dy : float
        Shift in the x / y direction (pixels).
    dtheta : float
        Rotation angle (degrees, counter-clockwise about the image centre).
 
    Returns
    -------
    np.2darray
        The shifted and rotated image.
    """
    # NaNs would spread through the interpolation; work on a copy with NaNs -> 0
    image = np.nan_to_num(np.copy(image), nan=0.0)
 
    # Shift the image (array axes are (y, x))
    shifted_image = shift(image, shift=(dy, dx), mode="nearest")
 
    # Rotate the image around its centre
    rotated_image = rotate(shifted_image, angle=dtheta, reshape=False, mode="nearest")
 
    return rotated_image
 
 
def plot_contour_overlay(
    ax, reference_frame, input_frame, title, n_levels=2, lower_per=50, upper_per=99
):
    """
    Overlay contour maps of two frames on a single axis.
 
    Contour levels are set per image from percentiles of its own pixel
    distribution, so the same structures are traced in both images.
    Reference contours are red, input contours are blue.
    """
    for frame, color in ((reference_frame, "red"), (input_frame, "blue")):
        if not np.any(np.isfinite(frame)):
            continue
        levels = np.unique(
            np.nanpercentile(frame, np.linspace(lower_per, upper_per, n_levels))
        )
        if len(levels) < 2:
            # Flat or degenerate image, nothing sensible to contour
            continue
        ax.contour(frame, levels=levels, colors=color, linewidths=0.8)
 
    # contour() uses a lower-left origin; flip to match imshow orientation
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel("X Pixel")
    ax.set_ylabel("Y Pixel")
 
 
def _hist_stats_label(name, values):
    """Legend label with simple summary statistics of the difference values."""
    return (
        f"{name}\n"
        f"mean={np.mean(values):.3g}, median={np.median(values):.3g}, "
        f"std={np.std(values):.3g}"
    )
 
 
def plot_difference_histograms(
    ax_prior,
    ax_post,
    ax_overlay,
    diff_prior,
    diff_post,
    n_bins=100,
    lower_per=1,
    upper_per=99,
):
    """
    Plot histograms of the difference images before and after alignment.
 
    Both histograms share the same bins (set from the combined
    `lower_per`-`upper_per` percentile range) so they can be compared directly.
    NaN / inf pixels are ignored. The statistics in the legends are computed
    over all finite pixels, not just the displayed range.
    """
    vals_prior = diff_prior[np.isfinite(diff_prior)]
    vals_post = diff_post[np.isfinite(diff_post)]
 
    lo, hi = np.percentile(
        np.concatenate([vals_prior, vals_post]), [lower_per, upper_per]
    )
    bins = np.linspace(lo, hi, n_bins + 1)
 
    label_prior = _hist_stats_label("Before alignment", vals_prior)
    label_post = _hist_stats_label("After alignment", vals_post)
 
    ax_prior.hist(vals_prior, bins=bins, color="tab:orange", alpha=0.8, label=label_prior)
    ax_prior.set_title("Difference Histogram Before Alignment")
 
    ax_post.hist(vals_post, bins=bins, color="tab:green", alpha=0.8, label=label_post)
    ax_post.set_title("Difference Histogram After Alignment")
 
    ax_overlay.hist(
        vals_prior, bins=bins, histtype="stepfilled", color="tab:orange",
        alpha=0.5, label=label_prior,
    )
    ax_overlay.hist(
        vals_post, bins=bins, histtype="stepfilled", color="tab:green",
        alpha=0.5, label=label_post,
    )
    ax_overlay.set_title("Difference Histograms Overlaid")
 
    for ax in (ax_prior, ax_post, ax_overlay):
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Difference (Reference - Input, normalized)")
        ax.set_ylabel("Pixel count")
        ax.set_xlim(lo, hi)
        ax.legend(fontsize=9)
 
 
def plot_difference_image(
    reference_frame,
    input_frame,
    dx,
    dy,
    dtheta,
    output_path=OUTPUT_PNG,
    show=False,
    custom_title=None,
    v_lower_per=15,
    v_upper_per=85,
):
    """
    Plot the difference image (and its histogram) before and after alignment.
 
    Parameters
    ----------
    reference_frame : np.2darray
        The reference image, already reprojected onto the input's grid.
    input_frame : np.2darray
        The target image (same shape as the reprojected reference).
    dx, dy : float
        Shift (pixels).
    dtheta : float
        Rotation (degrees, counter-clockwise about the centre).
    """
    # normalize both images to unity
    reference_frame = reference_frame / np.nanmax(reference_frame)
    input_frame = input_frame / np.nanmax(input_frame)
 
    aligned_frame = shift_image(input_frame, dx, dy, dtheta)
 
    diff_prior = reference_frame - input_frame
    diff_post = reference_frame - aligned_frame
 
    _, axes = plt.subplots(3, 3, figsize=(18, 18))
 
    ref_vmin, ref_vmax = np.nanpercentile(reference_frame, [v_lower_per, v_upper_per])
    in_vmin, in_vmax = np.nanpercentile(input_frame, [v_lower_per, v_upper_per])
 
    axes[0, 0].imshow(reference_frame, cmap="Reds", vmin=ref_vmin, vmax=ref_vmax)
    axes[0, 0].imshow(input_frame, cmap="Blues", vmin=in_vmin, vmax=in_vmax, alpha=0.5)
    axes[0, 0].set_title("Before Alignment (Red = Reference, Blue = Input)")
    axes[0, 0].set_xlabel("X Pixel")
    axes[0, 0].set_ylabel("Y Pixel")
 
    axes[0, 1].imshow(reference_frame, cmap="Reds", vmin=ref_vmin, vmax=ref_vmax)
    axes[0, 1].imshow(
        aligned_frame, cmap="Blues", vmin=in_vmin, vmax=in_vmax, alpha=0.5
    )
    axes[0, 1].set_title("After Alignment (Red = Reference, Blue = Aligned)")
    axes[0, 1].set_xlabel("X Pixel")
    axes[0, 1].set_ylabel("Y Pixel")
 
    axes[1, 0].imshow(
        diff_prior,
        cmap="gray",
        vmin=np.nanpercentile(diff_prior, v_lower_per),
        vmax=np.nanpercentile(diff_prior, v_upper_per),
    )
    axes[1, 0].set_title("Difference Before Alignment")
    axes[1, 0].set_xlabel("X Pixel")
    axes[1, 0].set_ylabel("Y Pixel")
 
    axes[1, 1].imshow(
        diff_post,
        cmap="gray",
        vmin=np.nanpercentile(diff_post, v_lower_per),
        vmax=np.nanpercentile(diff_post, v_upper_per),
    )
    axes[1, 1].set_title("Difference After Alignment")
    axes[1, 1].set_xlabel("X Pixel")
    axes[1, 1].set_ylabel("Y Pixel")
 
    # Third column: overlayed contour maps
    plot_contour_overlay(
        axes[0, 2],
        reference_frame,
        input_frame,
        "Contours Before Alignment (Red = Reference, Blue = Input)",
    )
    plot_contour_overlay(
        axes[1, 2],
        reference_frame,
        aligned_frame,
        "Contours After Alignment (Red = Reference, Blue = Aligned)",
    )
 
    # Image panels keep equal aspect; the histogram row does not
    for ax in axes[:2].ravel():
        ax.set_aspect("equal")
 
    # Third row: histograms of the difference images
    plot_difference_histograms(
        axes[2, 0], axes[2, 1], axes[2, 2], diff_prior, diff_post
    )
 
    offsets = f"Offsets: dx={dx:.2f}, dy={dy:.2f}, dtheta={dtheta:.2f} deg"
    if custom_title is not None:
        plt.suptitle(f"{custom_title}\n{offsets}")
    else:
        plt.suptitle(f"Difference Image\n{offsets}")
 
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Saved difference image to {output_path}")
 
    if show:
        plt.show()
    else:
        plt.close()
 
 
def main():
    reference, ref_wcs = load_image(REFERENCE_PATH)
    input_img, in_wcs = load_image(INPUT_PATH)
 
    print(f"Reference shape: {reference.shape}, input shape: {input_img.shape}")
 
    # Step 1: put the reference on the input image's WCS / pixel grid
    reference_on_input = reproject_reference(
        reference, ref_wcs, in_wcs, input_img.shape
    )
 
    # Step 2 + 3: shift/rotate the input and plot the differences
    plot_difference_image(
        reference_on_input,
        input_img,
        DX,
        DY,
        DTHETA,
        output_path=OUTPUT_PNG,
        show=SHOW,
        custom_title=f"Reference: {REFERENCE_PATH}\nInput: {INPUT_PATH}",
    )
 
 
if __name__ == "__main__":
    main()
 