"""
Rotation estimation for scale == 1, from DFT power spectra.
 
Replaces the log-polar / 2D phase_cross_correlation step.  Because scale is
known, the radial axis carries no shift, so we:
 
  * sample the spectrum on a *linear* polar grid (rings of constant |f|),
  * cross-correlate ring-by-ring along theta only and sum over rings,
  * read the angle off a 1D, trigonometrically-interpolated correlation.
 
Angle convention of the returned value: scipy.ndimage.rotate(reference, angle)
~= image   (i.e. `angle` is what you must apply to `reference` to get `image`).
"""
import numpy as np
from numpy.fft import fft2, fftshift, rfft, irfft
from scipy.ndimage import gaussian_filter, map_coordinates, rotate
 
 
# --------------------------------------------------------------------------
# pre-processing
# --------------------------------------------------------------------------
def radial_window(shape, taper=0.4):
    """Rotation-invariant (disk) Tukey window, centred on (N-1)/2 like
    scipy.ndimage.rotate.  Unlike a separable Hann it has no frame-fixed
    cross in its spectrum, and a disk inscribed in the frame never sees the
    NaN/empty corners that rotation creates."""
    ny, nx = shape
    y, x = np.indices(shape)
    r = np.hypot(y - (ny - 1) / 2, x - (nx - 1) / 2)
    r_out = min(ny, nx) / 2
    r_in = r_out * (1 - taper)
    w = np.ones(shape)
    ramp = (r > r_in) & (r < r_out)
    w[ramp] = 0.5 * (1 + np.cos(np.pi * (r[ramp] - r_in) / (r_out - r_in)))
    w[r >= r_out] = 0.0
    return w
 
 
def dog(img, sigma, ratio=1.6):
    """DoG with a fixed sigma *ratio*: sigma sets the band centre, ratio the
    (fractional) bandwidth.  Absolute sigma differences only scale amplitude."""
    return (gaussian_filter(img, sigma, mode="nearest")
            - gaussian_filter(img, ratio * sigma, mode="nearest"))
 
 
def dog_passband(sigma, ratio=1.6, level=0.2):
    """Frequency range (cycles/pixel) where the DoG response >= level * peak."""
    f = np.linspace(1e-4, 0.5, 4000)
    h = np.exp(-2 * (np.pi * sigma * f) ** 2) - np.exp(-2 * (np.pi * ratio * sigma * f) ** 2)
    keep = f[h >= level * h.max()]
    return keep[0], keep[-1]
 
 
def prep(img):
    """Fill NaNs with the median (not 0) and remove the background level."""
    img = np.asarray(img, dtype=float)
    med = np.nanmedian(img)
    return np.where(np.isfinite(img), img, med) - med
 
 
# --------------------------------------------------------------------------
# spectrum -> rings -> angular correlation
# --------------------------------------------------------------------------
def polar_rings(img, f_lo, f_hi, n_theta, pad=2):
    """Whitened rings of sqrt(power spectrum) on a linear (|f|, theta) grid.
 
    * theta covers [0, 180 deg) only (power spectrum is point-symmetric),
      so the angular axis is exactly periodic with period 180 deg.
    * 2x zero padding makes |F|^2 (FT of the autocorrelation) essentially
      band-limited, so cubic interpolation of it is accurate.
    * Each ring is mean-subtracted and unit-variance so that the steep radial
      power-law of the spectrum cannot dominate the correlation.
    """
    ny, nx = img.shape
    P = pad * max(ny, nx)
    pw = np.abs(fftshift(fft2(img, s=(P, P)))) ** 2
    c = P // 2                                  # DC bin after fftshift
    n_r = max(8, int(round((f_hi - f_lo) * P)))
    f = np.linspace(f_lo, f_hi, n_r)
    theta = np.arange(n_theta) * (np.pi / n_theta)
    rr = (f * P)[:, None]
    rows = c + rr * np.sin(theta)[None, :]
    cols = c + rr * np.cos(theta)[None, :]
    ring = map_coordinates(pw, [rows, cols], order=3, mode="nearest")
    ring = np.sqrt(np.clip(ring, 0, None))
    ring -= ring.mean(axis=1, keepdims=True)
    sd = ring.std(axis=1, keepdims=True)
    ring /= np.where(sd > 0, sd, 1.0)
    return ring, f
 
 
def angular_shift(A, B, w, expected=0.0, halfwidth=None, upsample=64):
    """Shift m (deg) such that A(theta) ~ B(theta - m), plus a peak SNR.
 
    C(m) = sum_r w_r sum_t A_r[t+m] B_r[t]  computed with an FFT along theta
    only; zero-padding the angular spectrum gives *exact* band-limited
    interpolation (no parabola/centroid bias).
    """
    n = A.shape[1]
    cross = (w[:, None] * rfft(A, axis=1) * np.conj(rfft(B, axis=1))).sum(axis=0)
    if n % 2 == 0:
        cross[-1] = 0.0                          # Nyquist term is ambiguous
    c = irfft(cross, n=n * upsample)
    ang = (np.arange(c.size) * (180.0 / c.size) + 90.0) % 180.0 - 90.0
    if halfwidth is not None:
        d = (ang - expected + 90.0) % 180.0 - 90.0
        cs = np.where(np.abs(d) <= halfwidth, c, -np.inf)
    else:
        cs = c
    k = int(np.argmax(cs))
    med = np.median(c)
    mad = 1.4826 * np.median(np.abs(c - med))
    return ang[k], (c[k] - med) / mad
 
 
# --------------------------------------------------------------------------
# driver (replaces the while-loop in run())
# --------------------------------------------------------------------------
def estimate_rotation_fixed_scale(
    reference, image, extra_rotation=6.0, sigmas=None, ratio=1.6,
    ctrl_tol=0.03, snr_min=6.0, search_halfwidth=3.0, min_ok=3,
    pad=2, upsample=64,
):
    """
    Returns dict(rot, rot_err, success, table).
 
    rot  : degrees, scipy convention: rotate(reference, rot) ~ image.
    table: per-sigma rows (sigma, a_main, a_ctrl, eps, snr_main, snr_ctrl, ok).
 
    `extra_rotation` should be larger than search_halfwidth + peak width so the
    true peak is separated from frame-fixed features (sensor striping, pixel-grid
    anisotropy) that all pile up at dtheta = 0.
    """
    ref, img = prep(reference), prep(image)
    L = max(ref.shape)
    win = radial_window(ref.shape)
 
    # control image: the *same* data rotated by a known angle (spline, no NaNs
    # inside the inscribed disk)
    img_rot = rotate(img, extra_rotation, reshape=False, order=3,
                     mode="constant", cval=0.0)
 
    if sigmas is None:                           # geometric: band centre ~ 1/sigma
        sigmas = np.geomspace(1.0, 8.0, 16)
 
    # my polar parameterisation rotates opposite to scipy's, hence the minus
    S = -1.0
    rows, ok_eps = [], []
    for s in sigmas:
        f_lo, f_hi = dog_passband(s, ratio)
        n_theta = max(128, 2 * int(2 * np.pi * f_hi * L))     # even
        rings = {}
        for name, im in (("ref", ref), ("img", img), ("rot", img_rot)):
            rings[name], f = polar_rings(dog(im, s, ratio) * win,
                                         f_lo, f_hi, n_theta, pad)
        w = f / f.sum()                                        # ring information ~ |f|
 
        m_main, snr_main = angular_shift(rings["rot"], rings["ref"], w,
                                         S * extra_rotation, search_halfwidth, upsample)
        m_ctrl, snr_ctrl = angular_shift(rings["rot"], rings["img"], w,
                                         S * extra_rotation, search_halfwidth, upsample)
        a_main, a_ctrl = S * m_main, S * m_ctrl
        eps = a_main - a_ctrl                                  # bias-cancelled
        ok = (abs(a_ctrl - extra_rotation) < ctrl_tol
              and snr_main > snr_min and snr_ctrl > snr_min)
        rows.append((s, a_main, a_ctrl, eps, snr_main, snr_ctrl, ok))
        if ok:
            ok_eps.append(eps)
 
    table = np.array(rows, dtype=float)
    if len(ok_eps) >= min_ok:
        rot = float(np.median(ok_eps))
        err = float(np.std(ok_eps, ddof=1))                    # conservative: sigmas correlated
        return dict(rot=rot, rot_err=err, success=True, table=table)
    return dict(rot=np.nan, rot_err=np.nan, success=False, table=table)
 
 
# --------------------------------------------------------------------------
# synthetic self-test:  python fixed_scale_rotation.py
# --------------------------------------------------------------------------
def _render(xy, flux, rot_deg, shift, N, psf, rng, noise=4.0):
    """Analytic star field; rot_deg in scipy's convention (about (N-1)/2)."""
    c = (N - 1) / 2
    phi = -np.deg2rad(rot_deg)       # scipy +angle == my -phi (checked numerically)
    x, y = xy[:, 0] - c, xy[:, 1] - c
    xr = np.cos(phi) * x - np.sin(phi) * y + c + shift[0]
    yr = np.sin(phi) * x + np.cos(phi) * y + c + shift[1]
    img = np.full((N, N), 50.0)
    for xi, yi, fl in zip(xr, yr, flux):
        x0, x1 = int(max(0, xi - 9)), int(min(N, xi + 10))
        y0, y1 = int(max(0, yi - 9)), int(min(N, yi + 10))
        if x1 <= x0 or y1 <= y0:
            continue
        gx = np.arange(x0, x1)[None, :]
        gy = np.arange(y0, y1)[:, None]
        img[y0:y1, x0:x1] += fl / (2 * np.pi * psf ** 2) * np.exp(
            -((gx - xi) ** 2 + (gy - yi) ** 2) / (2 * psf ** 2))
    return img + rng.normal(0, noise, img.shape)
 
 
if __name__ == "__main__":
    rng = np.random.default_rng(3)
    N, nstar = 300, 250
    xy = rng.uniform(-40, N + 40, (nstar, 2))
    flux = 10 ** rng.uniform(2.0, 4.2, nstar)
    ref = _render(xy, flux, 0.0, (0, 0), N, 1.6, rng)
 
    print(f"{'true':>8} {'rot':>9} {'err':>8} {'sig':>7} {'n_ok':>5}")
    for true in (0.0, 0.05, -0.13, 0.31, 0.8):
        img = _render(xy, flux, true, (3.3, -2.1), N, 1.8, rng)   # shifted + worse seeing
        out = estimate_rotation_fixed_scale(ref, img, extra_rotation=6.0)
        n_ok = int(out["table"][:, -1].sum())
        print(f"{true:8.3f} {out['rot']:9.4f} {out['rot'] - true:8.4f} "
              f"{out['rot_err']:7.4f} {n_ok:5d}")
    print("\nper-sigma table for last case: sigma, a_main, a_ctrl, eps, snr_main, snr_ctrl, ok")
    np.set_printoptions(precision=4, suppress=True)
    print(out["table"])
