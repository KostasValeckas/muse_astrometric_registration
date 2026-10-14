"""
Translation estimation (rotation already removed, scale == 1) with a known
control offset, mirroring fixed_scale_rotation.py.

Pipeline per band-pass sigma:
  DoG -> Tukey window -> cross-power spectrum (optionally partly whitened)
  -> integer peak inside a search window around the control offset
  -> continuous refinement: maximise  C(t) = Re sum_k X(k) exp(2*pi*i k.t)
     directly (exact gradient, no resampling / upsampling grid).

The same is done for a *control* pair (image shifted by a known offset vs the
unshifted image).  The control error is the method's bias at that sigma and
offset; subtracting it gives the final estimate.

Convention of the returned shift (tx, ty), in pixels:
    scipy.ndimage.shift(reference, (ty, tx)) ~= image
i.e. image content sits at reference position + (tx, ty).  To register the
image onto the reference, shift it by (-tx, -ty).

Note: (tx, ty) is the residual after derotating about the array centre
((N-1)/2), the pivot used by scipy.ndimage.rotate(..., reshape=False).
"""
import numpy as np
from numpy.fft import fft2, ifft2, fftfreq
from scipy.ndimage import shift as ndi_shift
from scipy.optimize import minimize
from scipy.signal.windows import tukey

from fixed_scale_rotation import prep, dog, _render   # shared helpers


def tukey2d(shape, taper=0.25):
    return np.outer(tukey(shape[0], taper), tukey(shape[1], taper))


def cross_spectrum(A, B, gamma=0.5):
    """X = FA conj(FB) / |FA conj(FB)|^gamma.
    gamma=0: plain cross-correlation; gamma=1: pure phase correlation."""
    X = fft2(A) * np.conj(fft2(B))
    if gamma:
        mag = np.abs(X)
        X = X / (mag + 1e-8 * mag.max()) ** gamma
    return X


def integer_peak(X, expected, halfwidth):
    """Integer-lag peak of ifft2(X) inside |lag - expected| <= halfwidth.
    Returns (tx, ty) integer start point and a peak SNR (peak vs MAD)."""
    Ny, Nx = X.shape
    c = ifft2(X).real
    ly, lx = fftfreq(Ny, 1.0 / Ny), fftfreq(Nx, 1.0 / Nx)       # signed integer lags
    m = (np.abs(ly - expected[1]) <= halfwidth)[:, None] & \
        (np.abs(lx - expected[0]) <= halfwidth)[None, :]
    iy, ix = np.unravel_index(np.argmax(np.where(m, c, -np.inf)), c.shape)
    med = np.median(c)
    mad = 1.4826 * np.median(np.abs(c - med))
    return (lx[ix], ly[iy]), (c[iy, ix] - med) / mad, c[iy, ix]


def refine_peak(X, t0, scale):
    """Maximise the band-limited correlation C(t) continuously, starting at t0."""
    Ny, Nx = X.shape
    ky, kx = fftfreq(Ny), fftfreq(Nx)                           # cycles / pixel
    Xs = X / scale

    def neg_c(t):
        ey = np.exp(2j * np.pi * ky * t[1])
        ex = np.exp(2j * np.pi * kx * t[0])
        v = ey @ Xs @ ex
        gx = ey @ Xs @ (2j * np.pi * kx * ex)
        gy = (2j * np.pi * ky * ey) @ Xs @ ex
        return -v.real, -np.array([gx.real, gy.real])

    res = minimize(neg_c, np.asarray(t0, float), jac=True, method="BFGS",
                   options=dict(gtol=1e-9, maxiter=100))
    return res.x, bool(res.success) and np.all(np.abs(res.x - t0) < 1.0)


def estimate_translation(
    reference, image, control_offset=(10.0, 10.0), sigmas=None, ratio=1.6,
    gamma=0.0, taper=0.25, ctrl_tol=0.02, snr_min=20.0, search_halfwidth=5.0,
    min_ok=3, window_iters=2,
):
    """
    Returns dict(shift=(tx, ty), shift_err=(sx, sy), success, table, raw).

    control_offset  : known (dx, dy) applied to `image` for the control pair.
                      Must exceed search_halfwidth + peak width so the dx=dy=0
                      artefacts (window, DFT edge effects) stay outside the search.
                      Use dx != dy: equal offsets cannot reveal an x/y swap.
    search_halfwidth: largest |residual| (px) you are willing to accept.
    table columns   : sigma, tx, ty, ctrl_x, ctrl_y, eps_x, eps_y, snr_main, snr_ctrl, ok
    raw             : median uncorrected shift (main - nominal offset), for comparison.
    window_iters    : after the first estimate the reference window is moved onto the
                      same sky area as the image window and the estimate is redone.
                      The control cannot see this bias (control pair has identical
                      windows); it grows with the true shift and with sigma.
    """
    ref, img = prep(reference), prep(image)
    cx, cy = control_offset
    win = tukey2d(ref.shape, taper)
    # zero-pad so the known shift never crops content and the correlation never wraps
    pad = int(np.ceil(max(abs(cx), abs(cy)) + search_halfwidth)) + 2

    if sigmas is None:
        sigmas = np.geomspace(1.0, 4.0, 10)

    rows = []
    for s in sigmas:
        # DoG -> window -> pad, all in each image's OWN frame.  The window must
        # travel with the content: windowing *after* a shift pulls the estimate
        # toward zero by tens of milli-pixels.
        Dref, Dimg = dog(ref, s, ratio), dog(img, s, ratio)
        U = np.pad(Dimg * win, pad)                      # unshifted image
        A = ndi_shift(U, (cy, cx), order=3, mode="constant", cval=0.0)   # shifted image

        # control: A is an exact shifted copy of U -> truth is (cx, cy)
        Xc = cross_spectrum(A, U, gamma)
        t0c, snr_c, pk_c = integer_peak(Xc, (cx, cy), search_halfwidth)
        tc, ok_c = refine_peak(Xc, t0c, pk_c * Xc.size)

        # main: A vs reference.  Image content at p <-> reference content at p - eps,
        # so the reference window is the image window displaced by -eps.
        w_ref = win
        for _ in range(window_iters + 1):
            R = np.pad(Dref * w_ref, pad)
            Xm = cross_spectrum(A, R, gamma)
            t0m, snr_m, pk_m = integer_peak(Xm, (cx, cy), search_halfwidth)
            tm, ok_m = refine_peak(Xm, t0m, pk_m * Xm.size)
            e = tm - (cx, cy)
            w_ref = ndi_shift(win, (-e[1], -e[0]), order=3, mode="constant", cval=0.0)
        eps = tm - tc                            # bias-cancelled residual
        ok = (ok_m and ok_c and np.all(np.abs(tc - (cx, cy)) < ctrl_tol)
              and snr_m > snr_min and snr_c > snr_min)
        rows.append((s, *tm, *tc, *eps, snr_m, snr_c, float(ok)))

    table = np.array(rows, float)
    good = table[table[:, -1] == 1]
    raw = np.median(table[:, 1:3], axis=0) - (cx, cy)
    if len(good) >= min_ok:
        est = np.median(good[:, 5:7], axis=0)
        err = good[:, 5:7].std(axis=0, ddof=1)   # conservative: sigmas are correlated
        return dict(shift=tuple(est), shift_err=tuple(err), success=True,
                    table=table, raw=tuple(raw))
    return dict(shift=(np.nan, np.nan), shift_err=(np.nan, np.nan), success=False,
                table=table, raw=tuple(raw))


# --------------------------------------------------------------------------
# synthetic self-test:  python fixed_scale_translation.py
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    gamma = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    rng = np.random.default_rng(5)
    N, nstar = 300, 250
    xy = rng.uniform(-40, N + 40, (nstar, 2))
    flux = 10 ** rng.uniform(2.0, 4.2, nstar)
    ref = _render(xy, flux, 0.0, (0, 0), N, 1.6, rng)

    print(f"gamma={gamma}   (errors in pixels; 'raw' = no control correction)")
    print(f"{'true tx,ty':>16} | {'err x':>8} {'err y':>8} | {'raw x':>8} {'raw y':>8} | {'sig x':>7} {'sig y':>7} {'n_ok':>4}")
    for tx, ty in [(0.0, 0.0), (0.05, 0.01), (0.37, -0.62), (3.3, -2.1), (-1.234, 2.718)]:
        img = _render(xy, flux, 0.0, (tx, ty), N, 1.8, rng)     # worse seeing, new noise
        o = estimate_translation(ref, img, control_offset=(10.0, 10.0), gamma=gamma)
        n_ok = int(o["table"][:, -1].sum())
        print(f"{tx:7.3f},{ty:7.3f} | {o['shift'][0]-tx:8.4f} {o['shift'][1]-ty:8.4f} | "
              f"{o['raw'][0]-tx:8.4f} {o['raw'][1]-ty:8.4f} | "
              f"{o['shift_err'][0]:7.4f} {o['shift_err'][1]:7.4f} {n_ok:4d}")
    np.set_printoptions(precision=4, suppress=True, linewidth=140)
    print("\nper-sigma table (last case): sigma, tx, ty, ctrl_x, ctrl_y, eps_x, eps_y, snr_m, snr_c, ok")
    print(o["table"])
