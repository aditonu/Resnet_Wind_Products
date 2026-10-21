"""
texture_check.py - dominant orientation of the SAR texture of each scene from the 2D FFT, without the network.

  LONG band  (default 0.8-2.5 km wavelength): wind streaks / boundary-layer rolls, parallel to the wind
  SHORT band (default 220-600 m wavelength):  ocean swell, not necessarily related to the local wind

Per scene:
  * stripe axis of each band (bearing mod 180) compared with the Copernicus wind axis
  * anisotropy (0 = no preferred direction / speckle only, 1 = a single direction)
  * contrast (band power / power of the smallest scales; ~1 = speckle only)
  * with --cv_pred: whether the network axis follows the LONG band (wind streaks) or the SHORT band (swell)

Outputs (out_dir): texture_by_scene.csv, texture_spectra.png (2D spectrum per scene, north-up; red line = expected
energy of wind-parallel streaks, blue dashed = network axis)

  python texture_check.py --npz .../stageA/dataset_A.npz --out_dir .../cv_results [--cv_pred .../cv_predictions.csv]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

PX_M = 100.0   # pixel size (m)


def alias_err(p, t):
    return (np.asarray(p, float) - np.asarray(t, float) + 90.0) % 180.0 - 90.0


def axial_mean(a, w=None):
    t = np.radians(2 * np.asarray(a, float))
    w = np.ones_like(t) if w is None else np.asarray(w, float)
    return float((0.5 * np.degrees(np.arctan2((w * np.sin(t)).sum(), (w * np.cos(t)).sum()))) % 180)


def spectra(X):
    """X (N,n,n) z-score patches -> power spectra (N,n,n), Hann-windowed and fft-shifted."""
    n = X.shape[-1]
    w = np.hanning(n)
    Xw = (X - X.mean(axis=(-2, -1), keepdims=True)) * np.outer(w, w)
    return np.abs(np.fft.fftshift(np.fft.fft2(Xw), axes=(-2, -1))) ** 2


def kgrid(n):
    """|k| (cycles/pixel) and wavevector bearing (deg clockwise from north); rows = south (+), cols = east (+)."""
    f = np.fft.fftshift(np.fft.fftfreq(n))
    ky, kx = np.meshgrid(f, f, indexing="ij")
    return np.hypot(kx, ky), np.degrees(np.arctan2(kx, -ky)) % 360.0


def band_stats(P, kmag, kbear, lam_min_px, lam_max_px, noise_kmin=0.42):
    """P (...,n,n) -> stripe_axis (deg mod 180), anisotropy (0..1), contrast (band power / speckle-floor power)."""
    m = (kmag >= 1.0 / lam_max_px) & (kmag <= 1.0 / lam_min_px)
    nm = kmag >= noise_kmin
    w = P[..., m]
    phi = np.radians(2 * kbear[m])
    C, S, tot = (w * np.cos(phi)).sum(-1), (w * np.sin(phi)).sum(-1), w.sum(-1)
    k_axis = (0.5 * np.degrees(np.arctan2(S, C))) % 180.0
    stripe_axis = (k_axis + 90.0) % 180.0           # stripes / crests are perpendicular to the wavevector
    aniso = np.hypot(C, S) / np.maximum(tot, 1e-12)
    contrast = w.mean(-1) / np.maximum(P[..., nm].mean(-1), 1e-12)
    return stripe_axis, aniso, contrast


def peak_wavelength_m(P, kmag, lam_min_px, lam_max_px):
    m = (kmag >= 1.0 / lam_max_px) & (kmag <= 1.0 / lam_min_px)
    return float(PX_M / kmag[m][np.argmax(P[m])])


def analyse(X, th, folder, bands, pred_by_scene=None):
    n = X.shape[-1]
    kmag, kbear = kgrid(n)
    rows, mean_spec = [], {}
    for sc in sorted(set(folder)):
        m = folder == sc
        P = spectra(X[m])
        Pm = P.mean(0)
        mean_spec[sc] = Pm
        wind_axis = axial_mean(th[m] % 180)
        r = dict(scene=sc, n=int(m.sum()), wind_axis=wind_axis)
        for name, (lo, hi) in bands.items():
            ax_s, an_s, ct_s = band_stats(Pm, kmag, kbear, lo, hi)            # scene-mean spectrum
            ax_p, an_p, _ = band_stats(P, kmag, kbear, lo, hi)                # har patch
            e_p = np.abs(alias_err(ax_p, th[m] % 180))
            r.update({f"{name}_axis": float(ax_s), f"{name}_err_vs_wind": float(alias_err(ax_s, wind_axis)),
                      f"{name}_aniso": float(an_s), f"{name}_contrast": float(ct_s),
                      f"{name}_patch_within20_pct": float(100 * (e_p <= 20).mean()),
                      f"{name}_peak_lambda_m": peak_wavelength_m(Pm, kmag, lo, hi)})
        if pred_by_scene is not None and sc in pred_by_scene:
            p = pred_by_scene[sc]
            r.update(resnet_axis=p, resnet_err_vs_wind=float(alias_err(p, wind_axis)),
                     **{f"resnet_minus_{b}": float(alias_err(p, r[f"{b}_axis"])) for b in bands})
        rows.append(r)
    return pd.DataFrame(rows), mean_spec


def verdict(r, bands=("long", "short")):
    if "resnet_axis" not in r or pd.isna(r.get("resnet_axis")):
        return ""
    if max(r[f"{b}_aniso"] for b in bands) < 0.1:
        return "weak texture (mostly speckle): direction hard to retrieve by any method"
    d = {b: abs(r[f"resnet_minus_{b}"]) for b in bands}
    best = min(d, key=d.get)
    if abs(r["resnet_err_vs_wind"]) <= 20:
        return "ResNet ~ wind (ok)"
    if d[best] <= 20:
        return f"ResNet follows the {best.upper()}-band texture, which is {abs(r[f'{best}_err_vs_wind']):.0f} deg from the wind"
    return "ResNet follows neither the wind nor a texture band (unclear)"


def plot(mean_spec, T, out_png, bands):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    scenes = list(T.scene)
    n = next(iter(mean_spec.values())).shape[0]; c = n // 2
    cols = 5; rows = int(np.ceil(len(scenes) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.9), squeeze=False)
    for a in axs.ravel(): a.axis("off")
    kk = np.fft.fftshift(np.fft.fftfreq(n))
    for a, (_, r) in zip(axs.ravel(), T.iterrows()):
        Pm = mean_spec[r.scene]
        a.imshow(np.log10(Pm + 1e-9), cmap="magma", extent=[kk[0], kk[-1], kk[-1], kk[0]])
        L = 0.5
        def line(bearing, **kw):
            b = np.radians(bearing)
            a.plot([-L * np.sin(b), L * np.sin(b)], [L * np.cos(b), -L * np.cos(b)], **kw)
        line(r.wind_axis + 90, color="red", lw=1)                     # expected energy of wind-parallel streaks
        if "resnet_axis" in r and pd.notna(r.get("resnet_axis")):
            line(r.resnet_axis + 90, color="cyan", lw=1, ls="--")
        for lo, hi in bands.values():
            for lam in (lo, hi):
                a.add_patch(plt.Circle((0, 0), 1.0 / lam, fill=False, color="white", lw=0.4, alpha=0.6))
        a.set_xlim(-0.5, 0.5); a.set_ylim(0.5, -0.5)
        a.set_title(f"{r.scene}\nwind axis {r.wind_axis:.0f}", fontsize=7)
    fig.suptitle("Scene-mean 2D spectrum (north-up). red = wind-parallel streaks expected, cyan dashed = ResNet", fontsize=9)
    plt.tight_layout(); plt.savefig(out_png, dpi=130); plt.close()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True); ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cv_pred", default=None, help="cv_predictions.csv of train_resnet_v2 (optional)")
    ap.add_argument("--min_speed", type=float, default=3.0)
    ap.add_argument("--long_m", default="800,2500"); ap.add_argument("--short_m", default="220,600")
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    d = np.load(a.npz, allow_pickle=False)
    th, sp = d["theta_from"].astype(float), d["wind_speed"].astype(float)
    ok = np.isfinite(th) & np.isfinite(sp) & (sp >= a.min_speed)
    X, th, folder = d["X"][ok].astype(np.float64), th[ok], np.array([str(f) for f in d["folder"][ok]])
    to_px = lambda s: tuple(float(v) / PX_M for v in s.split(","))
    bands = {"long": to_px(a.long_m), "short": to_px(a.short_m)}
    pred = None
    if a.cv_pred:
        cp = pd.read_csv(a.cv_pred)
        pred = {str(s): axial_mean(g.pred_axis_ens.values) for s, g in cp.groupby("scene")}
    T, spec = analyse(X, th, folder, bands, pred)
    T["verdict"] = [verdict(r) for _, r in T.iterrows()]
    T = T.sort_values("wind_axis").reset_index(drop=True)
    T.to_csv(os.path.join(a.out_dir, "texture_by_scene.csv"), index=False)
    cols = ["scene", "n", "wind_axis", "long_axis", "long_err_vs_wind", "long_aniso", "long_contrast",
            "short_axis", "short_err_vs_wind", "short_aniso", "short_contrast", "short_peak_lambda_m"]
    if pred is not None:
        cols += ["resnet_axis", "resnet_err_vs_wind", "resnet_minus_long", "resnet_minus_short"]
    pd.set_option("display.width", 250)
    print(T[cols].round(2).to_string(index=False))
    if pred is not None:
        print("\nVERDICT:")
        for _, r in T.iterrows():
            print(f"  {r.scene:20s} {r.verdict}")
    try:
        plot(spec, T, os.path.join(a.out_dir, "texture_spectra.png"), bands)
    except Exception as e:
        print("plot skipped:", e)
    print("\nsaved:", os.path.join(a.out_dir, "texture_by_scene.csv"))
    return T


if __name__ == "__main__":
    main()
