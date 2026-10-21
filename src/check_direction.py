"""
check_direction.py - independent check of the retrieved wind direction against the SAR image itself.

For every sea patch the orientation of the wind streaks is measured directly with a 2D FFT
(0.8-2.5 km band, no machine learning) and compared with the ResNet axis and the reference wind axis:
  - ResNet ~ FFT, both offset from the reference by a similar angle -> the processing is correct; the streaks
    in the image themselves differ from the reference wind.
  - ResNet differs from FFT -> possible problem in the network or the processing.
The UTM grid convergence (grid north vs true north) is reported as well.

    python check_direction.py --run_dir .../api_runs/<acquisition>_<box>
    python check_direction.py --run_dir .../scene_validation --tif_dir .../tifs/<scene>
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

import build_dataset as BD
import texture_check as TC
import train_resnet as T


def convergence_deg(lon, lat, crs):
    """Angle between UTM grid north and true north (deg)."""
    zone = int(str(crs)[-2:]); lon0 = -183 + 6 * zone
    return float((lon - lon0) * np.sin(np.radians(lat)))


def reference_column(W):
    """Reference wind direction column (ref_dir; cop_dir in files written by older versions)."""
    return "ref_dir" if "ref_dir" in W.columns else "cop_dir"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True); ap.add_argument("--tif_dir", default=None)
    ap.add_argument("--only_good", action="store_true")
    a = ap.parse_args(argv)
    tif_dir = a.tif_dir or os.path.join(a.run_dir, "tifs")
    tifs = {os.path.basename(p): p for p in BD.glob_tifs(tif_dir)}
    is_land = BD.land_mask_fn(True)
    kmag, kbear = TC.kgrid(BD.PATCH_PX)
    rows = []
    for csv in sorted(glob.glob(os.path.join(a.run_dir, "wind_field_*.csv"))):
        W = pd.read_csv(csv); W = W[W.status == "kept"]
        if a.only_good:
            W = W[W.quality == "good"]
        ref_col = reference_column(W)
        Z, keep = [], []
        for _, r in W.iterrows():
            name = next((n for n in tifs if n.startswith(r.patch_id + "_") and r.scene_id.split("_", 1)[1] in n), None)
            if name is None:
                continue
            vv, inc, azi, lon, lat, _ = BD.read_patch_tif(tifs[name])
            info, z, valid = BD.process_patch(vv, inc, azi, is_land(lat, lon), r.uid)
            if z is None:
                continue
            Z.append(z); keep.append(r)
            crs = __import__("rasterio").open(tifs[name]).crs.to_string()
        if not Z:
            continue
        K = pd.DataFrame(keep); P = TC.spectra(np.stack(Z).astype(float))
        fft_patch, aniso, _ = TC.band_stats(P, kmag, kbear, 8.0, 25.0)
        fft_scene, an_s, _ = TC.band_stats(P.mean(0), kmag, kbear, 8.0, 25.0)
        ref, res = K[ref_col].values % 180, K.axis_resnet.values % 180
        ref_ax, res_ax = T.axial_mean(ref), T.axial_mean(res)
        ae = T.alias_err
        rows.append(dict(scene=os.path.basename(csv)[11:-4], n=len(K), reference_axis=ref_ax, fft_axis=float(fft_scene),
                         resnet_axis=res_ax, fft_minus_ref=float(ae(fft_scene, ref_ax)), resnet_minus_ref=float(ae(res_ax, ref_ax)),
                         resnet_minus_fft=float(ae(res_ax, fft_scene)),
                         patch_median_resnet_minus_fft=float(np.median(ae(res, fft_patch)[aniso >= 0.2])) if (aniso >= 0.2).any() else np.nan,
                         grid_convergence_deg=convergence_deg(K.lon.mean(), K.lat.mean(), crs), scene_aniso=float(an_s)))
    R = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(R.round(1).to_string(index=False))
    for _, r in R.iterrows():
        if r.scene_aniso < 0.15:
            v = (f"very weak texture (anisotropy {r.scene_aniso:.2f}): the image shows almost no wind streaks, so neither the FFT "
                 f"nor the ResNet scene average is reliable and this check is not conclusive.")
        elif abs(r.resnet_minus_fft) <= 8:
            v = (f"ResNet follows the image texture ({r.resnet_minus_fft:+.1f} deg). The streaks themselves differ from the reference "
                 f"by {r.fft_minus_ref:+.1f} deg, so the difference comes from the image vs the reference, not from the processing.")
        else:
            v = f"ResNet differs from the FFT texture axis by {r.resnet_minus_fft:+.1f} deg; needs inspection."
        print(f"\n{r.scene}: {v}  (UTM grid convergence {r.grid_convergence_deg:+.2f} deg)")
    R.to_csv(os.path.join(a.run_dir, "check_direction.csv"), index=False)
    return R


if __name__ == "__main__":
    main()
