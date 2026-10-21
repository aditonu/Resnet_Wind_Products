"""
build_dataset.py - training dataset from the Earth Engine patch GeoTIFFs (images + wind labels).

Input
  A folder tree of patch GeoTIFFs (one sub-folder per scene, e.g. tifs/Gujarat/*.tif) named
      <patch>_<S1A|S1B|S1C|S1D>_<YYYYMMDD>_<ASC|DSC>_RO<orbit>_<HHMMSS>.tif   e.g. C001_S1A_20240305_DSC_RO092_003133.tif
  Each file: 50 x 50 pixels, 3 float32 bands (vv_db, incidence_deg, radar_azimuth_deg), no-data = -9999.

Processing (per patch)
  1. Acquisition date and time (UTC) from the file name.
  2. -9999 / NaN pixels are treated as missing.
  3. Land mask from the pixel-centre coordinates (global-land-mask, ~1 km).
  4. Patches with more than 25 % missing or land pixels are dropped; otherwise those pixels are filled with
     the nearest sea pixel plus Gaussian noise (as in Zanchetta & Zecchetto, 2021).
  5. Z-score normalisation with the mean and standard deviation of the valid sea pixels (network input).
  6. Values kept for CMOD5.N: mean backscatter (linear mean, in dB), median incidence, median look azimuth.
  7. Wind label from Copernicus Marine L4 at the patch centre and acquisition time (linear in time,
     bilinear in space), as meteorological "from" direction.

Outputs (out_dir)
  dataset_A.npz    X (N, 50, 50) float32 | valid (N, 50, 50) bool | theta_from (N,) deg | uid (N,) | metadata arrays
  patch_table.csv  one row per GeoTIFF (kept and dropped, with the reason)
  qc_report.txt    per-scene summary and wind-direction coverage

  python build_dataset.py --tif_root .../tifs --out_dir .../stageA
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import zlib
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy as rio_xy
from rasterio.warp import transform as rio_transform
from scipy import ndimage

import wind_labels as W

PATCH_PX = 50
NODATA = -9999.0
MAX_BAD_FRACTION = 0.25      # patches with more land / no-data are dropped (as in the paper)
FILL_NOISE_SIGMA = 0.5       # noise added to filled pixels (z-score units); not specified in the paper
FNAME = re.compile(r"^(?P<patch>[A-Za-z]+\d{3})_(?P<sat>S1[A-D])_(?P<date>\d{8})_(?P<pass>ASC|DSC)_RO(?P<orbit>\d{3})"
                   r"_(?P<time>\d{6})(?:_\d+|\s\(\d+\))?\.tif$", re.I)


# =====================================================================================================
# 1. Filename, tif, land mask
# =====================================================================================================
def parse_tif_name(path: str) -> dict:
    name = os.path.basename(path)
    m = FNAME.match(name)
    if not m:
        raise ValueError(f"Unexpected file name '{name}' (expected e.g. C001_S1A_20240305_DSC_RO092_003133.tif)")
    g = m.groupdict()
    t = g["time"]
    scene_id = f"{g['sat'].upper()}_{g['date']}_{g['pass'].upper()}_RO{g['orbit']}_{t}"
    when = datetime.strptime(g["date"] + t, "%Y%m%d%H%M%S")
    return dict(patch_id=g["patch"].upper(), scene_id=scene_id, sat=g["sat"].upper(), date=g["date"], time=t,
                orbit=g["orbit"], pass_=g["pass"].upper(), scene_time=when.strftime("%Y-%m-%dT%H:%M:%S"),
                date_utc=when.strftime("%Y-%m-%d"), time_utc=when.strftime("%H:%M:%S"))


def read_patch_tif(path: str):
    """-> vv, inc, azi (50x50 float64, bad pixels = NaN), lon (50x50), lat (50x50), centre (lon, lat)"""
    with rasterio.open(path) as ds:
        if ds.count != 3 or (ds.height, ds.width) != (PATCH_PX, PATCH_PX):
            hint = " (1-band image = QUICKLOOK export; export again with PATCH_IMAGE_TYPE='DATA')" if ds.count == 1 else ""
            raise ValueError(f"Expected 3 bands of 50x50, found {ds.count} bands of {ds.height}x{ds.width}{hint}")
        arr = ds.read().astype(np.float64)
        tr, crs = ds.transform, ds.crs
    bad = ~np.isfinite(arr) | (arr <= -9000)
    arr[bad] = np.nan
    vv, inc, azi = arr
    rr, cc = np.mgrid[0:PATCH_PX, 0:PATCH_PX]
    xs, ys = rio_xy(tr, rr.ravel(), cc.ravel(), offset="center")
    lon, lat = rio_transform(crs, "EPSG:4326", list(xs), list(ys))
    lon, lat = np.asarray(lon).reshape(PATCH_PX, PATCH_PX), np.asarray(lat).reshape(PATCH_PX, PATCH_PX)
    cx, cy = tr.c + tr.a * PATCH_PX / 2, tr.f + tr.e * PATCH_PX / 2
    clon, clat = rio_transform(crs, "EPSG:4326", [cx], [cy])
    return vv, inc, azi, lon, lat, (clon[0], clat[0])


def land_mask_fn(enabled=True):
    if not enabled:
        return lambda lat, lon: np.zeros(np.shape(lat), bool)
    try:
        from global_land_mask import globe
    except ImportError as e:
        raise ImportError("Install global-land-mask (pip install global-land-mask) or use --no_land_mask.") from e
    return lambda lat, lon: globe.is_land(lat, lon)


# =====================================================================================================
# 2. One patch: quality check, z-score, gap filling
# =====================================================================================================
def process_patch(vv, inc, azi, land, uid):
    """Returns (info, z (50x50 float32), valid (50x50 bool)); z is None when the patch is dropped."""
    miss = ~np.isfinite(vv)
    bad = miss | land
    info = dict(nodata_frac=float(miss.mean()), land_frac=float(land.mean()), bad_frac=float(bad.mean()))
    if info["bad_frac"] > MAX_BAD_FRACTION:
        info["status"] = f"dropped: {100 * info['bad_frac']:.0f}% land/no-data (>25%)"
        return info, None, None
    valid = ~bad
    x = vv[valid]
    mu, sd = x.mean(), x.std()                                   # population std (ddof=0)
    if sd < 1e-6:
        info["status"] = "dropped: flat patch (std~0)"
        return info, None, None
    z = (vv - mu) / sd
    z[bad] = 0.0
    if bad.any():                                                # nearest sea pixel + Gaussian noise
        idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
        zf = z[tuple(idx)]
        rng = np.random.default_rng(zlib.crc32(uid.encode()))
        z = np.where(bad, zf + FILL_NOISE_SIGMA * rng.standard_normal(z.shape), z)
    lin = 10.0 ** (x / 10.0)
    ok_i, ok_a = valid & np.isfinite(inc), valid & np.isfinite(azi)
    info.update(status="kept", vv_mean_db=float(10 * np.log10(lin.mean())), vv_std_db=float(sd),
                inc_median=float(np.median(inc[ok_i])) if ok_i.any() else np.nan,
                azi_median=float(np.median(azi[ok_a])) if ok_a.any() else np.nan)
    return info, z.astype(np.float32), valid


# =====================================================================================================
# 3. Whole folder
# =====================================================================================================
def default_wind_provider(cache_dir, dataset_id, no_download=False):
    def provider(lons, lats, scene_time, scene_id):
        if no_download:
            cached = [os.path.join(cache_dir, f) for f in os.listdir(cache_dir)] if os.path.isdir(cache_dir) else []
            hit = [c for c in cached if scene_id in c]
            if not hit:
                raise FileNotFoundError(f"--no_download is set and no cached wind file exists for {scene_id}")
            return W.wind_labels_for_scene(lons, lats, scene_time, cache_dir, wind_nc=hit[0], tag=scene_id)
        # Scenes outside the reprocessed product (recent dates) get no label and are not used for training.
        return W.wind_labels_for_scene(lons, lats, scene_time, cache_dir, dataset_id=dataset_id, tag=scene_id)
    return provider


def _u(col):
    """pandas string column -> numpy fixed-width unicode (pickle-free npz)."""
    return np.array([str(x) for x in col], dtype="U")


def axial_stats(theta_deg):
    th = np.radians(2 * np.asarray(theta_deg))
    R = float(np.hypot(np.sin(th).mean(), np.cos(th).mean()))
    mean_axis = float((0.5 * np.degrees(np.arctan2(np.sin(th).mean(), np.cos(th).mean()))) % 180)
    return mean_axis, R


def build(tif_root, out_dir, wind_provider, land_enabled=True, log=print):
    os.makedirs(out_dir, exist_ok=True)
    is_land = land_mask_fn(land_enabled)
    paths = sorted(glob_tifs(tif_root))
    rows, problems = [], []
    groups = defaultdict(list)
    for p in paths:
        try:
            meta = parse_tif_name(p)
        except ValueError as e:
            problems.append(str(e)); continue
        meta["tif_path"] = p
        meta["folder"] = os.path.relpath(os.path.dirname(p), tif_root)
        groups[meta["scene_id"]].append(meta)
    log(f"{len(paths)} GeoTIFFs | {len(groups)} scene(s) | {len(problems)} file-name problem(s)")

    X, V, T, keep_meta = [], [], [], []
    lines = []
    for scene_id, metas in sorted(groups.items()):
        kept_here = []
        for meta in sorted(metas, key=lambda m: m["patch_id"]):
            uid = f"{scene_id}_{meta['patch_id']}"
            row = dict(uid=uid, patch_id=meta["patch_id"], scene_id=scene_id, folder=meta["folder"], sat=meta["sat"],
                       date_utc=meta["date_utc"], time_utc=meta["time_utc"], orbit_pass=meta["pass_"], rel_orbit=meta["orbit"],
                       tif_path=meta["tif_path"])
            try:
                vv, inc, azi, lon, lat, (clon, clat) = read_patch_tif(meta["tif_path"])
            except Exception as e:
                row.update(status=f"dropped: unreadable ({e})"); rows.append(row); continue
            info, z, valid = process_patch(vv, inc, azi, is_land(lat, lon), uid)
            row.update(info, centre_lon=clon, centre_lat=clat)
            rows.append(row)
            if z is not None:
                kept_here.append((row, z, valid))

        # ---- wind labels for the whole scene in ONE download ----
        method_count = Counter()
        wind_note = ""
        if kept_here:
            lons = np.array([r["centre_lon"] for r, _, _ in kept_here]); lats = np.array([r["centre_lat"] for r, _, _ in kept_here])
            try:
                w = wind_provider(lons, lats, metas[0]["scene_time"], scene_id)
                wind_note = f"time fraction {w['frac']:.4f} | {os.path.basename(str(w.get('wind_nc', 'n/a')))}"
            except Exception as e:
                w = None; wind_note = f"WIND LABEL FAIL: {e}"
        for k, (row, z, valid) in enumerate(kept_here):
            if w is not None:
                row.update(wind_u=float(w["u"][k]), wind_v=float(w["v"][k]), wind_speed=float(w["speed"][k]),
                           wind_dir_from=float(w["dir_from"][k]), label_method=str(w["method"][k]))
            else:
                row.update(wind_u=np.nan, wind_v=np.nan, wind_speed=np.nan, wind_dir_from=np.nan, label_method="missing")
            row["has_label"] = bool(np.isfinite(row["wind_dir_from"]))
            row["wind_axis_mod180"] = row["wind_dir_from"] % 180 if row["has_label"] else np.nan
            method_count[row["label_method"]] += 1
            row["npz_index"] = len(X)
            X.append(z); V.append(valid); T.append(row["wind_dir_from"]); keep_meta.append(row)

        n_all = len(metas); n_keep = len(kept_here)
        drop = Counter(re.sub(r"\d+%", "N%", r["status"]) for r in rows if r["scene_id"] == scene_id and r["status"] != "kept")
        lines.append(f"SCENE {scene_id}  ({metas[0]['folder']})  {metas[0]['date_utc']} {metas[0]['time_utc']} UTC")
        lines.append(f"    tifs {n_all} | kept {n_keep} | dropped {n_all - n_keep} {dict(drop) if drop else ''}")
        lines.append(f"    labels: {dict(method_count)} | {wind_note}")
        if n_keep:
            th = np.array([r["wind_dir_from"] for r, _, _ in kept_here if np.isfinite(r["wind_dir_from"])])
            if len(th):
                cm = float(np.degrees(np.arctan2(np.sin(np.radians(th)).mean(), np.cos(np.radians(th)).mean())) % 360)
                dev = (th - cm + 180) % 360 - 180                      # 0/360 wrap-around safe
                lines.append(f"    wind dir (from): circ.mean {cm:.1f} deg, spread {dev.min():+.1f}..{dev.max():+.1f} | speed "
                             f"{np.nanmean([r['wind_speed'] for r, _, _ in kept_here]):.1f} m/s")
        log("\n".join(lines[-4:]))

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(out_dir, "patch_table.csv"), index=False)
    if X:
        kt = table[table.status == "kept"].sort_values("npz_index")
        np.savez_compressed(
            os.path.join(out_dir, "dataset_A.npz"),
            X=np.stack(X).astype(np.float32), valid=np.stack(V), theta_from=np.asarray(T, np.float32),
            uid=_u(kt.uid), scene_id=_u(kt.scene_id), folder=_u(kt.folder),
            centre_lon=kt.centre_lon.values.astype(np.float64), centre_lat=kt.centre_lat.values.astype(np.float64),
            vv_mean_db=kt.vv_mean_db.values.astype(np.float32), inc_median=kt.inc_median.values.astype(np.float32),
            azi_median=kt.azi_median.values.astype(np.float32), land_frac=kt.land_frac.values.astype(np.float32),
            wind_speed=kt.wind_speed.values.astype(np.float32), date_utc=_u(kt.date_utc),
            time_utc=_u(kt.time_utc))

    # ---- QC report ----
    rep = ["=" * 78, "STEP A  -  QC REPORT", "=" * 78] + lines
    lab = table[(table.status == "kept") & (table.get("has_label", False) == True)] if len(table) else table
    if len(lab):
        mean_axis, R = axial_stats(lab.wind_dir_from.values)
        bins = np.histogram(lab.wind_axis_mod180.values, bins=np.arange(0, 181, 30))[0]
        rep += ["", f"LABELLED PATCHES: {len(lab)}  | wind axis (mod 180) mean {mean_axis:.0f} deg, concentration R={R:.3f} "
                f"(1 = all patches share one axis)",
                "  axis histogram (30-deg bins from 0..180): " + "  ".join(f"[{30*i}-{30*i+30}) {n}" for i, n in enumerate(bins))]
        if R > 0.9 or (bins > 0).sum() < 4:
            rep.append("  !! WARNING: wind directions cover only a small part of the 0-180 deg axis range; "
                       "add scenes with other wind directions.")
    if problems:
        rep += ["", "FILE-NAME PROBLEMS (files skipped):"] + ["  " + p for p in problems]
    text = "\n".join(rep)
    with open(os.path.join(out_dir, "qc_report.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    log("\n" + "\n".join(rep[-6:]))
    return table


def glob_tifs(root):
    out = []
    for d, _, files in os.walk(root):
        out += [os.path.join(d, f) for f in files if f.lower().endswith(".tif")]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tif_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cache_dir", default=None, help="cache folder for Copernicus wind files (default: out_dir/wind_cache)")
    ap.add_argument("--dataset_id", default=W.DATASET_ID_MY)
    ap.add_argument("--no_download", action="store_true", help="use cached wind files only")
    ap.add_argument("--no_land_mask", action="store_true")
    a = ap.parse_args(argv)
    cache = a.cache_dir or os.path.join(a.out_dir, "wind_cache")
    build(a.tif_root, a.out_dir, default_wind_provider(cache, a.dataset_id, a.no_download), land_enabled=not a.no_land_mask)


if __name__ == "__main__":
    sys.exit(main())
