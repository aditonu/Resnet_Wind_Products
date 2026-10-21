"""
scat_compare.py - compares the SAR (ResNet) direction with an independent reference: scatterometer winds
                  (Copernicus Marine L3, WIND_GLO_PHY_L3_NRT_012_002 / WIND_GLO_PHY_L3_MY_012_005).

Scatterometers measure the wind vector directly from radar backscatter at several look angles, whereas the
Copernicus L4 product used as reference is ERA5 corrected with scatterometer data. Interpretation:
    ResNet ~ scatterometer and Copernicus L4 differs  -> Copernicus L4 is less accurate near the coast
    scatterometer ~ Copernicus L4 and ResNet differs  -> the SAR streaks themselves deviate from the surface wind
Scatterometer and Sentinel-1 overpasses are not simultaneous; the time difference (dt) of every match is reported.
For each patch the nearby scatterometer cell closest in time to the SAR acquisition is used (max --max_dt_h hours).

Requires the Copernicus Marine sign-in (copernicusmarine.login()).

  python scat_compare.py --run_dir .../api_runs/<scene>_<box>
"""
from __future__ import annotations

import argparse
import glob
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr

NRT = "cmems_obs-wind_glo_phy_nrt_l3-{sat}-{inst}-{pas}-{res}_P1D-i"
CANDIDATES = [("metopb", "ascat", "0.125deg"), ("metopc", "ascat", "0.125deg"), ("hy2b", "hscat", "0.25deg"),
              ("hy2c", "hscat", "0.25deg"), ("hy2d", "hscat", "0.25deg"), ("oceansat3", "oscat", "0.25deg")]


def dataset_ids():
    return [NRT.format(sat=s, inst=i, pas=p, res=r) for s, i, r in CANDIDATES for p in ("asc", "des")]


def wrap180(a):
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


def alias_err(p, t):
    return (np.asarray(p, float) - np.asarray(t, float) + 90.0) % 180.0 - 90.0


def axial_mean(a):
    t = np.radians(2 * np.asarray(a, float))
    return float((0.5 * np.degrees(np.arctan2(np.nanmean(np.sin(t)), np.nanmean(np.cos(t))))) % 180)


def dir_from_uv(u, v):
    return (270.0 - np.degrees(np.arctan2(v, u))) % 360.0


# ---------------------------------------------------------------------------------------------------
def download(ds_id, box, day0, day1, out_dir, pad=0.3):
    import copernicusmarine
    os.makedirs(out_dir, exist_ok=True)
    fn = f"{ds_id}_{day0:%Y%m%d}_{day1:%Y%m%d}_{box[0]:.2f}_{box[1]:.2f}.nc"
    path = os.path.join(out_dir, fn)
    if not os.path.exists(path):
        copernicusmarine.subset(dataset_id=ds_id, minimum_longitude=box[0] - pad, maximum_longitude=box[2] + pad,
                                minimum_latitude=box[1] - pad, maximum_latitude=box[3] + pad,
                                start_datetime=f"{day0:%Y-%m-%d} 00:00:00", end_datetime=f"{day1:%Y-%m-%d} 23:59:59",
                                output_directory=out_dir, output_filename=fn)
    return path


def _find(ds, keys):
    for k in ds.data_vars:
        if any(x == k.lower() for x in keys):
            return k
    for k in ds.data_vars:
        if any(x in k.lower() for x in keys):
            return k
    return None


def colocate(ds, lons, lats, t_sar, max_dist_deg=0.2):
    """For each point, the nearby valid cell closest in time to the SAR acquisition -> (dir_from, speed, dt_hours, dist_deg)."""
    uk, vk = _find(ds, ["eastward_wind"]), _find(ds, ["northward_wind"])
    if uk is None or vk is None:
        raise KeyError(f"eastward/northward wind not found: {list(ds.data_vars)}")
    tk = _find(ds, ["measurement_time"]) or next((k for k in ds.data_vars if "time" in k.lower()), None)
    U, V = ds[uk], ds[vk]
    lat_n = next(d for d in U.dims if "lat" in d.lower()); lon_n = next(d for d in U.dims if "lon" in d.lower())
    U = U.transpose(..., lat_n, lon_n); V = V.transpose(..., lat_n, lon_n)
    if tk is not None:
        TT = ds[tk].transpose(..., lat_n, lon_n).values
        TT = TT.astype("datetime64[s]") if np.issubdtype(TT.dtype, np.datetime64) else None
    else:
        TT = None
    Uv = U.values.reshape(-1, U.shape[-2], U.shape[-1]); Vv = V.values.reshape(-1, V.shape[-2], V.shape[-1])
    if TT is not None:
        TT = TT.reshape(-1, TT.shape[-2], TT.shape[-1])
    else:   # only the day is known (no exact time)
        days = ds["time"].values.astype("datetime64[s]") if "time" in ds else np.array([np.datetime64(t_sar, "D")] * Uv.shape[0])
        TT = np.broadcast_to(days.reshape(-1, 1, 1), Uv.shape).astype("datetime64[s]")
    glat, glon = ds[lat_n].values, ds[lon_n].values
    LON, LAT = np.meshgrid(glon, glat)
    t_sar = np.datetime64(t_sar, "s")
    out = []
    for lo, la in zip(lons, lats):
        d = np.hypot(LON - lo, (LAT - la))
        near = d <= max_dist_deg
        best = None
        for k in range(Uv.shape[0]):
            ok = near & np.isfinite(Uv[k]) & np.isfinite(Vv[k]) & ~np.isnat(TT[k])
            if not ok.any():
                continue
            dt = np.abs((TT[k][ok] - t_sar) / np.timedelta64(1, "h")).astype(float)
            dd = d[ok]
            j = np.lexsort((dd, dt))[0]
            idx = np.argwhere(ok)[j]
            cand = (dt[j], dd[j], float(Uv[k][tuple(idx)]), float(Vv[k][tuple(idx)]))
            if best is None or cand[:2] < best[:2]:
                best = cand
        if best is None:
            out.append((np.nan, np.nan, np.nan, np.nan))
        else:
            dt, dd, u, v = best
            out.append((float(dir_from_uv(u, v)), float(np.hypot(u, v)), dt, dd))
    return np.array(out, float)


def compare(W, scat, max_dt_h):
    m = np.isfinite(scat[:, 0]) & (scat[:, 2] <= max_dt_h)
    if m.sum() == 0:
        return None
    S = W[m].copy()
    S["scat_dir"], S["scat_speed"], S["dt_h"], S["dist_deg"] = scat[m, 0], scat[m, 1], scat[m, 2], scat[m, 3]
    r = dict(n=int(m.sum()), dt_median_h=float(np.median(S.dt_h)), scat_speed=float(S.scat_speed.mean()),
             cop_speed=float(S.cop_speed.mean()), sar_speed_cmod5n=float(S.speed_cmod5n.mean()))
    for name, col in (("resnet", "axis_resnet"), ("copernicus", "cop_dir")):
        e = alias_err(S[col] % 180, S.scat_dir % 180)
        r[f"{name}_minus_scat_bias"] = float(np.mean(e)); r[f"{name}_minus_scat_MAE"] = float(np.mean(np.abs(e)))
    g = S[S.quality == "good"]
    if len(g):
        e = alias_err(g.axis_resnet % 180, g.scat_dir % 180)
        r["resnet_good_minus_scat_MAE"] = float(np.mean(np.abs(e))); r["n_good"] = int(len(g))
    r["resnet_minus_copernicus_bias"] = float(np.mean(alias_err(S.axis_resnet % 180, S.cop_dir % 180)))
    r["scat_speed_minus_cop"] = float(np.mean(S.scat_speed - S.cop_speed))
    return r, S


def main(argv=None, opener=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--max_dt_h", type=float, default=4.0, help="maximum time difference between SAR and scatterometer (hours)")
    ap.add_argument("--cache_dir", default=None)
    a = ap.parse_args(argv)
    cache = a.cache_dir or os.path.join(a.run_dir, "scat_cache")
    results, all_rows = [], []
    for csv in sorted(glob.glob(os.path.join(a.run_dir, "wind_field_*.csv"))):
        W = pd.read_csv(csv).rename(columns={"ref_dir": "cop_dir", "ref_speed": "cop_speed"})
        W = W[(W.status == "kept") & W.cop_dir.notna()].reset_index(drop=True)
        if W.empty:
            continue
        t_sar = np.datetime64(f"{W.date_utc.iloc[0]}T{W.time_utc.iloc[0]}", "s")
        d = datetime.strptime(str(W.date_utc.iloc[0]), "%Y-%m-%d")
        box = (W.lon.min(), W.lat.min(), W.lon.max(), W.lat.max())
        scene = os.path.basename(csv)[11:-4]
        print(f"\n### {scene} | SAR time {t_sar} UTC | {len(W)} sea patches")
        best = None
        for ds_id in dataset_ids():
            try:
                ds = opener(ds_id) if opener else xr.open_dataset(download(ds_id, box, d - timedelta(days=1), d + timedelta(days=1), cache))
                sc = colocate(ds, W.lon.values, W.lat.values, t_sar)
                rr = compare(W, sc, a.max_dt_h)
                if rr is None:
                    ok = np.isfinite(sc[:, 2])
                    print(f"  {ds_id}: {'no match' if not ok.any() else f'time difference too large (min {np.nanmin(sc[:, 2]):.1f} h)'}")
                    continue
                r, S = rr
                print(f"  {ds_id}: {r['n']} matches, dt {r['dt_median_h']:.1f} h | ResNet-scat bias {r['resnet_minus_scat_bias']:+.1f} "
                      f"MAE {r['resnet_minus_scat_MAE']:.1f} | Copernicus-scat bias {r['copernicus_minus_scat_bias']:+.1f} "
                      f"MAE {r['copernicus_minus_scat_MAE']:.1f}")
                all_rows.append(dict(scene=scene, scat_dataset=ds_id, **r))
                # choose the smallest time difference (wind unchanged in between), with at least 10 matches
                key = (0 if r["n"] >= max(10, 0.1 * len(W)) else 1, r["dt_median_h"], -r["n"])
                if best is None or key < best[0]:
                    best = (key, ds_id, r, S)
            except Exception as e:
                print(f"  {ds_id}: not available ({str(e)[:100]})")
        if best is None:
            print("  -> no scatterometer match for this scene (time / location). Try a larger --max_dt_h.")
            continue
        _, ds_id, r, S = best
        S.to_csv(os.path.join(a.run_dir, f"scat_matches_{scene}.csv"), index=False)
        results.append(dict(scene=scene, scat_dataset=ds_id, **r))
        print(f"  -> SELECTED: {ds_id} (dt {r['dt_median_h']:.1f} h, n={r['n']})")
        print(f"     ResNet - scat     : bias {r['resnet_minus_scat_bias']:+.1f}, MAE {r['resnet_minus_scat_MAE']:.1f} deg")
        print(f"     Copernicus - scat : bias {r['copernicus_minus_scat_bias']:+.1f}, MAE {r['copernicus_minus_scat_MAE']:.1f} deg")
        print(f"     ResNet - Copernicus: bias {r['resnet_minus_copernicus_bias']:+.1f} deg")
        print(f"     speed: scat {r['scat_speed']:.1f} | Copernicus {r['cop_speed']:.1f} | SAR CMOD5.N {r['sar_speed_cmod5n']:.1f} m/s")
    if all_rows:
        pd.DataFrame(all_rows).to_csv(os.path.join(a.run_dir, "scat_compare_all_datasets.csv"), index=False)
    if results:
        R = pd.DataFrame(results)
        R.to_csv(os.path.join(a.run_dir, "scat_compare.csv"), index=False)
        pd.set_option("display.width", 250)
        print("\n=== SCATTEROMETER COMPARISON ===\n" + R.round(2).to_string(index=False))
        print("\nInterpretation: small ResNet-scat and large Copernicus-scat bias -> Copernicus L4 less accurate near the coast."
              "\n                small Copernicus-scat and large ResNet-scat bias -> SAR streaks deviate from the surface wind."
              "\n                With dt > 2 h the wind may have changed between the two acquisitions.")
    return results


if __name__ == "__main__":
    main()
