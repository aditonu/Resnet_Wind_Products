"""
infer_scene.py - wind field for one or more Sentinel-1 scenes from their 5 km patch GeoTIFFs.

For every 5 km patch:
  1. Read the GeoTIFF, reject land / no-data patches and normalise (same functions as build_dataset.py).
  2. ResNet ensemble (wind_model.py, final_model/) -> wind axis (180 deg ambiguity) and confidence.
  3. Quality flag ("good" when both tests pass, otherwise "low"):
       - neighbour consistency: axis within --max_nbr_dev (default 20 deg) of the neighbours (<= 7.6 km);
       - texture strength: anisotropy of the combined FFT spectrum of the patch and its neighbours in the
         0.8-2.5 km band >= --min_aniso (default 0.20).
     Calibrated on cross-validation errors (unseen scenes): "good" = 60 % of patches, MAE 13.6 deg
     (median 9.9 deg); "low" = MAE 36 deg.
     Area flag: mean anisotropy of the area < --min_area_aniso (default 0.21) marks an area with weak
     texture (weak scenes 0.11-0.19, good scenes 0.24-0.50).
  4. Ambiguity removal: of the two senses (a, a + 180) the one closer to the reference wind direction
     (Copernicus Marine L4 by default; the API passes NOAA GFS from Earth Engine).
  5. Wind speed: CMOD5.N inversion (xsarsea) with mean VV (linear mean, in dB), median incidence angle and
     phi = wind direction (from) - radar look azimuth.
  6. Outputs: wind_field_<scene>.csv, summary_<scene>.json, map_<scene>.png, validation_table.csv

Direction scores on training scenes are in-sample (column in_sample_direction); CMOD5.N is not trained,
so the speed comparison is independent on every scene.

  python infer_scene.py --tif_dir .../tifs --model_dir .../final_model --out_dir .../scene_validation \
                        --cache_dir .../stageA/wind_cache
  (--tif_dir may hold one scene folder or many; each scene is processed separately)
"""
from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd

import build_dataset as BD
import texture_check as TC
import train_resnet as T
import wind_labels as WL
from wind_model import load_ensemble, predict_axis

warnings.filterwarnings("ignore")
NBR_KM = 7.6          # 5 km grid: direct neighbours at 5 km, diagonal at 7.07 km


# =====================================================================================================
# CMOD5.N inversion
# =====================================================================================================
_GMF = None
WSPD_GRID = np.arange(0.2, 35.0001, 0.05)


def gmf():
    global _GMF
    if _GMF is None:
        from xsarsea import windspeed
        _GMF = windspeed.get_model("gmf_cmod5n")
    return _GMF


def cmod5n_sigma0_db(inc, wspd, phi):
    s = gmf()(np.array([float(inc)]), np.atleast_1d(np.asarray(wspd, float)), np.array([float(phi)]))
    return 10 * np.log10(np.asarray(s.values, float).ravel())


def invert_cmod5n(sigma0_db, inc, phi):
    """Wind speed for which CMOD5.N(inc, wspd, phi) matches sigma0 (0.2-35 m/s, 0.05 step); NaN for missing inputs."""
    out, flag = np.full(len(sigma0_db), np.nan), np.array([""] * len(sigma0_db), dtype=object)
    for i, (s, t, p) in enumerate(zip(sigma0_db, inc, phi)):
        if not (np.isfinite(s) and np.isfinite(t) and np.isfinite(p)):
            continue
        model = cmod5n_sigma0_db(t, WSPD_GRID, p)
        j = int(np.argmin(np.abs(model - s)))
        out[i] = WSPD_GRID[j]
        if s < model.min() + 0.05 or j == 0: flag[i] = "below_range"
        elif j == len(WSPD_GRID) - 1: flag[i] = "above_range"
    return out, flag


# =====================================================================================================
# helpers
# =====================================================================================================
def haversine_km(lon1, lat1, lon2, lat2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 12742.0 * np.arcsin(np.sqrt(a))


def neighbours(lons, lats, max_km=NBR_KM):
    D = haversine_km(lons[:, None], lats[:, None], lons[None, :], lats[None, :])
    np.fill_diagonal(D, np.inf)
    return [np.where(D[i] <= max_km)[0] for i in range(len(lons))]


def quality_checks(axis, Z, nbrs, max_dev, min_aniso, long_px=(8.0, 25.0)):
    n = len(axis)
    P = TC.spectra(Z.astype(np.float64))
    kmag, kbear = TC.kgrid(Z.shape[-1])
    dev, aniso = np.full(n, np.nan), np.full(n, np.nan)
    for i in range(n):
        if len(nbrs[i]):
            dev[i] = float(T.alias_err(axis[i], T.axial_mean(axis[nbrs[i]])))
        grp = np.concatenate([[i], nbrs[i]])
        _, an, _ = TC.band_stats(P[grp].mean(0), kmag, kbear, *long_px)
        aniso[i] = float(an)
    consistent = ~(np.abs(dev) > max_dev)              # no neighbours -> treated as consistent
    texture_ok = aniso >= min_aniso
    return dev, aniso, consistent, texture_ok


def dealias(axis, ref_from):
    a = np.asarray(axis) % 180
    alt = a + 180
    return np.where(np.abs(T.wrap180(a - ref_from)) <= np.abs(T.wrap180(alt - ref_from)), a, alt) % 360


def stats_dir(pred, ref):
    e = T.wrap180(np.asarray(pred) - np.asarray(ref))
    ea = T.alias_err(pred, ref)
    if len(e) == 0:
        return {}
    b = float(e.mean()); rmsd = float(np.sqrt((e ** 2).mean()))
    return dict(n=int(len(e)), dir_bias=b, dir_RMSD=rmsd, dir_cRMSD=float(np.sqrt(max(rmsd ** 2 - b ** 2, 0))),
                dir_aliased_MAE=float(np.abs(ea).mean()), dir_within20_pct=float(100 * (np.abs(ea) <= 20).mean()))


def stats_spd(pred, ref):
    m = np.isfinite(pred) & np.isfinite(ref)
    if m.sum() < 2:
        return {}
    d = pred[m] - ref[m]
    return dict(spd_bias=float(d.mean()), spd_RMSD=float(np.sqrt((d ** 2).mean())),
                spd_corr=float(np.corrcoef(pred[m], ref[m])[0, 1]) if np.std(pred[m]) > 0 and np.std(ref[m]) > 0 else np.nan)


# =====================================================================================================
# one scene
# =====================================================================================================
def process_scene(scene_id, metas, models, cfg, a, wind_provider=None, log=print):
    is_land = BD.land_mask_fn(not a.no_land_mask)
    rows, Z, land_pts, foot = [], [], [], {}
    for meta in sorted(metas, key=lambda m: m["patch_id"]):
        uid = f"{scene_id}_{meta['patch_id']}"
        r = dict(uid=uid, patch_id=meta["patch_id"], scene_id=scene_id, folder=meta["folder"],
                 date_utc=meta["date_utc"], time_utc=meta["time_utc"])
        try:
            vv, inc, azi, lon, lat, (clon, clat) = BD.read_patch_tif(meta["tif_path"])
        except Exception as e:
            r.update(status=f"unreadable ({e})"); rows.append(r); continue
        land = is_land(lat, lon)
        info, z, valid = BD.process_patch(vv, inc, azi, land, uid)
        r.update(info, lon=clon, lat=clat)
        land_pts.append(np.stack([lon[::2, ::2][land[::2, ::2]], lat[::2, ::2][land[::2, ::2]]], 1))
        foot[uid] = np.array([[lon[0, 0], lat[0, 0]], [lon[0, -1], lat[0, -1]], [lon[-1, -1], lat[-1, -1]], [lon[-1, 0], lat[-1, 0]]])
        if z is not None:
            r["k"] = len(Z); Z.append(z)
        rows.append(r)
    R = pd.DataFrame(rows)
    kept = R[R.status == "kept"].sort_values("k").reset_index(drop=True)
    log(f"\nSCENE {scene_id} ({metas[0]['folder']}) {metas[0]['date_utc']} {metas[0]['time_utc']} UTC | "
        f"tifs {len(R)} | sea patches {len(kept)} | land/no-data {len(R) - len(kept)}")
    if len(kept) == 0:
        return R, {}, land_pts, foot
    Z = np.stack(Z)
    axis, conf = predict_axis(models, cfg, Z)
    lons, lats = kept.lon.values, kept.lat.values
    nb = neighbours(lons, lats)
    dev, aniso, consistent, texture_ok = quality_checks(axis, Z, nb, a.max_nbr_dev, a.min_aniso)
    kept["axis_resnet"], kept["conf"], kept["nbr_dev"], kept["nbr_aniso"] = axis, conf, dev, aniso
    kept["quality"] = np.where(consistent & texture_ok, "good", "low")
    area_aniso = float(np.nanmean(aniso))
    area_ok = area_aniso >= a.min_area_aniso

    # ---- reference wind (ambiguity removal + comparison) ----
    ref_name = "Copernicus Marine"
    try:
        if wind_provider is not None:
            w = wind_provider(lons, lats, metas[0]["scene_time"], scene_id)
        else:
            w = WL.wind_labels_for_scene(lons, lats, metas[0]["scene_time"], a.cache_dir, dataset_id=a.dataset_id,
                                         tag=scene_id, wind_nc=a.wind_nc)
        ref_name = w.get("source", ref_name)
        kept["ref_dir"], kept["ref_speed"] = w["dir_from"], w["speed"]
        ref_note = f"Reference wind: {ref_name} ({w.get('detail') or os.path.basename(str(w.get('wind_nc', '')))})"
    except Exception as e:
        kept["ref_dir"], kept["ref_speed"] = np.nan, np.nan
        ref_note = f"Reference wind ({ref_name}) not available ({e}); 180 deg ambiguity not removed, axis only"
    has_ref = np.isfinite(kept.ref_dir.values)
    kept["dir_from"] = np.where(has_ref, dealias(axis, kept.ref_dir.values), axis % 180)
    kept["dealiased"] = has_ref

    # ---- CMOD5.N speed ----
    kept["phi"] = (kept.dir_from - kept.azi_median) % 360
    kept["speed_cmod5n"], kept["speed_flag"] = invert_cmod5n(kept.vv_mean_db.values, kept.inc_median.values, kept.phi.values)
    if has_ref.any():   # CMOD5.N with the reference direction (isolates the effect of direction errors on speed)
        phi_ref = (kept.ref_dir - kept.azi_median) % 360
        kept["speed_cmod5n_refdir"], _ = invert_cmod5n(kept.vv_mean_db.values, kept.inc_median.values, phi_ref.values)
    else:
        kept["speed_cmod5n_refdir"] = np.nan
    rad = np.radians(kept.dir_from.values)
    kept["u"], kept["v"] = -kept.speed_cmod5n * np.sin(rad), -kept.speed_cmod5n * np.cos(rad)   # vector "to" components

    # ---- summary ----
    in_sample = (str(metas[0]["folder"]) in [str(s) for s in cfg.get("train_scenes", [])]
                 or str(scene_id) in [str(s) for s in cfg.get("train_scene_ids", [])])
    S = dict(scene_id=scene_id, folder=metas[0]["folder"], date_utc=metas[0]["date_utc"], time_utc=metas[0]["time_utc"],
             in_sample_direction=in_sample, n_tifs=int(len(R)), n_sea=int(len(kept)),
             n_good=int((kept.quality == "good").sum()), area_aniso=area_aniso, area_texture_ok=bool(area_ok),
             reference=ref_name if has_ref.any() else "none", reference_note=ref_note,
             mean_speed_cmod5n=float(np.nanmean(kept.speed_cmod5n)))
    if has_ref.any():
        k = kept[has_ref]
        for tag, sub in (("all", k), ("good", k[k.quality == "good"])):
            for kk, vv in {**stats_dir(sub.dir_from.values, sub.ref_dir.values),
                           **stats_spd(sub.speed_cmod5n.values, sub.ref_speed.values)}.items():
                S[f"{tag}_{kk}"] = vv
        S["mean_speed_reference"] = float(np.nanmean(k.ref_speed))
    out = R.drop(columns=[c for c in ("k",) if c in R]).merge(
        kept.drop(columns=[c for c in R.columns if c not in ("uid",)] + ["k"], errors="ignore"), on="uid", how="left")
    log(f"  {ref_note}\n  area texture {area_aniso:.2f} -> {'OK' if area_ok else 'WEAK -> trust only black (good) arrows'}"
        f" | good quality {S['n_good']}/{S['n_sea']} | mean speed CMOD5.N {S['mean_speed_cmod5n']:.1f} m/s"
        + (f" vs reference {S['mean_speed_reference']:.1f}" if "mean_speed_reference" in S else ""))
    if "all_dir_aliased_MAE" in S:
        log(f"  direction vs reference (all): bias {S['all_dir_bias']:+.1f}, cRMSD {S['all_dir_cRMSD']:.1f}, "
            f"aliased MAE {S['all_dir_aliased_MAE']:.1f} deg{'  [IN-SAMPLE: training scene]' if in_sample else ''}")
        if "good_dir_aliased_MAE" in S:
            log(f"  direction vs reference (good): aliased MAE {S['good_dir_aliased_MAE']:.1f} deg (n={S['good_n']})")
        if "all_spd_bias" in S:
            log(f"  speed vs reference (all): bias {S['all_spd_bias']:+.2f}, RMSD {S['all_spd_RMSD']:.2f} m/s, r {S['all_spd_corr']:.2f}")
    return out, S, land_pts, foot


# =====================================================================================================
# Map (as in Fig. 5 of Zanchetta & Zecchetto, 2021)
# =====================================================================================================
def plot_map(out, S, land_pts, foot, png):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    sea = out[out.status == "kept"]
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    vmax = max(10.0, float(np.nanpercentile(sea.speed_cmod5n, 98)) if len(sea) else 10.0)
    if len(sea):
        pc = PolyCollection([foot[u] for u in sea.uid], array=sea.speed_cmod5n.values, cmap="jet", clim=(0, vmax),
                            edgecolors="none")
        ax.add_collection(pc)
        plt.colorbar(pc, ax=ax, label="wind speed CMOD5.N (m/s)", shrink=0.8)
    lp = np.concatenate(land_pts) if land_pts and sum(len(x) for x in land_pts) else np.zeros((0, 2))
    if len(lp):
        ax.scatter(lp[:, 0], lp[:, 1], s=1, c="0.6", marker="s", linewidths=0)
    for u in out.uid:       # patch grid lines
        if u in foot:
            f = foot[u]; ax.plot(np.r_[f[:, 0], f[0, 0]], np.r_[f[:, 1], f[0, 1]], color="k", lw=0.15, alpha=0.4)
    if len(sea):
        rad = np.radians(sea.dir_from.values + 180)              # arrow points where wind goes
        uu, vv = np.sin(rad), np.cos(rad)
        good = (sea.quality == "good").values
        ax.quiver(sea.lon[good], sea.lat[good], uu[good], vv[good], color="k", scale=22, width=0.004)
        ax.quiver(sea.lon[~good], sea.lat[~good], uu[~good], vv[~good], color="w", edgecolor="k", linewidth=0.5,
                  scale=22, width=0.004)
    allf = np.concatenate(list(foot.values())) if foot else np.zeros((1, 2))
    px, py = 0.04 * np.ptp(allf[:, 0]) + 0.01, 0.04 * np.ptp(allf[:, 1]) + 0.01
    ax.set_xlim(allf[:, 0].min() - px, allf[:, 0].max() + px); ax.set_ylim(allf[:, 1].min() - py, allf[:, 1].max() + py)
    ax.set_aspect(1 / np.cos(np.radians(allf[:, 1].mean())))
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ref = S.get("reference", "none")
    note = "" if ref != "none" else "  (180 deg ambiguity NOT removed)"
    if not S.get("area_texture_ok", True):
        note += "\nWARNING: weak SAR texture in this area - trust only BLACK (good) arrows"
    ax.set_title(f"Sentinel-1 wind field  {S.get('date_utc', '')} {S.get('time_utc', '')} UTC  ({S.get('folder', '')})\n"
                 f"ResNet direction (ambiguity removed with {ref}) + CMOD5.N speed\nblack arrows = good quality, white = low quality{note}",
                 fontsize=8)
    plt.tight_layout(); plt.savefig(png, dpi=160); plt.close()


# =====================================================================================================
def run(a, wind_provider=None):
    os.makedirs(a.out_dir, exist_ok=True)
    models, cfg = load_ensemble(a.model_dir)
    groups = {}
    for p in sorted(BD.glob_tifs(a.tif_dir)):
        try:
            m = BD.parse_tif_name(p)
        except ValueError:
            continue
        m["tif_path"] = p
        m["folder"] = os.path.basename(os.path.dirname(p))
        groups.setdefault(m["scene_id"], []).append(m)
    if not groups:
        raise SystemExit(f"No correctly named GeoTIFFs found in {a.tif_dir}")
    print(f"{sum(len(v) for v in groups.values())} tifs | {len(groups)} scene(s) | model: {a.model_dir} "
          f"({len(models)} models, low-pass {cfg['lowpass_px']} px)")
    summaries = []
    for sid, metas in sorted(groups.items()):
        out, S, land_pts, foot = process_scene(sid, metas, models, cfg, a, wind_provider)
        tag = f"{metas[0]['folder']}_{sid}"
        out.to_csv(os.path.join(a.out_dir, f"wind_field_{tag}.csv"), index=False)
        json.dump(S, open(os.path.join(a.out_dir, f"summary_{tag}.json"), "w"), indent=2, default=str)
        try:
            plot_map(out, S, land_pts, foot, os.path.join(a.out_dir, f"map_{tag}.png"))
        except Exception as e:
            print("  map skipped:", e)
        summaries.append(S)
    V = pd.DataFrame(summaries)
    V.to_csv(os.path.join(a.out_dir, "validation_table.csv"), index=False)
    cols = [c for c in ("folder", "in_sample_direction", "area_texture_ok", "n_sea", "n_good", "all_dir_bias", "all_dir_cRMSD", "all_dir_aliased_MAE",
                        "good_dir_aliased_MAE", "all_spd_bias", "all_spd_RMSD", "all_spd_corr",
                        "mean_speed_cmod5n", "mean_speed_reference") if c in V]
    pd.set_option("display.width", 250)
    print(f"\n=== VALIDATION TABLE (vs reference wind: {', '.join(sorted(set(map(str, V.get('reference', ['none'])))))}) ===\n" + V[cols].round(2).to_string(index=False))
    print("\nNOTE: direction scores of scenes with in_sample_direction=True are in-sample; the independent direction score is the"
          "\n      cross-validation result. The CMOD5.N speed comparison is independent on every scene.")
    print(f"\nOutputs: {a.out_dir}")
    return V


def main(argv=None, wind_provider=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tif_dir", required=True); ap.add_argument("--model_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cache_dir", default=None, help="Copernicus wind cache folder (reuses the build_dataset cache)")
    ap.add_argument("--wind_nc", default=None); ap.add_argument("--dataset_id", default=WL.DATASET_ID_MY)
    ap.add_argument("--max_nbr_dev", type=float, default=20.0); ap.add_argument("--min_aniso", type=float, default=0.20)
    ap.add_argument("--min_area_aniso", type=float, default=0.21)
    ap.add_argument("--no_land_mask", action="store_true")
    a = ap.parse_args(argv)
    a.cache_dir = a.cache_dir or os.path.join(a.out_dir, "wind_cache")
    return run(a, wind_provider)


if __name__ == "__main__":
    main()
