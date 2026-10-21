"""
wind_labels.py - Copernicus Marine wind at given points and time (u, v, speed, direction).

Used for the training labels, for the 180 deg ambiguity removal and as reference in the validation.

  1. Downloads the hourly wind (eastward_wind, northward_wind) around the scene, enlarged by PAD_DEG so that
     bilinear interpolation has neighbouring grid cells.
  2. Time interpolation between the hourly fields before and after the acquisition,
     fraction = (minutes * 60 + seconds) / 3600.
  3. Bilinear interpolation of u and v at the patch centres (directions are never interpolated).
  4. Direction is meteorological ("from"), degrees clockwise from north:
        dir_from = (270 - degrees(atan2(v, u))) mod 360      (0 = wind from the north, 90 = from the east)

Sign in with copernicusmarine.login() (or environment variables); credentials are never stored in the code.
"""

from __future__ import annotations

import glob
import os
from datetime import datetime, timezone

import numpy as np
import xarray as xr

# ----- Copernicus Marine products -----
DATASET_ID_MY = "cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H"       # reprocessed (multi-year)
DATASET_ID_NRT = "cmems_obs-wind_glo_phy_nrt_l4_0.125deg_PT1H"     # near-real-time (recent dates)
VAR_U, VAR_V = "eastward_wind", "northward_wind"
PAD_DEG = 0.375          # 3 grid cells (0.125 deg) on each side: enough for bilinear interpolation
MAX_FALLBACK_DEG = 0.25  # near the coast (NaN cells) the nearest valid cell within this distance is used


# =====================================================================================================
# 1. Helpers
# =====================================================================================================
def uv_to_dir_from(u, v):
    """Meteorological FROM direction (deg, 0..360). Wind from North -> 0, from East -> 90, from South -> 180."""
    return (270.0 - np.degrees(np.arctan2(v, u))) % 360.0


def uv_to_speed(u, v):
    return np.hypot(u, v)


def to_utc_naive(t) -> np.datetime64:
    """datetime / str / np.datetime64 -> np.datetime64 (UTC, timezone-naive, seconds)."""
    if isinstance(t, np.datetime64):
        return t.astype("datetime64[s]")
    if isinstance(t, datetime):
        if t.tzinfo is not None:
            t = t.astimezone(timezone.utc).replace(tzinfo=None)
        return np.datetime64(t, "s")
    return np.datetime64(str(t).replace("Z", ""), "s")


def hour_bracket(t):
    """Scene time t -> (t_lo, t_hi, fraction), e.g. 00:31:33 -> (00:00, 01:00, 0.5258)."""
    t = to_utc_naive(t)
    t_lo = t.astype("datetime64[h]").astype("datetime64[s]")
    frac = float((t - t_lo) / np.timedelta64(1, "s")) / 3600.0
    return t_lo, t_lo + np.timedelta64(1, "h"), frac


# =====================================================================================================
# 2. Download (Copernicus Marine)
# =====================================================================================================
def download_wind_subset(lon_min, lon_max, lat_min, lat_max, scene_time, out_dir,
                         dataset_id=DATASET_ID_MY, pad_deg=PAD_DEG, tag="scene", force=False):
    """Downloads the hourly fields before and after scene_time over the area; returns the NetCDF path.

    An existing file is reused (cache) unless force=True.
    """
    import copernicusmarine

    t_lo, t_hi, _ = hour_bracket(scene_time)
    os.makedirs(out_dir, exist_ok=True)
    fname = f"wind_{tag}_{str(t_lo)[:13].replace('-', '').replace('T', '_')}h.nc"
    path = os.path.join(out_dir, fname)
    if os.path.exists(path) and not force:
        return path
    copernicusmarine.subset(
        dataset_id=dataset_id,
        variables=[VAR_U, VAR_V],
        minimum_longitude=float(lon_min - pad_deg), maximum_longitude=float(lon_max + pad_deg),
        minimum_latitude=float(lat_min - pad_deg), maximum_latitude=float(lat_max + pad_deg),
        start_datetime=str(t_lo).replace("T", " "), end_datetime=str(t_hi).replace("T", " "),
        output_directory=out_dir, output_filename=fname,
    )
    return path


def open_wind(path_or_ds) -> xr.Dataset:
    ds = xr.open_dataset(path_or_ds) if isinstance(path_or_ds, (str, os.PathLike)) else path_or_ds
    for v in (VAR_U, VAR_V):
        if v not in ds:
            raise KeyError(f"Variable '{v}' not found. Variables: {list(ds.data_vars)}")
    ds = ds.sortby("latitude").sortby("longitude")
    return ds


# =====================================================================================================
# 3. Interpolation: time (linear) + space (bilinear)
# =====================================================================================================
def _time_interp(ds: xr.Dataset, scene_time):
    """u, v at scene_time (linear between the two hourly fields)."""
    t_lo, t_hi, frac = hour_bracket(scene_time)
    tol = np.timedelta64(2, "m")
    times = ds["time"].values.astype("datetime64[s]")

    def pick(t):
        i = int(np.argmin(np.abs(times - t)))
        if abs(times[i] - t) > tol:
            raise ValueError(f"No wind field at {t} in the file (times: {times[:4]}...). "
                             f"The download window must cover the acquisition time.")
        return ds.isel(time=i)

    lo = pick(t_lo)
    if frac < 1e-9:                           # acquisition exactly on the hour
        return lo[VAR_U], lo[VAR_V], frac
    hi = pick(t_hi)
    u = (1 - frac) * lo[VAR_U] + frac * hi[VAR_U]
    v = (1 - frac) * lo[VAR_V] + frac * hi[VAR_V]
    return u, v, frac


def wind_at_points(ds, scene_time, lons, lats, fallback=True, max_fallback_deg=MAX_FALLBACK_DEG):
    """Wind at the points (lons, lats). Returns dict(u, v, speed, dir_from, method, frac).

    method: 'bilinear' (4 valid neighbours), 'nearest' (NaN near the coast; nearest valid cell within
            max_fallback_deg), 'missing' (no valid cell).
    """
    ds = open_wind(ds)
    lons, lats = np.atleast_1d(np.asarray(lons, float)), np.atleast_1d(np.asarray(lats, float))
    u2d, v2d, frac = _time_interp(ds, scene_time)

    glon, glat = u2d["longitude"].values, u2d["latitude"].values
    if lons.min() < glon.min() or lons.max() > glon.max() or lats.min() < glat.min() or lats.max() > glat.max():
        raise ValueError(
            f"Patch centres outside the wind grid: points lon {lons.min():.3f}..{lons.max():.3f}, "
            f"lat {lats.min():.3f}..{lats.max():.3f} | grid lon {glon.min():.3f}..{glon.max():.3f}, "
            f"lat {glat.min():.3f}..{glat.max():.3f}. Increase pad_deg.")

    pts = dict(longitude=xr.DataArray(lons, dims="p"), latitude=xr.DataArray(lats, dims="p"))
    u = u2d.interp(**pts, method="linear").values.astype(float)
    v = v2d.interp(**pts, method="linear").values.astype(float)
    method = np.where(np.isfinite(u) & np.isfinite(v), "bilinear", "missing").astype(object)

    if fallback and (method == "missing").any():
        U, V = u2d.values, v2d.values
        ok = np.isfinite(U) & np.isfinite(V)
        LON, LAT = np.meshgrid(glon, glat)           # (lat, lon)
        for k in np.where(method == "missing")[0]:
            d = np.hypot(LON - lons[k], LAT - lats[k])
            d[~ok] = np.inf
            j = np.unravel_index(np.argmin(d), d.shape)
            if np.isfinite(d[j]) and d[j] <= max_fallback_deg:
                u[k], v[k], method[k] = U[j], V[j], "nearest"

    return dict(u=u, v=v, speed=uv_to_speed(u, v), dir_from=uv_to_dir_from(u, v), method=method, frac=frac)


def wind_labels_for_scene(patch_lons, patch_lats, scene_time, cache_dir, dataset_id=DATASET_ID_MY,
                          tag="scene", wind_nc=None, pad_deg=PAD_DEG):
    """Wind labels for all patch centres of one scene (downloads the field if needed).

    wind_nc: path of an already downloaded NetCDF file (skips the download).
    """
    patch_lons, patch_lats = np.asarray(patch_lons, float), np.asarray(patch_lats, float)
    if wind_nc is None:
        wind_nc = download_wind_subset(patch_lons.min(), patch_lons.max(), patch_lats.min(), patch_lats.max(),
                                       scene_time, cache_dir, dataset_id=dataset_id, pad_deg=pad_deg, tag=tag)
    out = wind_at_points(wind_nc, scene_time, patch_lons, patch_lats)
    out["wind_nc"] = str(wind_nc)
    return out
