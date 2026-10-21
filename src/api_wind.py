"""
api_wind.py - wind-field retrieval engine: area of interest + date range -> 5 km wind field.

Processing
  1. Validates the box (max 50 x 50 km) and builds a 100 m UTM grid of 5 km patches centred on it
     (same grid as gee/s1_patch_export.js).
  2. Finds Sentinel-1 IW GRDH VV slices in Google Earth Engine and groups them into acquisitions
     (platform, date, pass, relative orbit).
  3. Processes each acquisition exactly like the training data: VV to linear power, 10 m -> 100 m area
     mean, mosaic, back to dB; incidence from the 'angle' band; look azimuth = platform heading + 90 deg.
  4. Downloads the grid (ee.data.computePixels) and writes one GeoTIFF per 5 km patch.
  5. Runs infer_scene: ResNet direction, quality flag, 180 deg ambiguity removal with Copernicus Marine L4
     wind (reprocessed product, near-real-time product for recent dates), CMOD5.N speed and map.
     Optional alternative: NOAA GFS from Earth Engine (reference="gfs").

Usage
    import api_wind as A
    A.init(ee_project="your-gee-project")
    results = A.run_api((lon_min, lat_min, lon_max, lat_max), "2024-09-01", "2024-09-30",
                        model_dir=".../final_model", out_root=".../api_runs")
"""
from __future__ import annotations

import math
import os
import re
from argparse import Namespace
from datetime import datetime, timedelta, timezone

import numpy as np

PIXEL_M, PATCH_PX = 100, 50
PATCH_M = PIXEL_M * PATCH_PX
MAX_BOX_KM = 50.0
MAX_RANGE_DAYS = 60       # max span end - start (61 days inclusive); every acquisition is downloaded
NODATA = -9999.0


# =====================================================================================================
# 0. Earth Engine init
# =====================================================================================================
def init(ee_project=None, authenticate=True):
    import ee
    try:
        ee.Initialize(project=ee_project)
    except Exception:
        if not authenticate:
            raise
        ee.Authenticate()
        ee.Initialize(project=ee_project)
    return ee


# =====================================================================================================
# 1. Box and grid
# =====================================================================================================
def haversine_km(lon1, lat1, lon2, lat2):
    p = math.pi / 180
    a = math.sin((lat2 - lat1) * p / 2) ** 2 + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin((lon2 - lon1) * p / 2) ** 2
    return 12742.0 * math.asin(math.sqrt(a))


def box_size_km(box):
    lon_min, lat_min, lon_max, lat_max = box
    lat_c = (lat_min + lat_max) / 2
    return haversine_km(lon_min, lat_c, lon_max, lat_c), haversine_km(lon_min, lat_min, lon_min, lat_max)


def check_box(box, max_km=MAX_BOX_KM):
    lon_min, lat_min, lon_max, lat_max = map(float, box)
    if not (-180 <= lon_min < lon_max <= 180 and -90 <= lat_min < lat_max <= 90):
        raise ValueError(f"Invalid box {box}: expected lon_min < lon_max and lat_min < lat_max.")
    w, h = box_size_km(box)
    if w > max_km + 0.5 or h > max_km + 0.5:
        raise ValueError(f"Box is {w:.1f} x {h:.1f} km; the maximum is {max_km:.0f} x {max_km:.0f} km.")
    if w < 5 or h < 5:
        raise ValueError(f"Box is {w:.1f} x {h:.1f} km; the minimum is 5 x 5 km (one patch).")
    return w, h


def make_grid(box):
    """100 m UTM grid centred on the box with as many 5 km patches as needed to cover it (max 10 x 10)."""
    from pyproj import Transformer
    lon_min, lat_min, lon_max, lat_max = map(float, box)
    w, h = check_box(box)
    lon_c, lat_c = (lon_min + lon_max) / 2, (lat_min + lat_max) / 2
    zone = int(math.floor((lon_c + 180) / 6) + 1)
    epsg = (32600 if lat_c >= 0 else 32700) + zone
    xc, yc = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True).transform(lon_c, lat_c)
    n_cols = min(10, max(1, int(math.ceil(w * 1000 / PATCH_M - 1e-6))))
    n_rows = min(10, max(1, int(math.ceil(h * 1000 / PATCH_M - 1e-6))))
    X0 = xc - n_cols * PATCH_M / 2.0
    Y0 = yc + n_rows * PATCH_M / 2.0
    return dict(crs=f"EPSG:{epsg}", X0=X0, Y0=Y0, n_rows=n_rows, n_cols=n_cols,
                width=n_cols * PATCH_PX, height=n_rows * PATCH_PX, box=(lon_min, lat_min, lon_max, lat_max),
                box_km=(round(w, 1), round(h, 1)))


def acquisition_key(platform, t0_ms, orbit_pass, rel_orbit):
    d = datetime.fromtimestamp(t0_ms / 1000, tz=timezone.utc)
    return f"S1{platform}_{d:%Y%m%d}_{'ASC' if str(orbit_pass).upper().startswith('ASC') else 'DSC'}_RO{int(rel_orbit):03d}"


def group_slices(rows):
    """Groups slice metadata (id, t0, t1, pass, rel_orbit, platform, heading, cover) into acquisitions,
    sorted by AOI coverage (descending) and time."""
    groups = {}
    for p in rows:
        if p.get("cover", 0) <= 0.001:
            continue
        key = acquisition_key(p["platform"], p["t0"], p["pass"], p["rel_orbit"])
        g = groups.setdefault(key, dict(key=key, platform=p["platform"], pass_=p["pass"], rel_orbit=int(p["rel_orbit"]),
                                         slices=[], cover=0.0))
        hd = p.get("heading")
        has = hd is not None and abs(float(hd)) <= 360
        heading = float(hd) if has else (-12.0 if str(p["pass"]).upper().startswith("ASC") else -168.0)
        sid = str(p["id"]) if "/" in str(p["id"]) else "COPERNICUS/S1_GRD/" + str(p["id"])
        g["slices"].append(dict(id=sid, t0=p["t0"], t1=p["t1"], cover=p["cover"], heading=heading,
                                heading_source="metadata" if has else "fallback_from_pass",
                                azimuth=((heading + 90) % 360 + 360) % 360))
        g["cover"] = min(1.0, g["cover"] + p["cover"])
    out = list(groups.values())
    for g in out:
        g["t0"] = min(s["t0"] for s in g["slices"]); g["t1"] = max(s["t1"] for s in g["slices"])
    out.sort(key=lambda g: (-g["cover"], g["t0"]))
    return out


def tif_name(pnum, g):
    d = datetime.fromtimestamp(g["t0"] / 1000, tz=timezone.utc)
    return (f"P{pnum:03d}_S1{g['platform']}_{d:%Y%m%d}_{'ASC' if str(g['pass_']).upper().startswith('ASC') else 'DSC'}"
            f"_RO{g['rel_orbit']:03d}_{d:%H%M%S}.tif")


def write_patch_tifs(arr, grid, g, folder):
    """arr: (3, H, W) float32 [vv_db, incidence_deg, radar_azimuth_deg] on the grid -> one 3-band tif per 5 km patch."""
    import rasterio
    from rasterio.transform import from_origin
    os.makedirs(folder, exist_ok=True)
    paths = []
    for r in range(grid["n_rows"]):
        for c in range(grid["n_cols"]):
            pnum = r * grid["n_cols"] + c + 1
            sub = arr[:, r * PATCH_PX:(r + 1) * PATCH_PX, c * PATCH_PX:(c + 1) * PATCH_PX].astype("float32")
            sub = np.where(np.isfinite(sub), sub, NODATA).astype("float32")
            p = os.path.join(folder, tif_name(pnum, g))
            with rasterio.open(p, "w", driver="GTiff", height=PATCH_PX, width=PATCH_PX, count=3, dtype="float32",
                               crs=grid["crs"], nodata=NODATA,
                               transform=from_origin(grid["X0"] + c * PATCH_M, grid["Y0"] - r * PATCH_M, PIXEL_M, PIXEL_M)) as d:
                d.write(sub)
            paths.append(p)
    return paths


# =====================================================================================================
# 2. Sentinel-1 from Google Earth Engine
# =====================================================================================================
def _ee_box(box):
    import ee
    lon_min, lat_min, lon_max, lat_max = box
    return ee.Geometry.Polygon([[[lon_min, lat_max], [lon_max, lat_max], [lon_max, lat_min], [lon_min, lat_min],
                                 [lon_min, lat_max]]], "EPSG:4326", False)


def _s1_collection(aoi, start, end_exclusive):
    import ee
    return (ee.ImageCollection("COPERNICUS/S1_GRD").filterDate(start, end_exclusive).filterBounds(aoi)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.eq("resolution_meters", 10)))


def _parse_range(start_date, end_date):
    d0 = datetime.strptime(start_date, "%Y-%m-%d")
    d1 = datetime.strptime(end_date or start_date, "%Y-%m-%d")
    if d1 < d0:
        raise ValueError(f"End date {end_date} is before start date {start_date}.")
    if (d1 - d0).days > MAX_RANGE_DAYS:
        raise ValueError(f"Date range is {(d1 - d0).days + 1} days; the maximum is {MAX_RANGE_DAYS + 1} days.")
    return d0, d1


def _meta_fc(col, aoi, area, with_heading=True):
    import ee
    def meta(img):
        hd = (ee.Algorithms.If(img.propertyNames().contains("platformHeading"), img.get("platformHeading"), 9999)
              if with_heading else 9999)
        return ee.Feature(None, dict(
            id=img.id(), t0=img.get("system:time_start"), t1=img.get("system:time_end"),
            **{"pass": img.get("orbitProperties_pass")}, rel_orbit=img.get("relativeOrbitNumber_start"),
            platform=img.get("platform_number"), heading=hd,
            cover=img.geometry().intersection(aoi, ee.ErrorMargin(1)).area(1).divide(area)))
    return ee.FeatureCollection(col.map(meta)).getInfo()


def find_acquisitions(box, start_date, end_date=None):
    """All acquisitions between start and end (UTC, inclusive), in time order."""
    aoi = _ee_box(box)
    d0, d1 = _parse_range(start_date, end_date)
    col = _s1_collection(aoi, d0.strftime("%Y-%m-%d"), (d1 + timedelta(days=1)).strftime("%Y-%m-%d"))
    gs = group_slices([f["properties"] for f in _meta_fc(col, aoi, aoi.area(1))["features"]])
    return sorted(gs, key=lambda g: g["t0"])


def available_dates(box, start_date, end_date=None, window_days=15):
    """Acquisitions within +-window_days of the range: [(date_time_utc, coverage, key), ...]."""
    aoi = _ee_box(box)
    d0, d1 = _parse_range(start_date, end_date)
    col = _s1_collection(aoi, (d0 - timedelta(days=window_days)).strftime("%Y-%m-%d"),
                         (d1 + timedelta(days=window_days + 1)).strftime("%Y-%m-%d"))
    gs = group_slices([f["properties"] for f in _meta_fc(col, aoi, aoi.area(1), with_heading=False)["features"]])
    return sorted([(datetime.fromtimestamp(g["t0"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M"), round(g["cover"], 2), g["key"])
                   for g in gs])


def build_product(g, grid):
    import ee
    proj = ee.Projection(grid["crs"], [PIXEL_M, 0, grid["X0"], 0, -PIXEL_M, grid["Y0"]])
    per = []
    for s in g["slices"]:
        img = ee.Image(s["id"])
        vv_lin = img.select("VV").multiply(math.log(10) / 10).exp().rename("vv_lin")
        inc = img.select("angle").rename("incidence_deg")
        azi = inc.multiply(0).add(s["azimuth"]).rename("radar_azimuth_deg")
        per.append(vv_lin.addBands(inc).addBands(azi)
                   .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=1024).reproject(proj))
    mosaic = ee.ImageCollection(per).mosaic()
    vv_db = mosaic.select("vv_lin").log10().multiply(10).rename("vv_db")
    return (vv_db.addBands(mosaic.select(["incidence_deg", "radar_azimuth_deg"]))
            .updateMask(vv_db.mask()).toFloat().unmask(NODATA))


def download_grid(product, grid):
    import ee
    arr = ee.data.computePixels({
        "expression": product, "fileFormat": "NUMPY_NDARRAY",
        "grid": {"dimensions": {"width": grid["width"], "height": grid["height"]},
                 "affineTransform": {"scaleX": PIXEL_M, "shearX": 0, "translateX": grid["X0"],
                                     "shearY": 0, "scaleY": -PIXEL_M, "translateY": grid["Y0"]},
                 "crsCode": grid["crs"]}})
    out = np.stack([np.asarray(arr[b], np.float32) for b in ("vv_db", "incidence_deg", "radar_azimuth_deg")])
    return out


# =====================================================================================================
# 3. Reference wind for the 180 deg ambiguity
# =====================================================================================================
GFS_ID = "NOAA/GFS0P25"
GFS_BANDS = ("u_component_of_wind_10m_above_ground", "v_component_of_wind_10m_above_ground")
GFS_START_MS = int(datetime(2015, 7, 1, tzinfo=timezone.utc).timestamp() * 1000)
CFS_ID = "NOAA/CFSV2/FOR6H_HARMONIZED"
CFS_BANDS = ("u-component_of_wind_height_above_ground", "v-component_of_wind_height_above_ground")
HOUR_MS = 3600 * 1000


def bracket(times_ms, t_ms):
    """Indices and weights for linear interpolation in time at t_ms ([(i, w), ...], nearest only if not bracketed)."""
    times = np.asarray(times_ms, float)
    before, after = np.where(times <= t_ms)[0], np.where(times >= t_ms)[0]
    if len(before) and len(after):
        i0, i1 = before[np.argmax(times[before])], after[np.argmin(times[after])]
        if times[i1] == times[i0]:
            return [(int(i0), 1.0)]
        w1 = (t_ms - times[i0]) / (times[i1] - times[i0])
        return [(int(i0), 1.0 - w1), (int(i1), float(w1))]
    if len(times) == 0:
        return []
    return [(int(np.argmin(np.abs(times - t_ms))), 1.0)]


def select_gfs(rows, t_ms):
    """rows: [(index, creation_time, forecast_time), ...] -> [(index, weight), ...] from the latest model run
    issued before t_ms that brackets t_ms (shortest lead time)."""
    runs = sorted({r[1] for r in rows if r[1] <= t_ms}, reverse=True)
    for run in runs:
        sub = [r for r in rows if r[1] == run]
        times = [r[2] for r in sub]
        if min(times) <= t_ms <= max(times):
            return [(sub[i][0], w) for i, w in bracket(times, t_ms)], run
    return [], None


def _sample_uv(img, bands, lons, lats):
    import ee
    fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(x), float(y)]), {"i": i})
                               for i, (x, y) in enumerate(zip(lons, lats))])
    feats = (img.select(list(bands), ["u", "v"]).resample("bilinear")
             .sampleRegions(collection=fc, properties=["i"], scale=1000, geometries=False).getInfo()["features"])
    u, v = np.full(len(lons), np.nan), np.full(len(lons), np.nan)
    for f in feats:
        p = f["properties"]
        u[int(p["i"])], v[int(p["i"])] = p.get("u", np.nan), p.get("v", np.nan)
    return u, v


def _weighted(col_id, parts, bands):
    import ee
    img = None
    for idx, w in parts:
        im = ee.Image(f"{col_id}/{idx}").select(list(bands)).multiply(w)
        img = im if img is None else img.add(im)
    return img


def gfs_reference(lons, lats, t_ms):
    """10 m wind (u, v) from NOAA GFS at the scene time (CFSv2 before July 2015 or if GFS is missing)."""
    import ee
    if t_ms >= GFS_START_MS:
        col = (ee.ImageCollection(GFS_ID).filterDate(ee.Date(t_ms - 27 * HOUR_MS), ee.Date(t_ms + 3 * HOUR_MS + 1))
               .filter(ee.Filter.lte("creation_time", t_ms))
               .filter(ee.Filter.gte("creation_time", t_ms - 24 * HOUR_MS))
               .filter(ee.Filter.gte("forecast_time", t_ms - 3 * HOUR_MS))
               .filter(ee.Filter.lte("forecast_time", t_ms + 3 * HOUR_MS)))
        rows = col.reduceColumns(ee.Reducer.toList(3), ["system:index", "creation_time", "forecast_time"]).get("list").getInfo()
        parts, run = select_gfs([(r[0], float(r[1]), float(r[2])) for r in rows], t_ms)
        if parts:
            u, v = _sample_uv(_weighted(GFS_ID, parts, GFS_BANDS), GFS_BANDS, lons, lats)
            run_s = datetime.fromtimestamp(run / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H UTC")
            return u, v, "NOAA GFS 0.25 deg", f"run {run_s}, interpolated to scene time"
    col = ee.ImageCollection(CFS_ID).filterDate(ee.Date(t_ms - 6 * HOUR_MS), ee.Date(t_ms + 6 * HOUR_MS + 1))
    rows = col.reduceColumns(ee.Reducer.toList(2), ["system:index", "system:time_start"]).get("list").getInfo()
    parts = [(rows[i][0], w) for i, w in bracket([float(r[1]) for r in rows], t_ms)]
    if not parts:
        raise RuntimeError("no GFS or CFSv2 wind found in Earth Engine for this time")
    u, v = _sample_uv(_weighted(CFS_ID, parts, CFS_BANDS), CFS_BANDS, lons, lats)
    return u, v, "NOAA CFSv2", "6-hourly, interpolated to scene time"


def gfs_provider(log=print):
    """Reference-wind provider for infer_scene using NOAA GFS / CFSv2 from Earth Engine (no extra sign-in)."""
    def provider(lons, lats, scene_time, scene_id):
        from wind_labels import to_utc_naive
        t_ms = int(to_utc_naive(scene_time).astype("datetime64[ms]").astype("int64"))
        u, v, src, detail = gfs_reference(lons, lats, t_ms)
        if not np.isfinite(u).any():
            raise RuntimeError(f"{src}: no valid wind values over the area")
        speed = np.hypot(u, v)
        dir_from = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0
        log(f"  Reference wind: {src} ({detail})")
        return dict(u=u, v=v, speed=speed, dir_from=dir_from, source=src, detail=detail, wind_nc=src)
    return provider


def copernicus_provider(cache_dir, log=print):
    """Reference-wind provider using Copernicus Marine L4 (reprocessed, then near-real-time); needs a login."""
    import wind_labels as WL

    def provider(lons, lats, scene_time, scene_id):
        last = None
        for ds in (WL.DATASET_ID_MY, WL.DATASET_ID_NRT):
            try:
                tag = f"{scene_id}_{'nrt' if '_nrt_' in ds else 'my'}"
                out = WL.wind_labels_for_scene(lons, lats, scene_time, cache_dir, dataset_id=ds, tag=tag)
                out["source"] = "Copernicus Marine L4"
                log(f"  Copernicus dataset: {ds}")
                return out
            except Exception as e:
                last = e
                log(f"  Copernicus {ds} unavailable ({str(e)[:120]}); trying next product")
        raise RuntimeError(f"No Copernicus Marine wind available: {last}")
    return provider


# =====================================================================================================
# 4. Main entry point
# =====================================================================================================
def _process_acquisition(g, arr, grid, model_dir, out_root, provider, show, log):
    """Writes the patch GeoTIFFs of one acquisition and runs infer_scene on them."""
    import infer_scene as I
    valid = float(np.mean(arr[0] > NODATA / 2))
    t = datetime.fromtimestamp(g["t0"] / 1000, tz=timezone.utc)
    log(f"\n##### {g['key']}  {t:%Y-%m-%d %H:%M:%S} UTC | box cover {g['cover']:.0%} | valid pixels {valid:.0%} | "
        f"look azimuth {g['slices'][0]['azimuth']:.1f} deg ({g['slices'][0]['heading_source']})")
    lo = grid["box"]
    run_dir = os.path.join(out_root, f"{g['key']}_{t:%H%M%S}_{lo[0]:.2f}_{lo[1]:.2f}_{lo[2]:.2f}_{lo[3]:.2f}")
    tif_dir = os.path.join(run_dir, "tifs", "api_box")
    for f in (os.listdir(tif_dir) if os.path.isdir(tif_dir) else []):
        if f.endswith(".tif"):
            os.remove(os.path.join(tif_dir, f))
    write_patch_tifs(arr, grid, g, tif_dir)
    a = Namespace(tif_dir=tif_dir, model_dir=model_dir, out_dir=run_dir, cache_dir=os.path.join(out_root, "wind_cache"),
                  wind_nc=None, dataset_id=None, max_nbr_dev=20.0, min_aniso=0.20, min_area_aniso=0.21, no_land_mask=False)
    V = I.run(a, wind_provider=provider)
    S = V.iloc[0].to_dict() if len(V) else {}
    maps = sorted(os.path.join(run_dir, f) for f in os.listdir(run_dir) if f.startswith("map_") and f.endswith(".png"))
    csvs = sorted(os.path.join(run_dir, f) for f in os.listdir(run_dir) if f.startswith("wind_field_") and f.endswith(".csv"))
    if show and maps:
        try:
            from IPython.display import Image, display
            display(Image(maps[0]))
        except Exception:
            pass
    return dict(run_dir=run_dir, map=maps[0] if maps else None, csv=csvs[0] if csvs else None, summary=S,
                acquisition=g["key"], time_utc=f"{t:%Y-%m-%d %H:%M:%S}", cover=g["cover"])


def run_api(box, start_date, end_date=None, model_dir=None, out_root=None, max_scenes=6, min_cover=0.5,
            reference="copernicus", provider=None, acquisitions=None, fetch=None, show=True, log=print):
    """Retrieves the wind field for every Sentinel-1 acquisition covering the box.

    box: (lon_min, lat_min, lon_max, lat_max); start_date, end_date: 'YYYY-MM-DD' (UTC, inclusive).
    Acquisitions covering at least min_cover of the box are processed (at most max_scenes).
    Returns one dict per acquisition with run_dir, csv, map, summary, acquisition, time_utc and cover.
    reference: "copernicus" (Copernicus Marine, default; needs copernicusmarine.login()) or "gfs" (NOAA GFS from Earth Engine).
    For offline tests, pass acquisitions=[...], fetch(grid, group) -> array and provider instead of using GEE."""
    if model_dir is None or out_root is None:
        raise ValueError("model_dir and out_root are required.")
    _parse_range(start_date, end_date)
    grid = make_grid(box)
    rng = start_date if not end_date or end_date == start_date else f"{start_date} .. {end_date}"
    log(f"Box {grid['box_km'][0]} x {grid['box_km'][1]} km | grid {grid['n_rows']} x {grid['n_cols']} patches (5 km) | "
        f"{grid['crs']} | dates {rng}")
    groups = acquisitions if acquisitions is not None else find_acquisitions(box, start_date, end_date)
    if not groups:
        near = [] if acquisitions is not None else available_dates(box, start_date, end_date)
        msg = f"{rng}: no Sentinel-1 IW acquisition found over this box."
        msg += (" Nearest acquisitions (UTC, coverage, key):\n" + "\n".join(f"   {d}  cover {c:.0%}  {k}" for d, c, k in near)
                if near else " None within 15 days of the range either.")
        raise SystemExit(msg)
    log("Found: " + " | ".join(f"{x['key']} ({x['cover']:.0%})" for x in groups))
    use = [x for x in groups if x["cover"] >= min_cover]
    skipped = [x["key"] for x in groups if x["cover"] < min_cover]
    if skipped:
        log(f"Skipped (cover below {min_cover:.0%}): {skipped}")
    if not use:
        raise SystemExit("No acquisition covers enough of the box. Move the box or widen the date range.")
    if len(use) > max_scenes:
        log(f"{len(use)} acquisitions found; processing the first {max_scenes} (increase max_scenes for more).")
        use = use[:max_scenes]
    if provider is None:
        provider = (gfs_provider(log) if reference == "gfs"
                    else copernicus_provider(os.path.join(out_root, "wind_cache"), log))
    results = []
    for g in use:
        try:
            arr = fetch(grid, g) if fetch is not None else download_grid(build_product(g, grid), grid)
            results.append(_process_acquisition(g, arr, grid, model_dir, out_root, provider, show, log))
        except SystemExit:
            raise
        except Exception as e:
            log(f"  {g['key']} failed: {e}")
            results.append(dict(acquisition=g["key"], error=str(e)))
    rows = []
    for r in results:
        S = r.get("summary", {})
        rows.append(dict(acquisition=r["acquisition"], time_utc=r.get("time_utc", ""), cover=round(r.get("cover", np.nan), 2),
                         n_sea=S.get("n_sea"), n_good=S.get("n_good"), area_texture_ok=S.get("area_texture_ok"),
                         speed_cmod5n=round(S.get("mean_speed_cmod5n", np.nan), 2),
                         speed_reference=round(S.get("mean_speed_reference", np.nan), 2),
                         dir_bias_vs_reference=round(S.get("all_dir_bias", np.nan), 1), reference=S.get("reference", ""),
                         error=r.get("error", "")))
    try:
        import pandas as pd
        T = pd.DataFrame(rows)
        pd.set_option("display.width", 220)
        log("\n=== API SUMMARY ===\n" + T.to_string(index=False))
        T.to_csv(os.path.join(out_root, "last_api_summary.csv"), index=False)
    except Exception:
        pass
    log(f"Results: {out_root}")
    return results

