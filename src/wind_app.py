"""
wind_app.py - user-facing wind-field API for Google Colab.

Two steps for the user:
  1. show()            draws a map; the user draws the area of interest (max 50 x 50 km) over the sea.
  2. run(start, end)   retrieves the wind field for every Sentinel-1 acquisition between the two dates
                       over the drawn area and displays a map for each one.

    import wind_app as W
    W.configure(model_dir=".../final_model", out_root=".../api_runs")
    W.show(center=(12.0, 80.4))
    results = W.run("2024-09-20", "2024-09-30")

Programmatic use (box given as coordinates):
    results = W.get_wind_field((lon_min, lat_min, lon_max, lat_max), "2024-09-20", "2024-09-30")

Outputs per acquisition (in out_root/<acquisition>_<box>/):
    wind_vectors.geojson   one point per 5 km sea patch (GeoJSON FeatureCollection)
    wind_field_*.csv       full per-patch table
    map_*.png              wind-vector map, land masked
    summary_*.json         acquisition summary and comparison with the reference wind (NOAA GFS)
"""
from __future__ import annotations

import json
import math
import os

import api_wind as A
import box_picker as BP

_STATE = {"model_dir": None, "out_root": None, "results": []}


def configure(model_dir, out_root):
    """Sets the trained model folder and the output folder."""
    _STATE["model_dir"], _STATE["out_root"] = model_dir, out_root
    os.makedirs(out_root, exist_ok=True)


def _num(x, nd=2):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if not math.isfinite(x) else round(x, nd)


def to_geojson(csv_path, out_path, acquisition=None):
    """Writes the sea patches of a wind_field CSV as a GeoJSON FeatureCollection of points.

    Properties: patch_id, time_utc, wind_from_deg (meteorological, direction the wind blows from),
    wind_speed_ms (CMOD5.N), u_ms, v_ms (eastward / northward components), quality ('good' / 'low'),
    ambiguity_removed (False when no reference wind was available and only the axis is known).
    """
    import pandas as pd
    T = pd.read_csv(csv_path)
    sea = T[T["status"] == "kept"] if "status" in T.columns else T
    feats = []
    for _, r in sea.iterrows():
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [_num(r["lon"], 5), _num(r["lat"], 5)]},
            "properties": {
                "patch_id": str(r.get("patch_id", "")),
                "time_utc": f"{r.get('date_utc', '')} {r.get('time_utc', '')}".strip(),
                "wind_from_deg": _num(r.get("dir_from"), 1),
                "wind_speed_ms": _num(r.get("speed_cmod5n"), 2),
                "u_ms": _num(r.get("u"), 2),
                "v_ms": _num(r.get("v"), 2),
                "quality": str(r.get("quality", "")),
                "ambiguity_removed": str(r.get("dealiased", "")).lower() in ("true", "1", "1.0"),
            }})
    fc = {"type": "FeatureCollection", "properties": {"acquisition": acquisition, "patch_size_km": 5,
                                                      "source": "Sentinel-1 IW GRDH VV; ResNet direction; CMOD5.N speed"},
          "features": feats}
    with open(out_path, "w") as f:
        json.dump(fc, f, indent=1)
    return out_path


def get_wind_field(box, start_date, end_date=None, model_dir=None, out_root=None, show=True, **kwargs):
    """Wind field for every Sentinel-1 acquisition over `box` between start_date and end_date (UTC, inclusive).

    box: (lon_min, lat_min, lon_max, lat_max), at most 50 x 50 km.
    Returns a list with one dict per acquisition: acquisition, time_utc, cover, geojson, csv, map, run_dir, summary
    (or acquisition and error if that acquisition failed).
    """
    model_dir = model_dir or _STATE["model_dir"]
    out_root = out_root or _STATE["out_root"]
    if not model_dir or not out_root:
        raise ValueError("Call configure(model_dir, out_root) first or pass model_dir and out_root.")
    results = A.run_api(tuple(map(float, box)), start_date, end_date, model_dir=model_dir, out_root=out_root,
                        show=show, **kwargs)
    for r in results:
        if r.get("csv"):
            r["geojson"] = to_geojson(r["csv"], os.path.join(r["run_dir"], "wind_vectors.geojson"), r["acquisition"])
    _STATE["results"] = results
    return results


def last_results():
    """Results of the most recent request (list of dicts, see get_wind_field)."""
    return list(_STATE["results"])


def zip_results(results=None, zip_path="wind_results.zip"):
    """Zips the output files (GeoJSON, CSV, maps, summaries) of the given or latest request; returns the zip path."""
    import zipfile
    results = last_results() if results is None else results
    keep = (".geojson", ".csv", ".png", ".json")
    n = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for r in results:
            run_dir = r.get("run_dir")
            if not run_dir or not os.path.isdir(run_dir):
                continue
            for f in sorted(os.listdir(run_dir)):
                if f.endswith(keep):
                    z.write(os.path.join(run_dir, f), arcname=os.path.join(os.path.basename(run_dir), f)); n += 1
    if n == 0:
        os.remove(zip_path)
        raise ValueError("No results to download: run Step 2 first.")
    return zip_path


def _print_outputs(results):
    ok = [r for r in results if r.get("geojson")]
    if not ok:
        return
    print("\nOutput files:")
    for r in ok:
        print(f"  {r['acquisition']}  {r['time_utc']} UTC\n    {r['geojson']}\n    {r['csv']}\n    {r['map']}")


# ---------------------------------------------------------------------------------------------------------
# Interactive use
# ---------------------------------------------------------------------------------------------------------
def show(center=(12.0, 80.4), zoom=8):
    """Step 1: map on which the user draws the area of interest (max 50 x 50 km)."""
    BP.show(center=center, zoom=zoom)


def run(start_date, end_date):
    """Step 2: wind field over the area drawn in show(), for start_date..end_date (UTC, 'YYYY-MM-DD', inclusive)."""
    box = BP.get_box()
    if box is None:
        print("No area of interest selected. Draw a rectangle on the map in Step 1, then run this step again.")
        return []
    print(f"Area of interest: {', '.join(f'{v:.4f}' for v in box)}  |  dates: {start_date} to {end_date}\n")
    try:
        results = get_wind_field(box, str(start_date), str(end_date))
    except (ValueError, SystemExit) as e:
        print(f"Error: {e}")
        return []
    _print_outputs(results)
    return results
