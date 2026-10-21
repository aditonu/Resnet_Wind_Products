"""
box_picker.py - interactive area-of-interest selection on a Leaflet map (Google Colab).

The rectangle tool is limited while drawing: the box can never exceed MAX_KM x MAX_KM, and its size is
shown live. Drawing a new box replaces the previous one. The selected box is passed to Python through a
Colab callback and returned by get_box() as (lon_min, lat_min, lon_max, lat_max).

    import box_picker as BP
    BP.show(center=(12.0, 80.4), zoom=8)
    box = BP.get_box()
"""
from __future__ import annotations

import math

MAX_KM = 50.0
SAFETY_KM = 0.2                            # keeps drawn boxes safely inside api_wind.check_box
KM_PER_DEG = 6371.0 * math.pi / 180.0      # same Earth radius as api_wind.haversine_km

CDN = dict(leaflet_css="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css",
           leaflet_js="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js",
           draw_css="https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.css",
           draw_js="https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.js")

_BOX = {"box": None}


def _set_box(lon_min, lat_min, lon_max, lat_max):
    _BOX["box"] = (float(lon_min), float(lat_min), float(lon_max), float(lat_max))


def get_box():
    """Last box drawn on the map, or None."""
    return _BOX["box"]


def build_html(center=(12.0, 80.4), zoom=8, max_km=MAX_KM, libs=None, height_px=520):
    """HTML of the map with the size-limited rectangle tool."""
    libs = libs or CDN
    lim = max_km - SAFETY_KM
    return f"""
<link rel="stylesheet" href="{libs['leaflet_css']}"/>
<link rel="stylesheet" href="{libs['draw_css']}"/>
<script src="{libs['leaflet_js']}"></script>
<script src="{libs['draw_js']}"></script>
<div style="font:13px sans-serif;margin:4px 0">Draw the area of interest over the sea with the rectangle tool (&#9723;).
 Maximum size: <b>{max_km:.0f} x {max_km:.0f} km</b>. <span id="bp_info" style="color:#b00"></span></div>
<div id="bp_map" style="height:{height_px}px;width:100%"></div>
<div id="bp_out" style="font:13px monospace;margin-top:4px">Area of interest: not selected</div>
<script>
(function() {{
  const LIM = {lim}, KM = {KM_PER_DEG};
  const map = L.map('bp_map').setView([{center[0]}, {center[1]}], {zoom});
  const esri = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}',
      {{maxZoom: 18, attribution: 'Esri'}}).addTo(map);
  const carto = L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}.png',
      {{maxZoom: 18, attribution: 'CARTO'}});
  L.control.layers({{'Satellite (Esri)': esri, 'Map (CARTO)': carto}}).addTo(map);
  const drawn = new L.FeatureGroup().addTo(map);

  function sizeKm(b) {{       // width at the centre latitude, height along the meridian (as api_wind.box_size_km)
    const latc = (b.getSouth() + b.getNorth()) / 2;
    return [KM * Math.cos(latc * Math.PI / 180) * (b.getEast() - b.getWest()), KM * (b.getNorth() - b.getSouth())];
  }}
  // Keep the dragged corner within LIM km of the start corner.
  const orig = L.Draw.Rectangle.prototype._drawShape;
  L.Draw.Rectangle.prototype._drawShape = function(latlng) {{
    const s = this._startLatLng;
    const dLat = LIM / KM;
    const lat = Math.max(Math.min(latlng.lat, s.lat + dLat), s.lat - dLat);
    const latc = (lat + s.lat) / 2;
    const dLon = LIM / (KM * Math.cos(latc * Math.PI / 180));
    const lng = Math.max(Math.min(latlng.lng, s.lng + dLon), s.lng - dLon);
    const capped = (lat !== latlng.lat) || (lng !== latlng.lng);
    orig.call(this, L.latLng(lat, lng));
    const wh = sizeKm(L.latLngBounds(s, L.latLng(lat, lng)));
    document.getElementById('bp_info').textContent =
      wh[0].toFixed(1) + ' x ' + wh[1].toFixed(1) + ' km' + (capped ? '  (maximum size)' : '');
  }};
  new L.Control.Draw({{
    draw: {{rectangle: {{shapeOptions: {{color: '#ff3333', weight: 2}}}},
            polygon: false, polyline: false, circle: false, marker: false, circlemarker: false}},
    edit: false
  }}).addTo(map);
  map.on(L.Draw.Event.CREATED, function(e) {{
    drawn.clearLayers(); drawn.addLayer(e.layer);
    const b = e.layer.getBounds(), wh = sizeKm(b);
    const box = [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()];
    document.getElementById('bp_out').textContent = 'Area of interest (lon_min, lat_min, lon_max, lat_max): ' +
      box.map(v => v.toFixed(4)).join(', ') + '   |   ' + wh[0].toFixed(1) + ' x ' + wh[1].toFixed(1) + ' km';
    window.__bp_box = box;
    if (window.google && google.colab && google.colab.kernel) {{
      google.colab.kernel.invokeFunction('box_picker.set_box', box, {{}});
    }}
  }});
}})();
</script>"""


def register_callbacks():
    try:
        from google.colab import output
        output.register_callback("box_picker.set_box", _set_box)
        return True
    except ImportError:
        return False


def show(center=(12.0, 80.4), zoom=8, max_km=MAX_KM):
    from IPython.display import HTML, display
    if not register_callbacks():
        print("Not running in Colab: the drawn box is not passed to Python.")
    display(HTML(build_html(center, zoom, max_km)))
