/**
 * Sentinel-1 IW GRDH (VV) patch export for SAR wind-direction retrieval
 * =====================================================================
 * Google Earth Engine Code Editor script.
 *
 * Purpose
 *   Builds the training / test inputs for the ResNet wind-direction model
 *   (Zanchetta & Zecchetto, 2021, Remote Sensing of Environment 253, 112178).
 *
 * Processing
 *   1. Selects Sentinel-1 GRDH acquisitions (IW mode, VV) intersecting a
 *      50 km x 50 km area of interest within a date range.
 *   2. Averages the 10 m backscatter to 100 m in linear power units on a UTM
 *      grid centred on the area of interest.
 *   3. Splits the grid into 10 x 10 patches of 5 km x 5 km (50 x 50 pixels).
 *   4. Exports one GeoTIFF per patch with three float32 bands:
 *        vv_db              VV backscatter (dB)
 *        incidence_deg      ellipsoid incidence angle (deg)
 *        radar_azimuth_deg  radar look azimuth (deg clockwise from north)
 *
 * Output file names (parsed by build_dataset.py, do not change the pattern)
 *   <PatchID>_S1<X>_<YYYYMMDD>_<ASC|DSC>_RO<orbit>_<HHMMSS>.tif
 *   e.g. C001_S1A_20240305_DSC_RO092_003133.tif
 *   Patch IDs C001..C100 are row-major from the north-west corner.
 *
 * Usage
 *   1. Set AOI, START_DATE and END_DATE in Section 1.
 *   2. Run, check the acquisition list in the Console, then start the export
 *      tasks in the Tasks tab (see the note at the end of the file).
 *   3. Copy the exported GeoTIFFs to  tifs/<scene_name>/  in Google Drive.
 *
 * Notes
 *   - The look azimuth is not provided per pixel in COPERNICUS/S1_GRD; it is
 *     derived as platformHeading + 90 deg (right-looking SAR) and is constant
 *     per slice.
 *   - Masked pixels are written as NODATA (-9999).
 */


// =========================================================================
// 1. USER SETTINGS
// =========================================================================
var AOI_BOUNDS = [80.2000, 14.7000, 80.6648, 15.1499];   // [lon_min, lat_min, lon_max, lat_max], ~50 x 50 km
var START_DATE = '2024-03-01';                           // UTC, inclusive
var END_DATE   = '2024-03-07';                           // UTC, inclusive

var ORBIT_PASS = 'ANY';                  // 'ANY' | 'ASCENDING' | 'DESCENDING'
var FORCE_ACQUISITION_KEY    = '';       // e.g. 'S1A_20240305_DSC_RO092' (keys are printed in the Console)
var PROCESS_ALL_ACQUISITIONS = false;    // false: export only the acquisition with the largest AOI coverage

var DRIVE_FOLDER        = 'S1_patches';  // patch images go to <DRIVE_FOLDER>_images
var EXPORT_PATCH_IMAGES = true;
var EXPORT_PIXEL_TABLE  = false;         // optional per-pixel CSV (not used by the Python pipeline)
var PATCH_IMAGE_TYPE    = 'DATA';        // 'DATA' (3-band float32) | 'QUICKLOOK' (8-bit VV for viewing)

var PIXEL_M  = 100;                      // output resolution (m)
var PATCH_PX = 50;                       // patch side in pixels (50 x 100 m = 5 km)
var N_ROWS = 10, N_COLS = 10;            // patches per side

var VIS_MIN = -25, VIS_MAX = -5;         // dB display range
var NODATA = -9999;
var EDGE_SHRINK_M = 20;                  // export regions are shrunk by 20 m so each patch is exactly 50 x 50 px


// =========================================================================
// 2. HELPERS
// =========================================================================
var DEG = Math.PI / 180;
function sinh(x)  { return (Math.exp(x) - Math.exp(-x)) / 2; }
function cosh(x)  { return (Math.exp(x) + Math.exp(-x)) / 2; }
function atanh(x) { return 0.5 * Math.log((1 + x) / (1 - x)); }
function pad(n, w) { var s = String(n); while (s.length < w) { s = '0' + s; } return s; }
function round(v, d) { var m = Math.pow(10, d); return Math.round(v * m) / m; }

// WGS84 <-> UTM (Krueger series). Used client-side to build the metric patch grid.
var WGS84_A = 6378137.0, WGS84_F = 1 / 298.257223563, UTM_K0 = 0.9996;
var _n = WGS84_F / (2 - WGS84_F);
var _n2 = _n * _n, _n3 = _n2 * _n, _n4 = _n3 * _n;
var _A = WGS84_A / (1 + _n) * (1 + _n2 / 4 + _n4 / 64);
var _alpha = [_n / 2 - 2 * _n2 / 3 + 5 * _n3 / 16 + 41 * _n4 / 180,
              13 * _n2 / 48 - 3 * _n3 / 5 + 557 * _n4 / 1440,
              61 * _n3 / 240 - 103 * _n4 / 140,
              49561 * _n4 / 161280];
var _beta  = [_n / 2 - 2 * _n2 / 3 + 37 * _n3 / 96 - _n4 / 360,
              _n2 / 48 + _n3 / 15 - 437 * _n4 / 1440,
              17 * _n3 / 480 - 37 * _n4 / 840,
              4397 * _n4 / 161280];
var _delta = [2 * _n - 2 * _n2 / 3 - 2 * _n3 + 116 * _n4 / 45,
              7 * _n2 / 3 - 8 * _n3 / 5 - 227 * _n4 / 45,
              56 * _n3 / 15 - 136 * _n4 / 35,
              4279 * _n4 / 630];

function lonLatToUtm(lon, lat, zone, falseN) {
  var lon0 = ((zone - 1) * 6 - 180 + 3) * DEG;
  var phi = lat * DEG, dl = lon * DEG - lon0;
  var e = 2 * Math.sqrt(_n) / (1 + _n);
  var t = sinh(atanh(Math.sin(phi)) - e * atanh(e * Math.sin(phi)));
  var xi0 = Math.atan2(t, Math.cos(dl));
  var eta0 = atanh(Math.sin(dl) / Math.sqrt(1 + t * t));
  var xi = xi0, eta = eta0;
  for (var j = 1; j <= 4; j++) {
    xi  += _alpha[j - 1] * Math.sin(2 * j * xi0) * cosh(2 * j * eta0);
    eta += _alpha[j - 1] * Math.cos(2 * j * xi0) * sinh(2 * j * eta0);
  }
  return { x: 500000 + UTM_K0 * _A * eta, y: falseN + UTM_K0 * _A * xi };
}

function utmToLonLat(x, y, zone, falseN) {
  var lon0 = ((zone - 1) * 6 - 180 + 3) * DEG;
  var xi = (y - falseN) / (UTM_K0 * _A), eta = (x - 500000) / (UTM_K0 * _A);
  var xi0 = xi, eta0 = eta;
  for (var j = 1; j <= 4; j++) {
    xi0  -= _beta[j - 1] * Math.sin(2 * j * xi) * cosh(2 * j * eta);
    eta0 -= _beta[j - 1] * Math.cos(2 * j * xi) * sinh(2 * j * eta);
  }
  var chi = Math.asin(Math.sin(xi0) / cosh(eta0));
  var phi = chi;
  for (var k = 1; k <= 4; k++) { phi += _delta[k - 1] * Math.sin(2 * k * chi); }
  var lon = lon0 + Math.atan2(sinh(eta0), Math.cos(xi0));
  return { lon: lon / DEG, lat: phi / DEG };
}


// =========================================================================
// 3. PATCH GRID (UTM, centred on the AOI)
// =========================================================================
var LON_MIN = AOI_BOUNDS[0], LAT_MIN = AOI_BOUNDS[1], LON_MAX = AOI_BOUNDS[2], LAT_MAX = AOI_BOUNDS[3];
var AOI = ee.Geometry.Rectangle([LON_MIN, LAT_MIN, LON_MAX, LAT_MAX], 'EPSG:4326', false);
var AOI_AREA = AOI.area(1);

var CENTER_LON = (LON_MIN + LON_MAX) / 2, CENTER_LAT = (LAT_MIN + LAT_MAX) / 2;
var UTM_ZONE = Math.floor((CENTER_LON + 180) / 6) + 1;
var FALSE_N  = CENTER_LAT >= 0 ? 0 : 10000000;
var UTM_CRS  = 'EPSG:' + ((CENTER_LAT >= 0 ? 32600 : 32700) + UTM_ZONE);

var PATCH_M = PATCH_PX * PIXEL_M;
var GRID_W  = N_COLS * PATCH_M, GRID_H = N_ROWS * PATCH_M;
var ctr = lonLatToUtm(CENTER_LON, CENTER_LAT, UTM_ZONE, FALSE_N);
var X0 = ctr.x - GRID_W / 2;     // west edge (UTM easting)
var Y0 = ctr.y + GRID_H / 2;     // north edge (UTM northing)
var PROJ_100M = ee.Projection(UTM_CRS, [PIXEL_M, 0, X0, 0, -PIXEL_M, Y0]);

function utmRect(xmin, ymin, xmax, ymax, shrink) {
  return ee.Geometry.Rectangle([xmin + shrink, ymin + shrink, xmax - shrink, ymax - shrink], UTM_CRS, false);
}
var GRID_REGION = utmRect(X0, Y0 - GRID_H, X0 + GRID_W, Y0, EDGE_SHRINK_M);

var PATCHES = [], PATCH_IDS = [];
for (var pr = 0; pr < N_ROWS; pr++) {
  for (var pc = 0; pc < N_COLS; pc++) {
    var pid  = 'C' + pad(pr * N_COLS + pc + 1, 3);
    var xmin = X0 + pc * PATCH_M, xmax = xmin + PATCH_M;
    var ymax = Y0 - pr * PATCH_M, ymin = ymax - PATCH_M;
    PATCHES.push({
      id: pid, row: pr, col: pc, xmin: xmin, ymin: ymin, xmax: xmax, ymax: ymax,
      C: utmToLonLat((xmin + xmax) / 2, (ymin + ymax) / 2, UTM_ZONE, FALSE_N)
    });
    PATCH_IDS.push(pid);
  }
}

var patchOutlines = ee.FeatureCollection(PATCHES.map(function (p) {
  return ee.Feature(utmRect(p.xmin, p.ymin, p.xmax, p.ymax, 0), { patch_id: p.id });
}));

print('Patches:', PATCHES.length, '| pixels per patch:', PATCH_PX * PATCH_PX, '| CRS:', UTM_CRS);
Map.centerObject(AOI, 9);
Map.addLayer(AOI, { color: 'yellow' }, 'AOI', true);
Map.addLayer(patchOutlines.style({ color: 'red', fillColor: '00000000', width: 1 }), {}, 'Patch grid', true);


// =========================================================================
// 4. SENTINEL-1 ACQUISITIONS
// =========================================================================
var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterDate(START_DATE, ee.Date(END_DATE).advance(1, 'day'))   // filterDate end is exclusive
  .filterBounds(AOI)
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .filter(ee.Filter.eq('resolution_meters', 10));                // GRDH
if (ORBIT_PASS !== 'ANY') { s1 = s1.filter(ee.Filter.eq('orbitProperties_pass', ORBIT_PASS)); }

print('Sentinel-1 slices in date range:', s1.size());

var sliceMeta = ee.FeatureCollection(s1.map(function (img) {
  var cover = img.geometry().intersection(AOI, ee.ErrorMargin(1)).area(1).divide(AOI_AREA);
  return ee.Feature(null, {
    id: img.id(),
    t0: img.get('system:time_start'),
    t1: img.get('system:time_end'),
    pass: img.get('orbitProperties_pass'),
    rel_orbit: img.get('relativeOrbitNumber_start'),
    platform: img.get('platform_number'),
    heading: ee.Algorithms.If(img.propertyNames().contains('platformHeading'), img.get('platformHeading'), 9999),
    cover: cover
  });
}));

sliceMeta.evaluate(function (res, err) {
  if (err) { print('Error reading Sentinel-1 metadata:', err); return; }

  var rows = res.features.map(function (f) { return f.properties; })
                         .filter(function (p) { return p.cover > 0.001; });
  if (rows.length === 0) {
    print('No Sentinel-1 IW GRDH VV acquisition intersects the AOI in this date range.');
    return;
  }

  // Group slices into acquisitions (same platform, date, pass and relative orbit).
  var groups = {};
  rows.forEach(function (p) {
    var iso = new Date(p.t0).toISOString();
    var key = 'S1' + p.platform + '_' + iso.slice(0, 10).replace(/-/g, '') + '_' +
              (p.pass === 'ASCENDING' ? 'ASC' : 'DSC') + '_RO' + pad(p.rel_orbit, 3);
    if (!groups[key]) {
      groups[key] = { key: key, platform: p.platform, pass: p.pass, relOrbit: p.rel_orbit, slices: [], cover: 0 };
    }
    // Fallback headings are typical values at low latitudes when platformHeading is missing.
    var hasHeading = (p.heading !== undefined && p.heading !== null && Math.abs(p.heading) <= 360);
    var heading = hasHeading ? p.heading : (p.pass === 'ASCENDING' ? -12.0 : -168.0);
    groups[key].slices.push({
      id: (String(p.id).indexOf('/') >= 0 ? p.id : 'COPERNICUS/S1_GRD/' + p.id),
      t0: p.t0, t1: p.t1, cover: p.cover,
      azimuth: (((heading + 90) % 360) + 360) % 360           // look azimuth = heading + 90 deg
    });
    groups[key].cover = Math.min(1, groups[key].cover + p.cover);
  });

  var list = Object.keys(groups).map(function (k) { return groups[k]; });
  list.forEach(function (g) {
    g.t0 = Math.min.apply(null, g.slices.map(function (s) { return s.t0; }));
    g.t1 = Math.max.apply(null, g.slices.map(function (s) { return s.t1; }));
  });
  list.sort(function (a, b) { return (b.cover - a.cover) || (a.t0 - b.t0); });

  print('--- Acquisitions (largest AOI coverage first) ---');
  list.forEach(function (g) {
    print(g.key + ' | slices: ' + g.slices.length + ' | AOI coverage: ' + round(100 * g.cover, 1) + ' % | ' +
          new Date(g.t0).toISOString() + ' | look azimuth: ' + round(g.slices[0].azimuth, 1) + ' deg');
  });

  var selected;
  if (FORCE_ACQUISITION_KEY !== '') {
    selected = list.filter(function (g) { return g.key === FORCE_ACQUISITION_KEY; });
    if (selected.length === 0) { print('FORCE_ACQUISITION_KEY not found in the list above.'); return; }
  } else if (PROCESS_ALL_ACQUISITIONS) {
    selected = list;
  } else {
    selected = [list[0]];
  }
  if (selected[0].cover < 0.999) {
    print('WARNING: selected acquisition covers ' + round(100 * selected[0].cover, 1) +
          ' % of the AOI; uncovered pixels are written as NODATA.');
  }

  selected.forEach(function (g, i) { processAcquisition(g, i === 0); });
});


// =========================================================================
// 5. PROCESS ONE ACQUISITION
// =========================================================================
function processAcquisition(g, showOnMap) {
  var tag = g.key + '_' + new Date(g.t0).toISOString().slice(11, 19).replace(/:/g, '');

  // 10 m -> 100 m: mean of linear backscatter, incidence angle and look azimuth.
  var perSlice = g.slices.map(function (s) {
    var img   = ee.Image(s.id);
    var vvLin = img.select('VV').multiply(Math.LN10 / 10).exp().rename('vv_lin');
    var inc   = img.select('angle').rename('incidence_deg');
    var azi   = inc.multiply(0).add(s.azimuth).rename('radar_azimuth_deg');
    return vvLin.addBands(inc).addBands(azi)
      .reduceResolution({ reducer: ee.Reducer.mean(), maxPixels: 1024 })
      .reproject(PROJ_100M);
  });
  var mosaic = ee.ImageCollection(perSlice).mosaic();

  var vvDb = mosaic.select('vv_lin').log10().multiply(10).rename('vv_db');
  var product = vvDb.addBands(mosaic.select(['incidence_deg', 'radar_azimuth_deg']))
                    .updateMask(vvDb.mask()).toFloat();

  if (showOnMap) {
    Map.addLayer(product.select('vv_db'), { min: VIS_MIN, max: VIS_MAX }, 'VV 100 m (dB) ' + tag, true);
    Map.addLayer(patchOutlines.style({ color: 'red', fillColor: '00000000', width: 1 }), {}, 'Patch grid', true);
  }

  // Optional: one CSV row per pixel.
  if (EXPORT_PIXEL_TABLE) {
    var idx = ee.Image.pixelCoordinates(PROJ_100M).add(0.25).floor().rename(['grid_col', 'grid_row']);
    var ll  = ee.Image.pixelLonLat().rename(['lon', 'lat']);
    var ids   = ee.List(PATCH_IDS);
    var cLatL = ee.List(PATCHES.map(function (p) { return round(p.C.lat, 7); }));
    var cLonL = ee.List(PATCHES.map(function (p) { return round(p.C.lon, 7); }));

    var iso0     = new Date(g.t0).toISOString();
    var dateStr  = iso0.slice(0, 10).replace(/-/g, '/');
    var timeStr  = iso0.slice(11, 19);
    var sceneIds = g.slices.map(function (s) { return s.id.split('/').pop(); }).join(';');

    var sampled = product.addBands(idx).addBands(ll).sample({
      region: GRID_REGION, projection: PROJ_100M, dropNulls: false, geometries: false, tileScale: 4
    }).filter(ee.Filter.and(
      ee.Filter.gte('grid_row', 0), ee.Filter.lt('grid_row', N_ROWS * PATCH_PX),
      ee.Filter.gte('grid_col', 0), ee.Filter.lt('grid_col', N_COLS * PATCH_PX)));

    var pixelFC = sampled.map(function (f) {
      var r = ee.Number(f.get('grid_row')), c = ee.Number(f.get('grid_col'));
      var rowP = r.divide(PATCH_PX).floor().toInt(), colP = c.divide(PATCH_PX).floor().toInt();
      var pIdx = rowP.multiply(N_COLS).add(colP).toInt();
      return ee.Feature(null, {
        patch_id: ids.get(pIdx), patch_row: rowP, patch_col: colP,
        px_row: r.mod(PATCH_PX).toInt(), px_col: c.mod(PATCH_PX).toInt(),
        'lon(px_center)': f.get('lon'), 'lat(px_center)': f.get('lat'),
        vv_db: f.get('vv_db'), incidence_deg: f.get('incidence_deg'), radar_azimuth_deg: f.get('radar_azimuth_deg'),
        'date(YYYY/MM/DD)': dateStr, Time: timeStr, s1_scene_ids: sceneIds,
        'lat(patch_center)': cLatL.get(pIdx), 'lon(patch_center)': cLonL.get(pIdx)
      });
    });

    Export.table.toDrive({
      collection: pixelFC,
      description: 'PIXELS_' + tag,
      folder: DRIVE_FOLDER,
      fileNamePrefix: tag + '_patch_pixels_all',
      fileFormat: 'CSV',
      selectors: ['patch_id', 'patch_row', 'patch_col', 'px_row', 'px_col',
                  'lon(px_center)', 'lat(px_center)', 'vv_db', 'incidence_deg', 'radar_azimuth_deg',
                  'date(YYYY/MM/DD)', 'Time', 's1_scene_ids', 'lat(patch_center)', 'lon(patch_center)']
    });
  }

  // One GeoTIFF per patch (exactly 50 x 50 px on the 100 m grid).
  if (EXPORT_PATCH_IMAGES) {
    var exportImg = (PATCH_IMAGE_TYPE === 'DATA')
      ? product.select(['vv_db', 'incidence_deg', 'radar_azimuth_deg']).unmask(NODATA).toFloat()
      : product.select('vv_db').visualize({ min: VIS_MIN, max: VIS_MAX }).unmask(0);

    PATCHES.forEach(function (p) {
      Export.image.toDrive({
        image: exportImg,
        description: 'PATCH_' + p.id + '_' + tag,
        folder: DRIVE_FOLDER + '_images',
        fileNamePrefix: p.id + '_' + tag,
        region: utmRect(p.xmin, p.ymin, p.xmax, p.ymax, EDGE_SHRINK_M),
        crs: UTM_CRS,
        crsTransform: [PIXEL_M, 0, p.xmin, 0, -PIXEL_M, p.ymax],
        maxPixels: 1e8,
        fileFormat: 'GeoTIFF'
      });
    });
  }

  print('Export tasks created for ' + tag + '. Start them from the Tasks tab.');
}


/**
 * Starting many export tasks
 *   The Code Editor requires each task to be started manually. To start all
 *   of them, open the browser console (F12) with the Tasks tab visible and run:
 *
 *   function runTaskList(){var t=document.getElementsByClassName('task local type-EXPORT_IMAGE awaiting-user-config');
 *     for(var i=0;i<t.length;i++){t[i].getElementsByClassName('run-button')[0].click();}}
 *   function confirmAll(){var b=document.getElementsByClassName('goog-buttonset-default goog-buttonset-action');
 *     for(var i=0;i<b.length;i++){b[i].click();}}
 *   runTaskList();   // then, once the dialogs are open: confirmAll();
 *
 *   This relies on the Code Editor page structure and may stop working if the UI changes.
 */
