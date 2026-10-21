# Resnet_Wind_Products

Ocean wind field from Sentinel-1 SAR images for the coasts of Tamil Nadu and Gujarat (wind farm planning).

You draw an area on the map and choose dates. The API gives the wind direction and speed every 5 km.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aditonu/Resnet_Wind_Products/blob/main/notebooks/02_wind_field_api.ipynb)

**New user?** Read the [Setup Guide](SETUP_GUIDE.md).

## Models used

| Part | Model | Details |
|---|---|---|
| Wind direction | **ResNet (M64RN4)** | 4 residual blocks × 64 filters (Zanchetta & Zecchetto, 2021). Ensemble of 3 models, trained on 12 Sentinel-1 scenes (21 Oct 2026). Input: one 5 km patch (50 × 50 pixels of 100 m). |
| Wind speed | **CMOD5.N** | C-band VV geophysical model function (Hersbach, 2010), from the `xsarsea` library. Not trained. |
| Correct direction (180°) and validation | **Copernicus Marine L4 wind** | hourly, 0.125° (`cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H`, near-real-time product for recent dates) |
| Satellite data | **Sentinel-1 IW GRDH, VV** | from Google Earth Engine (`COPERNICUS/S1_GRD`) |

## How to use the API

1. Click **Open in Colab** above.
2. Run **Setup**. Enter your Earth Engine project id, then sign in to Google and Copernicus Marine.
3. **Step 1:** draw a box on the sea (max 50 km × 50 km).
4. **Step 2:** choose the start and end date and run the cell. Wind maps appear below.
5. **Step 3:** download the results (zip).

You need two free accounts: [Google Earth Engine](https://earthengine.google.com/) and
[Copernicus Marine](https://marine.copernicus.eu/).

## Output

For every Sentinel-1 image in the date range:

| File | What it has |
|---|---|
| `wind_vectors.geojson` | wind direction, speed, u, v and quality for each 5 km point |
| `wind_field_*.csv` | the same as a table |
| `map_*.png` | wind arrows on a map (black = good, white = low quality) |
| `summary_*.json` | image details and comparison with Copernicus wind |

Wind direction is the direction the wind comes **from** (0° = north, 90° = east).

## How it works

1. Sentinel-1 image is downloaded from Google Earth Engine with three bands (see below) and averaged to 100 m.
2. The area is cut into 5 km × 5 km patches.
3. A ResNet model finds the wind direction from the wind streaks in each patch.
4. Copernicus Marine wind decides which of the two opposite directions is correct.
5. Wind speed is calculated with the CMOD5.N model.

## Data from Google Earth Engine

From Sentinel-1 (`COPERNICUS/S1_GRD`: IW mode, VV polarisation, GRDH 10 m) we take **three bands for every 5 km patch**:

| Band | Source in Earth Engine | Used for |
|---|---|---|
| **VV backscatter** σ⁰ (dB) | `VV` band, 10 m → 100 m (mean in linear power, then back to dB) | **direction**: the ResNet reads the wind streaks in this image; **speed**: mean VV of the patch goes into CMOD5.N |
| **Incidence angle** (°) | `angle` band | **speed**: the same wind gives different backscatter at different radar angles |
| **Radar look azimuth** (°) | `platformHeading + 90°` (Sentinel-1 looks to the right) | **speed**: CMOD5.N needs the angle between wind and radar, φ = wind direction − look azimuth |

Direction needs only VV; speed needs all three.

Each patch is saved as a 3-band GeoTIFF of 50 × 50 pixels (100 m), named like `C001_S1A_20240305_DSC_RO092_003133.tif` (patch, satellite, date, pass, orbit, time). With `gee/s1_patch_export.js` we exported **31 Sentinel-1 scenes** (10 × 10 patches each, 50 km × 50 km): **3,001 patch files**, of which **2,795** are sea patches after removing land and no-data, and **2,092** have a Copernicus Marine wind label for training and validation.

In the API the same three bands are downloaded directly with `ee.data.computePixels`, with exactly the same processing.

## Results

Tested on scenes the model did not see in training (details in `Validation_Results/`):

| | Result |
|---|---|
| Wind direction, mean error | 16.0° (median 10.2°) on 12 scenes |
| Wind speed, bias / RMSD | +0.86 m/s / 1.85 m/s on 23 scenes |

## Files

| Folder / file | What it is |
|---|---|
| `notebooks/02_wind_field_api.ipynb` | the API (draw box + dates → wind field) |
| `notebooks/01_training_and_validation.ipynb` | dataset, training and validation |
| `gee/s1_patch_export.js` | Earth Engine script used to download training images |
| `src/` | Python code |
| `SETUP_GUIDE.md` | step-by-step setup for new users |
| `models/final_model/` | trained model (weights of the 3 ResNet M64RN4 models) |
| `Validation_Results/` | validation table and maps |
| `requirements.txt` | Python libraries |

## Reference

Zanchetta, A., Zecchetto, S. (2021). Wind direction retrieval from Sentinel-1 SAR images using ResNet.
*Remote Sensing of Environment*, 253, 112178. https://www.sciencedirect.com/science/article/pii/S0034425720305514
