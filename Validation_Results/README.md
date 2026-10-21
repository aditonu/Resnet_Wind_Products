# Validation Results

Model: 3 ResNet models, trained on 21 Oct 2026 on 12 Sentinel-1 scenes (1,135 patches of 5 km, wind ≥ 3 m/s).
Reference wind: Copernicus Marine hourly L4 (0.125°).

## Wind direction

Each scene was tested with a model that **did not see that scene** in training (scene-wise cross-validation).

| Group | Scenes | Patches | Mean error | Median error | Within 15° |
|---|---|---|---|---|---|
| Training scenes (cross-validation) | 12 | 1,135 | 16.0° | 10.2° | 63 % |
| Test-only scenes (weak texture / low wind) | 10 | 678 | 43.0° | 41.1° | 18 % |

Error = smallest angle between the model axis and the Copernicus wind axis (180° ambiguity not counted).

## Wind speed (CMOD5.N)

| Group | Patches | Bias | RMSD |
|---|---|---|---|
| All 23 labelled scenes | 2,092 | +0.86 m/s | 1.85 m/s |
| Training scenes | 1,140 | +0.94 m/s | 1.69 m/s |
| Good-quality arrows only | 1,212 | +0.97 m/s | 1.71 m/s |

The speed model is not trained, so these numbers are independent on every scene.

## Notes

- Clear wind streaks (wind above ~5 m/s): direction error is usually 8–15°.
- Low wind or weak texture: errors of 30–60°. These arrows are marked "low" quality (white) on the maps.
- Chennai and Lima have a systematic offset of about 25° between the streaks and the Copernicus wind.
- 8 scenes from October 2026 have no Copernicus reference yet and are not in these tables.

## Files

| File | Content |
|---|---|
| `validation_summary.csv` | one row per scene: direction and speed errors |
| `cv_summary.json`, `cv_metrics_by_scene.csv` | cross-validation results |
| `validation_table_full.csv` | all statistics of the final model per scene |
| `texture_by_scene.csv` | wind-streak texture strength per scene |
| `maps/` | wind-field maps |
