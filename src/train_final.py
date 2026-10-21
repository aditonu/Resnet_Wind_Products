"""
train_final.py - final ResNet ensemble trained on all good-texture scenes (used by the inference and the API).

Settings (selected by cross-validation; 10 unseen scenes: MAE ~16 deg, median ~10.5 deg):
  * wind speed >= 3 m/s
  * training scenes: all except the excluded ones (weak-texture scenes and Durban by default)
  * Gaussian low-pass sigma 1.5 px, then per-patch z-score
  * D4 augmentation, Adam with weight decay, dropout 0.3, learning-rate decay, early stopping (patience 15)

Ensemble of N_MODELS networks; each uses two different validation scenes for early stopping and is trained on
the rest, so together the ensemble learns from all scenes.

Outputs (out_dir): model_0.pt ..., config.json (settings and per-model information), train_log.txt,
                   check_predictions.csv (training scenes are in-sample, not a validation; excluded scenes are unseen)

  python train_final.py --npz .../stageA/dataset_A.npz --out_dir .../final_model
"""
from __future__ import annotations

import argparse, json, os, time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

import train_resnet as T
import train_resnet_v2 as V
from wind_model import load_ensemble, predict_axis


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True); ap.add_argument("--out_dir", required=True)
    ap.add_argument("--exclude_folders", default="durban,oman,san_diego,bahia,nothwest_medagascar")
    ap.add_argument("--n_models", type=int, default=3)
    ap.add_argument("--lowpass_px", type=float, default=1.5); ap.add_argument("--min_speed", type=float, default=3.0)
    ap.add_argument("--batch_size", type=int, default=64); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4); ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr_patience", type=int, default=5); ap.add_argument("--max_epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=15); ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    log_lines = []
    def log(msg):
        print(msg, flush=True); log_lines.append(msg)

    D = T.load_npz(a.npz, a.min_speed)
    low = {str(f).lower(): str(f) for f in set(D["folder"])}
    excl = [x.strip() for x in a.exclude_folders.split(",") if x.strip()]
    bad = [e for e in excl if e.lower() not in low]
    if bad: raise ValueError(f"Folders not in the dataset: {bad}. Available: {sorted(low.values())}")
    excl = [low[e.lower()] for e in excl]
    folder = np.array([str(f) for f in D["folder"]])
    ok = ~np.isin(folder, excl)
    X = V.preprocess(D["X"], a.lowpass_px)
    scenes = sorted(set(folder[ok]), key=lambda s: T.axial_mean(D["th"][(folder == s) & ok]))
    log(f"FINAL training | {len(scenes)} scenes: {scenes}\n  excluded: {excl} | low-pass {a.lowpass_px} px | device {T.DEVICE}")
    if len(scenes) < 2 * a.n_models + 2:
        log("  (few scenes: some validation scenes are repeated across models)")

    # validation pairs i and i + N/2 of the axis-sorted list -> the two validation scenes have clearly different axes
    h = len(scenes) // 2
    val_sets = [[scenes[i % len(scenes)], scenes[(i + h) % len(scenes)]] for i in range(a.n_models)]
    files, info, t0 = [], [], time.time()
    for i, vs in enumerate(val_sets):
        m_va = np.isin(folder, vs) & ok
        m_tr = ok & ~m_va
        Xtr, thtr = V.augment8(X[m_tr], D["th"][m_tr]); Xva, thva = V.augment8(X[m_va], D["th"][m_va])
        log(f"\nMODEL {i}: val {vs} | train {int(m_tr.sum())} patches -> {len(Xtr)} samples [{(time.time() - t0) / 60:.1f} min]")
        model, best = V.train_one(Xtr, thtr, Xva, thva, a, a.seed + 100 * i, tag=f"model {i}")
        f = f"model_{i}.pt"
        torch.save(model.state_dict(), os.path.join(a.out_dir, f))
        files.append(f); info.append(dict(file=f, val_scenes=vs, val_loss=float(best), seed=a.seed + 100 * i))
        log(f"  saved {f} (val loss {best:.4f})")

    cfg = dict(model_files=files, models=info, lowpass_px=a.lowpass_px, min_speed=a.min_speed, dropout=a.dropout,
               train_scenes=scenes, train_scene_ids=sorted({str(x) for x in D["scene"][ok]}), excluded_scenes=excl, patch_px=T.PATCH_PX, pixel_m=100,
               input="per-patch z-score of VV dB (Step A), then preprocess(lowpass_px)",
               output="wind AXIS deg clockwise from North [0,180); true direction = axis or axis+180",
               angle_convention="meteorological FROM, north-up images", tta="8 dihedral transforms",
               cv_reference="cross-validation: 10 unseen scenes MAE 15.9 deg, median 10.5 deg; weak-texture scenes ~45 deg (random)",
               created_utc=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), args=vars(a))
    json.dump(cfg, open(os.path.join(a.out_dir, "config.json"), "w"), indent=2)

    # ---- sanity check: reload the saved files and predict (same path as the API) ----
    models, cfg2 = load_ensemble(a.out_dir)
    ax, conf = predict_axis(models, cfg2, D["X"])
    rows = []
    for sc in sorted(set(folder)):
        m = folder == sc
        e = T.alias_err(ax[m], D["th"][m] % 180)
        kind = "train (in-sample, NOT validation)" if sc in scenes else "weak/excluded (unseen)"
        rows.append(dict(scene=sc, kind=kind, n=int(m.sum()), true_axis=T.axial_mean(D["th"][m]),
                         pred_axis=T.axial_mean(ax[m]), MAE=float(np.abs(e).mean()), mean_conf=float(conf[m].mean())))
    C = pd.DataFrame(rows).sort_values(["kind", "true_axis"])
    pd.DataFrame(dict(uid=D["uid"], scene=folder, copernicus_dir=D["th"], pred_axis=ax, conf=conf)).to_csv(
        os.path.join(a.out_dir, "check_predictions.csv"), index=False)
    log("\nSANITY CHECK (reloaded models):\n" + C.round(1).to_string(index=False))
    log("\nNOTE: the MAE of training scenes is in-sample; the independent score is the cross-validation result.")
    log(f"\nDONE in {(time.time() - t0) / 60:.1f} min -> {a.out_dir}")
    open(os.path.join(a.out_dir, "train_log.txt"), "w").write("\n".join(log_lines))
    return C


if __name__ == "__main__":
    main()
