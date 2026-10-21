"""
train_resnet_v2.py - scene-wise cross-validation of the wind-direction network.

  * Every scene is the test scene of exactly one fold, so all scenes receive an unseen-scene score.
  * Augmentation with the 8 transforms of the dihedral group D4 (rotations and flips).
  * Test-time augmentation: the predictions for the 8 transforms are mapped back to the original frame
    and averaged (axial circular mean).
  * N_SEEDS models per fold form an ensemble; the spread between seeds measures training noise.
  * Each patch gets a confidence in [0, 1] (agreement of the predictions).

Folds: scenes are sorted by their wind axis and assigned round-robin, so each fold covers different axes.
Fold k: test = scenes of fold k, early-stopping validation = scenes of fold k+1, training = all others.

Options
  --exclude_folders a,b   scenes not used at all
  --eval_only a,b         scenes never used for training or validation, only tested with the ensemble of all
                          fold models (e.g. scenes with weak wind-streak texture)
  --lowpass_px 1.5        Gaussian low-pass on the input (removes swell and speckle), then per-patch z-score

  python train_resnet_v2.py --npz .../stageA/dataset_A.npz --out_dir .../cv_results --exclude_folders durban
"""
from __future__ import annotations

import argparse, json, os, time
import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from torch.utils.data import DataLoader, TensorDataset

import train_resnet as T

# ----- 8 transforms (dihedral group).  label: theta' = s*theta + a  (bearing clockwise from North, north-up image) -----
#   rot90 CCW: -90 | flipLR: -theta | flipUD: 180-theta | transpose (swap row/col): 270-theta | anti-transpose: 90-theta
D4 = [("orig", lambda x: x, 1, 0),
      ("rot90", lambda x: np.rot90(x, 1, axes=(-2, -1)), 1, -90),
      ("rot180", lambda x: np.rot90(x, 2, axes=(-2, -1)), 1, -180),
      ("rot270", lambda x: np.rot90(x, 3, axes=(-2, -1)), 1, -270),
      ("flipLR", lambda x: np.flip(x, -1), -1, 0),
      ("flipUD", lambda x: np.flip(x, -2), -1, 180),
      ("transpose", lambda x: np.swapaxes(x, -1, -2), -1, 270),
      ("antitranspose", lambda x: np.flip(np.flip(np.swapaxes(x, -1, -2), -1), -2), -1, 90)]


def preprocess(X, lowpass_px):
    """(N,50,50) z-score patches -> Gaussian low-pass (sigma in px) -> per-patch z-score again; lowpass_px <= 0: unchanged.
    The same function must be used for training and inference."""
    X = np.asarray(X, np.float32)
    if lowpass_px <= 0:
        return X
    Xf = ndimage.gaussian_filter(X, sigma=(0, lowpass_px, lowpass_px), mode="reflect")
    Xf -= Xf.mean(axis=(1, 2), keepdims=True)
    Xf /= Xf.std(axis=(1, 2), keepdims=True) + 1e-6
    return Xf.astype(np.float32)


def augment8(X, th):
    """X (N,50,50), th (N,) -> X8 (8N,1,50,50), th8 (8N,)"""
    Xa, ta = [], []
    for _, f, s, a in D4:
        Xa.append(np.ascontiguousarray(f(X))); ta.append((s * np.asarray(th, float) + a) % 360.0)
    return np.concatenate(Xa)[:, None].astype(np.float32), np.concatenate(ta).astype(np.float32)


def axial_mean_R(angles_deg, axis=-1):
    """Circular mean of axial data (period 180) along `axis` -> (mean in [0,180), resultant length R in [0,1])."""
    t = np.radians(2 * np.asarray(angles_deg, float))
    c, s = np.cos(t).mean(axis), np.sin(t).mean(axis)
    return (0.5 * np.degrees(np.arctan2(s, c))) % 180.0, np.hypot(c, s)


@torch.no_grad()
def predict_tta(model, X, bs=256):
    """X (N,50,50) -> (N,8) predictions, each already rotated BACK to the original image's frame (deg)."""
    out = []
    for _, f, s, a in D4:
        p = T.predict(model, np.ascontiguousarray(f(X))[:, None].astype(np.float32), bs)    # prediction for transformed image
        out.append((s * (p - a)) % 360.0)                                                    # theta = s*(theta' - a)
    return np.stack(out, 1)


# ----- one model -----
def train_one(Xtr, thtr, Xva, thva, a, seed, tag=""):
    torch.manual_seed(seed); np.random.seed(seed)
    model = T.WindDirNet(dropout=a.dropout).to(T.DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=a.lr_patience) if a.lr_patience > 0 else None
    g = torch.Generator(); g.manual_seed(seed)
    dl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(np.radians(thtr))),
                    batch_size=a.batch_size, shuffle=True, drop_last=True, generator=g)
    best, best_state, wait, best_ep = np.inf, None, 0, 0
    for ep in range(1, a.max_epochs + 1):
        model.train()
        for xb, tb in dl:
            xb, tb = xb.to(T.DEVICE), tb.to(T.DEVICE)
            loss = T.loss_fn(model(xb), tb); opt.zero_grad(); loss.backward(); opt.step()
        vl = T.mean_loss(model, Xva, thva)
        if sched: sched.step(vl)
        if vl < best - 1e-6:
            best, best_ep, wait = vl, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= a.patience: break
    model.load_state_dict(best_state)
    print(f"    {tag} seed {seed}: best epoch {best_ep}/{ep}, val loss {best:.4f}", flush=True)
    return model, best


# ----- cross-validation -----
def make_folds(D, mask_ok, k):
    scenes = sorted(str(s) for s in set(D["folder"][mask_ok]))
    ax = {s: T.axial_mean(D["th"][(D["folder"] == s) & mask_ok]) for s in scenes}
    order = sorted(scenes, key=lambda s: ax[s])
    fold_of = {s: i % k for i, s in enumerate(order)}
    return fold_of, ax


def run(a):
    os.makedirs(a.out_dir, exist_ok=True)
    split = lambda s: [x.strip() for x in s.split(",") if x.strip()]
    D = T.load_npz(a.npz, a.min_speed)
    low = {f.lower(): f for f in set(D["folder"])}
    def resolve(names):
        bad = [e for e in split(names) if e.lower() not in low]
        if bad: raise ValueError(f"Folders not in the dataset: {bad}. Available: {sorted(map(str, low.values()))}")
        return [str(low[e.lower()]) for e in split(names)]
    excl, eval_only = resolve(a.exclude_folders), resolve(a.eval_only)
    if a.lowpass_px > 0:                       # remove small scales (swell, speckle); keep km-scale texture
        D["X"] = preprocess(D["X"], a.lowpass_px)
        print(f"  Gaussian low-pass sigma = {a.lowpass_px} px (~{a.lowpass_px * 100:.0f} m), then z-score")
    ok = ~np.isin(D["folder"], excl)                       # scenes used at all
    ok_tr = ok & ~np.isin(D["folder"], eval_only)          # scenes available for training / validation
    m_eo = ok & np.isin(D["folder"], eval_only)            # test only (never trained on)
    if a.folds < 3: raise ValueError("--folds must be >= 3")
    fold_of, ax = make_folds(D, ok_tr, a.folds)
    print(f"\n{len(fold_of)} CV scenes, {a.folds} folds, {a.seeds} seeds/fold | excluded: {excl} | eval-only (never trained on): {eval_only}")
    for f in range(a.folds):
        print(f"  fold {f}: " + ", ".join(f"{s}({ax[s]:.0f})" for s in fold_of if fold_of[s] == f))
    t0 = time.time()
    rows, eo_preds = [], {}
    for f in range(a.folds):
        te_sc = [str(s) for s in fold_of if fold_of[s] == f]
        va_sc = [str(s) for s in fold_of if fold_of[s] == (f + 1) % a.folds]
        m_te, m_va = np.isin(D["folder"], te_sc) & ok_tr, np.isin(D["folder"], va_sc) & ok_tr
        m_tr = ok_tr & ~m_te & ~m_va
        Xtr, thtr = augment8(D["X"][m_tr], D["th"][m_tr]); Xva, thva = augment8(D["X"][m_va], D["th"][m_va])
        print(f"\nFOLD {f}: test {te_sc} | val {va_sc} | train {int(m_tr.sum())} patches ({len(set(D['folder'][m_tr]))} scenes) "
              f"-> {len(Xtr)} samples  [{(time.time() - t0) / 60:.1f} min]", flush=True)
        preds = []
        for sd in range(a.seeds):
            model, _ = train_one(Xtr, thtr, Xva, thva, a, a.seed + 1000 * f + sd, tag=f"fold {f}")
            preds.append(predict_tta(model, D["X"][m_te]))                                  # (n,8)
            if m_eo.any():
                eo_preds.setdefault(sd, []).append(predict_tta(model, D["X"][m_eo]))
            if a.save_models:
                torch.save(model.state_dict(), os.path.join(a.out_dir, f"model_fold{f}_seed{sd}.pt"))
        P = np.stack(preds, 0)                                                              # (seeds,n,8)
        ens, conf = axial_mean_R(P.transpose(1, 0, 2).reshape(P.shape[1], -1))             # all seeds x 8 tta
        single = np.stack([axial_mean_R(P[s])[0] for s in range(a.seeds)], 0)              # (seeds,n) TTA only
        notta = P[:, :, 0] % 180                                                            # no TTA, no ensemble
        idx = np.where(m_te)[0]
        for j, i in enumerate(idx):
            rows.append(dict(uid=D["uid"][i], scene=D["folder"][i], fold=f, wind_speed=float(D["speed"][i]),
                             copernicus_dir=float(D["th"][i]), pred_axis_ens=float(ens[j]), conf=float(conf[j]),
                             **{f"pred_tta_seed{s}": float(single[s, j]) for s in range(a.seeds)},
                             **{f"pred_plain_seed{s}": float(notta[s, j]) for s in range(a.seeds)}))
        pd.DataFrame(rows).to_csv(os.path.join(a.out_dir, "cv_predictions.csv"), index=False)   # saves progressively

    if m_eo.any():      # eval-only scenes: ensemble of all fold models (none was trained on these scenes)
        allP = np.concatenate([np.concatenate(v, 1) for v in eo_preds.values()], 1)        # (n, folds*seeds*8)
        ens, conf = axial_mean_R(allP)
        single = {s: axial_mean_R(np.concatenate(v, 1))[0] for s, v in eo_preds.items()}
        plain = {s: v[0][:, 0] % 180 for s, v in eo_preds.items()}
        for j, i in enumerate(np.where(m_eo)[0]):
            rows.append(dict(uid=D["uid"][i], scene=D["folder"][i], fold=-1, wind_speed=float(D["speed"][i]),
                             copernicus_dir=float(D["th"][i]), pred_axis_ens=float(ens[j]), conf=float(conf[j]),
                             **{f"pred_tta_seed{s}": float(single[s][j]) for s in single},
                             **{f"pred_plain_seed{s}": float(plain[s][j]) for s in plain}))
        pd.DataFrame(rows).to_csv(os.path.join(a.out_dir, "cv_predictions.csv"), index=False)

    R = pd.DataFrame(rows)
    ae = lambda p, t: T.alias_err(p, t)
    out = []
    for sc, g in R.groupby("scene"):
        t = g.copernicus_dir.values; e = ae(g.pred_axis_ens.values, t)
        hi = g.conf >= g.conf.median()
        row = dict(scene=sc, group="eval_only" if int(g.fold.iloc[0]) < 0 else "cv", fold=int(g.fold.iloc[0]), n=len(g), true_axis=T.axial_mean(t),
                   pred_axis_scene=T.axial_mean(g.pred_axis_ens.values))
        row["scene_mean_err"] = float(ae(row["pred_axis_scene"], row["true_axis"]))
        row.update(MAE_ens=float(np.abs(e).mean()), median_abs=float(np.median(np.abs(e))), within15_pct=float(100 * (np.abs(e) <= 15).mean()),
                   MAE_confident_half=float(np.abs(e[hi.values]).mean()), MAE_uncertain_half=float(np.abs(e[~hi.values]).mean()),
                   MAE_plain_single=float(np.mean([np.abs(ae(g[f"pred_plain_seed{s}"].values, t)).mean() for s in range(a.seeds)])),
                   MAE_tta_single=float(np.mean([np.abs(ae(g[f"pred_tta_seed{s}"].values, t)).mean() for s in range(a.seeds)])),
                   seed_spread=float(np.std([np.abs(ae(g[f"pred_tta_seed{s}"].values, t)).mean() for s in range(a.seeds)])) if a.seeds > 1 else np.nan)
        out.append(row)
    S = pd.DataFrame(out).sort_values(["group", "true_axis"])
    S.to_csv(os.path.join(a.out_dir, "cv_metrics_by_scene.csv"), index=False)
    pd.set_option("display.width", 220)
    print("\n=== CROSS-VALIDATION (every scene unseen) ===")
    print(S[["group", "scene", "n", "true_axis", "pred_axis_scene", "scene_mean_err", "MAE_ens", "median_abs", "within15_pct",
             "MAE_plain_single", "MAE_tta_single", "seed_spread", "MAE_confident_half", "MAE_uncertain_half"]].round(1).to_string(index=False))
    def summarise(Rg, Sg):
        e_all = ae(Rg.pred_axis_ens.values, Rg.copernicus_dir.values)
        return dict(scenes=int(len(Sg)), patches=int(len(Rg)), MAE_pooled=float(np.abs(e_all).mean()),
                    MAE_mean_over_scenes=float(Sg.MAE_ens.mean()), median_abs=float(np.median(np.abs(e_all))),
                    within15_pct=float(100 * (np.abs(e_all) <= 15).mean()), MAE_plain_single=float(Sg.MAE_plain_single.mean()),
                    MAE_tta_single=float(Sg.MAE_tta_single.mean()), scene_mean_abs_err=float(Sg.scene_mean_err.abs().mean()))
    summary = {"ALL": summarise(R, S)}
    for grp in ("cv", "eval_only"):
        if (S.group == grp).any():
            summary[grp] = summarise(R[(R.fold >= 0) if grp == "cv" else (R.fold < 0)], S[S.group == grp])
    for k, v in summary.items():
        print(f"\nSUMMARY [{k}]:", json.dumps({kk: round(vv, 1) for kk, vv in v.items()}))
    print("(random guess MAE = 45)")
    json.dump(dict(summary=summary, args=vars(a), folds={s: int(v) for s, v in fold_of.items()}),
              open(os.path.join(a.out_dir, "cv_summary.json"), "w"), indent=2)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax_ = plt.subplots(1, 2, figsize=(11, 4.5))
        sc_ = ax_[0].scatter(R.copernicus_dir % 180, R.pred_axis_ens, c=R.conf, s=8, cmap="viridis", vmin=0, vmax=1)
        ax_[0].plot([0, 180], [0, 180], "k--", lw=.8); plt.colorbar(sc_, ax=ax_[0], label="confidence")
        ax_[0].set_xlabel("Copernicus axis (mod 180)"); ax_[0].set_ylabel("ResNet axis (CV, ensemble+TTA)")
        ax_[1].errorbar(S.true_axis, S.pred_axis_scene, fmt="o"); ax_[1].plot([0, 180], [0, 180], "k--", lw=.8)
        for _, r in S.iterrows(): ax_[1].annotate(r.scene, (r.true_axis, r.pred_axis_scene), fontsize=6)
        ax_[1].set_xlabel("scene true axis"); ax_[1].set_ylabel("scene mean predicted axis")
        plt.tight_layout(); plt.savefig(os.path.join(a.out_dir, "results_cv.png"), dpi=150)
    except Exception as e:
        print("plot skipped:", e)
    return S, summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True); ap.add_argument("--out_dir", required=True)
    ap.add_argument("--folds", type=int, default=7); ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--exclude_folders", default="", help="scenes not used at all")
    ap.add_argument("--eval_only", default="", help="scenes used only for testing (ensemble of all fold models)")
    ap.add_argument("--lowpass_px", type=float, default=0.0, help="Gaussian low-pass sigma in pixels; 0 = off")
    ap.add_argument("--min_speed", type=float, default=3.0)
    ap.add_argument("--batch_size", type=int, default=64); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4); ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr_patience", type=int, default=5, help="halve the learning rate after this many epochs without improvement (0 = off)")
    ap.add_argument("--max_epochs", type=int, default=120); ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--save_models", action="store_true")
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
