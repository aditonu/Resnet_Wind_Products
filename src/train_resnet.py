"""
train_resnet.py - ResNet M64RN4 wind-direction network, loss, metrics, and a single train/val/test run.
Reference: Zanchetta & Zecchetto (2021), Remote Sensing of Environment 253, 112178 (Sections 3.1-3.3, 4).

The network, loss and metric functions defined here are used by train_resnet_v2.py (cross-validation),
train_final.py (final ensemble) and wind_model.py (inference).

Single run (scene-wise split):
  1. Loads dataset_A.npz; patches with wind speed below --min_speed (default 3 m/s) are removed.
  2. Splits by scene, not by patch (neighbouring patches of a scene share nearly the same wind, so a
     random patch split would overestimate the skill): test scenes, validation scenes (early stopping),
     training scenes (all others).
  3. Augmentation of training and validation data: rotations by 90/180/270 deg and flips, with the
     target angle transformed accordingly.
  4. M64RN4 (4 residual blocks x 64 channels, FC 512-128-32), loss 1 - cos^2(theta - theta_target),
     Adam (lr 1e-3), early stopping on the validation loss (patience 15).
  5. Scene-wise metrics on validation and test scenes, compared with a constant-direction baseline.

Directions are meteorological ("from"), degrees clockwise from north; images are north-up.

  python train_resnet.py --npz .../stageA/dataset_A.npz --out_dir .../single_run
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PATCH_PX = 50
AUG_NAMES = ["orig", "rot90ccw", "rot180", "rot270ccw", "flipLR", "flipUD"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULTS = dict(
    test_folders="netherland,Chennai,Gujarat",
    val_folders="vietnam,chile",
    exclude_folders="",            # scenes not used at all, e.g. "durban"
    min_speed=3.0,                 # m/s; 0 = no filter
    batch_size=64,                 # paper: 64
    lr=1e-3,                       # paper
    max_epochs=200,
    patience=15,                   # paper
    seed=42,
)


# =====================================================================================================
# 1. Data
# =====================================================================================================
def load_npz(path, min_speed):
    d = np.load(path, allow_pickle=False)
    th = d["theta_from"].astype(np.float64)
    spd = d["wind_speed"].astype(np.float64)
    ok = np.isfinite(th) & np.isfinite(spd)
    n_all = int(ok.sum())
    if min_speed > 0:
        ok &= spd >= min_speed
    print(f"  npz: {len(th)} patches | labelled {n_all} | speed >= {min_speed} m/s: {int(ok.sum())}")
    out = dict(X=d["X"][ok].astype(np.float32), th=th[ok].astype(np.float32), speed=spd[ok].astype(np.float32),
               folder=d["folder"][ok], uid=d["uid"][ok], scene=d["scene_id"][ok])
    return out


def split_by_scene(D, test_folders, val_folders, exclude_folders):
    avail = sorted(set(D["folder"]))
    low = {f.lower(): f for f in avail}

    def resolve(names):
        res = []
        for n in names:
            if n == "":
                continue
            if n.lower() not in low:
                raise ValueError(f"Folder '{n}' is not in the dataset. Available: {avail}")
            res.append(low[n.lower()])
        return res

    te, va, ex = resolve(test_folders), resolve(val_folders), resolve(exclude_folders)
    if set(te) & set(va) or set(te) & set(ex) or set(va) & set(ex):
        raise ValueError("test / val / exclude folders overlap")
    f = D["folder"]
    idx = dict(test=np.isin(f, te), val=np.isin(f, va))
    idx["train"] = ~(idx["test"] | idx["val"] | np.isin(f, ex))
    if idx["train"].sum() == 0 or idx["val"].sum() == 0:
        raise ValueError("train or validation set is empty; check the folder lists")
    return idx, dict(test=te, val=va, exclude=ex)


def subset(D, m):
    return {k: v[m] for k, v in D.items()}


# =====================================================================================================
# 2. Augmentation (image + target angle). north-up, bearing clockwise from North:
#    rot90 CCW k times: theta - 90k | flipLR: -theta | flipUD: 180 - theta
# =====================================================================================================
def augment(X, th):
    Xa, tha, tag = [], [], []
    for i in range(len(X)):
        im, t = X[i], float(th[i])
        variants = [(im, t)]
        variants += [(np.rot90(im, k).copy(), t - 90 * k) for k in (1, 2, 3)]
        variants += [(np.flip(im, 1).copy(), -t), (np.flip(im, 0).copy(), 180 - t)]
        for name, (a, b) in zip(AUG_NAMES, variants):
            Xa.append(a); tha.append(b % 360.0); tag.append(name)
    return np.stack(Xa)[:, None], np.array(tha, np.float32), np.array(tag)


# =====================================================================================================
# 3. Model M64RN4
# =====================================================================================================
class RNB(nn.Module):
    def __init__(self, c_in, d):
        super().__init__()
        self.c1, self.b1 = nn.Conv2d(c_in, d, 3, padding=1), nn.BatchNorm2d(d)
        self.c2, self.b2 = nn.Conv2d(d, d, 3, padding=1), nn.BatchNorm2d(d)
        self.short = nn.Identity() if c_in == d else nn.Conv2d(c_in, d, 1)
        self.pool, self.b3 = nn.MaxPool2d(2), nn.BatchNorm2d(d)

    def forward(self, x):
        y = self.b1(torch.relu(self.c1(x)))
        y = self.b2(torch.relu(self.c2(y)))
        y = torch.relu(y + self.short(x))
        return self.b3(self.pool(y))


class WindDirNet(nn.Module):
    def __init__(self, size=PATCH_PX, d=64, n_blocks=4, dropout=0.0):
        super().__init__()
        dr = (lambda: nn.Dropout(dropout)) if dropout > 0 else (lambda: nn.Identity())
        blocks, c = [], 1
        for _ in range(n_blocks):
            blocks.append(RNB(c, d)); c = d
        self.features = nn.Sequential(*blocks)
        with torch.no_grad():
            n_flat = self.features.eval()(torch.zeros(1, 1, size, size)).numel()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flat, 512), nn.ReLU(), nn.BatchNorm1d(512), dr(),
            nn.Linear(512, 128), nn.ReLU(), nn.BatchNorm1d(128), dr(),
            nn.Linear(128, 32), nn.ReLU(), nn.BatchNorm1d(32),
            nn.Linear(32, 1))

    def forward(self, x):
        return self.fc(self.features(x)).squeeze(1)


def loss_fn(pred, true):
    return (1.0 - torch.cos(pred - true) ** 2).mean()


# =====================================================================================================
# 4. Metrics
# =====================================================================================================
def wrap180(a):
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def axial_mean(a_deg):
    """Mean direction of axial data (period 180), result in [0, 180)."""
    t = np.radians(2 * np.asarray(a_deg, float))
    return float((0.5 * np.degrees(np.arctan2(np.sin(t).mean(), np.cos(t).mean()))) % 180)


def alias_err(pred, true):
    """Signed error on the 180-periodic axis, in [-90, 90); independent of the 180 deg ambiguity."""
    return (np.asarray(pred) - np.asarray(true) + 90.0) % 180.0 - 90.0


def dealias(pred, ref):
    """Of the two senses (pred, pred + 180) the one closer to ref."""
    pred, ref = np.asarray(pred) % 360, np.asarray(ref) % 360
    alt = (pred + 180) % 360
    return np.where(np.abs(wrap180(pred - ref)) <= np.abs(wrap180(alt - ref)), pred, alt)


def metrics(pred, true):
    ae = alias_err(pred, true)
    de = wrap180(dealias(pred, true) - true)
    beta = float(de.mean())
    rmsd = float(np.sqrt((de ** 2).mean()))
    return dict(n=int(len(pred)),
                aliased_MAE=float(np.abs(ae).mean()), aliased_RMS=float(np.sqrt((ae ** 2).mean())),
                aliased_within10_pct=float(100 * (np.abs(ae) <= 10).mean()),
                aliased_over20_pct=float(100 * (np.abs(ae) > 20).mean()),
                dealiased_bias=beta, dealiased_cRMSd=float(np.sqrt(max(rmsd ** 2 - beta ** 2, 0.0))))


def scene_table(split, S, pred, base_axis):
    """Per-scene patch-level metrics and the error of the scene-mean axis."""
    rows = []
    for fol in sorted(set(S["folder"])):
        m = S["folder"] == fol
        mt = metrics(pred[m], S["th"][m])
        true_axis, pred_axis = axial_mean(S["th"][m]), axial_mean(pred[m])
        rows.append(dict(split=split, scene=fol, true_axis=true_axis, pred_axis_mean=pred_axis,
                         scene_mean_err=float(alias_err(pred_axis, true_axis)),
                         baseline_MAE=float(np.abs(alias_err(base_axis, S["th"][m])).mean()), **mt))
    m_all = metrics(pred, S["th"])
    rows.append(dict(split=split, scene="ALL", true_axis=np.nan, pred_axis_mean=np.nan, scene_mean_err=np.nan,
                     baseline_MAE=float(np.abs(alias_err(base_axis, S["th"])).mean()), **m_all))
    return pd.DataFrame(rows)


@torch.no_grad()
def predict(model, X, bs=256):
    """X (N,1,50,50) -> angle in degrees (0..360); only the axis (mod 180) is meaningful."""
    model.eval()
    out = [model(torch.from_numpy(X[i:i + bs]).to(DEVICE)).cpu().numpy() for i in range(0, len(X), bs)]
    return np.degrees(np.concatenate(out)) % 360.0


@torch.no_grad()
def mean_loss(model, X, th_deg, bs=256):
    model.eval()
    tot = 0.0
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(DEVICE)
        tb = torch.from_numpy(np.radians(th_deg[i:i + bs])).to(DEVICE)
        tot += loss_fn(model(xb), tb).item() * len(xb)
    return tot / len(X)


# =====================================================================================================
# 5. Main
# =====================================================================================================
def run(a):
    os.makedirs(a.out_dir, exist_ok=True)
    np.random.seed(a.seed); torch.manual_seed(a.seed)
    split_list = lambda s: [x.strip() for x in s.split(",") if x.strip()]

    print("Loading ...")
    D = load_npz(a.npz, a.min_speed)
    idx, names = split_by_scene(D, split_list(a.test_folders), split_list(a.val_folders), split_list(a.exclude_folders))
    TR, VA, TE = (subset(D, idx[k]) for k in ("train", "val", "test"))

    print("\nSplit (scene-wise):")
    for nm, S in (("TRAIN", TR), ("VAL", VA), ("TEST", TE)):
        print(f"  {nm:5s} {len(S['X']):5d} patches | scenes: " +
              ", ".join(f"{f}({int((S['folder'] == f).sum())})" for f in sorted(set(S['folder']))))
    if names["exclude"]:
        print(f"  EXCLUDED scenes: {names['exclude']}")
    ax_tr = (TR["th"] % 180)
    print(f"  train wind-axis (mod 180) histogram, 30-deg bins: {np.histogram(ax_tr, bins=np.arange(0, 181, 30))[0].tolist()}")

    Xtr, thtr, _ = augment(TR["X"], TR["th"])
    Xva, thva, tagva = augment(VA["X"], VA["th"])
    Xte = TE["X"][:, None]
    Xva_orig = VA["X"][:, None]
    print(f"  after augmentation: train {len(Xtr)} samples | val {len(Xva)} samples")

    base_axis = axial_mean(TR["th"])
    base_val_loss = float(np.mean(1 - np.cos(np.radians(thva - base_axis)) ** 2))
    print(f"  BASELINE (constant {base_axis:.0f}/{base_axis + 180:.0f} deg): val loss {base_val_loss:.4f} "
          f"(the network must do better)")

    model = WindDirNet().to(DEVICE)
    print(f"  model params {sum(p.numel() for p in model.parameters()):,} | device {DEVICE}")
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    dl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(np.radians(thtr))),
                    batch_size=a.batch_size, shuffle=True, drop_last=True)
    print(f"  batch {a.batch_size} -> {len(dl)} batches / epoch\n")

    batch_rows, epoch_rows = [], []
    best, best_ep, wait, step, t0 = np.inf, 0, 0, 0, time.time()
    ckpt = os.path.join(a.out_dir, "best_model.pt")
    for ep in range(1, a.max_epochs + 1):
        model.train()
        run_l, n = 0.0, 0
        for b, (xb, tb) in enumerate(dl, 1):
            xb, tb = xb.to(DEVICE), tb.to(DEVICE)
            loss = loss_fn(model(xb), tb)
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1; run_l += loss.item() * len(xb); n += len(xb)
            batch_rows.append(dict(epoch=ep, batch=b, global_step=step, train_loss=loss.item(), val_loss=np.nan))
        tl, vl = run_l / n, mean_loss(model, Xva, thva)
        batch_rows[-1]["val_loss"] = vl
        epoch_rows.append(dict(epoch=ep, train_loss=tl, val_loss=vl))
        improved = vl < best - 1e-6
        if improved:
            best, best_ep, wait = vl, ep, 0
            torch.save(model.state_dict(), ckpt)
        else:
            wait += 1
        print(f"epoch {ep:3d} | train {tl:.4f} | val {vl:.4f} | best {best:.4f} (ep {best_ep}){' *' if improved else ''}"
              f" | {time.time() - t0:.0f}s", flush=True)
        if wait >= a.patience:
            print(f"Early stopping: no improvement of the validation loss for {a.patience} epochs.")
            break

    pd.DataFrame(batch_rows).to_csv(os.path.join(a.out_dir, "loss_table_per_batch.csv"), index=False)
    pd.DataFrame(epoch_rows).to_csv(os.path.join(a.out_dir, "loss_table_per_epoch.csv"), index=False)

    # ---- best model (best-validation epoch) ----
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    pv, pt = predict(model, Xva_orig), predict(model, Xte)
    tab = pd.concat([scene_table("VAL", VA, pv, base_axis), scene_table("TEST", TE, pt, base_axis)], ignore_index=True)
    tab.to_csv(os.path.join(a.out_dir, "metrics_by_scene.csv"), index=False)
    for nm, S, p in (("val", VA, pv), ("test", TE, pt)):
        pd.DataFrame(dict(uid=S["uid"], scene=S["folder"], wind_speed=S["speed"], copernicus_dir=S["th"],
                          pred_aliased_deg=p, aliased_err=alias_err(p, S["th"]),
                          pred_dealiased_deg=dealias(p, S["th"]))).to_csv(
            os.path.join(a.out_dir, f"predictions_{nm}.csv"), index=False)

    pd.set_option("display.width", 200)
    cols = ["split", "scene", "n", "true_axis", "pred_axis_mean", "scene_mean_err", "aliased_MAE", "baseline_MAE",
            "aliased_within10_pct", "dealiased_bias", "dealiased_cRMSd"]
    print(f"\n=== RESULTS (best epoch {best_ep}, val loss {best:.4f}, baseline {base_val_loss:.4f}) ===")
    print(tab[cols].round(1).to_string(index=False))
    t_all = tab[(tab.split == "TEST") & (tab.scene == "ALL")].iloc[0]
    verdict = "ResNet beats the baseline" if t_all.aliased_MAE < t_all.baseline_MAE - 2 else \
        "!! ResNet does NOT beat the baseline (texture not learned or too little data)"
    print(f"\nTEST aliased MAE {t_all.aliased_MAE:.1f} deg vs constant-baseline {t_all.baseline_MAE:.1f} deg "
          f"(random guess ~45) -> {verdict}")

    with open(os.path.join(a.out_dir, "config.json"), "w") as f:
        json.dump(dict(vars(a), best_epoch=best_ep, best_val_loss=best, baseline_val_loss=base_val_loss,
                       train_scenes=sorted(set(TR["folder"])), val_scenes=names["val"], test_scenes=names["test"]),
                  f, indent=2, default=str)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        ep_df = pd.DataFrame(epoch_rows)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(ep_df.epoch, ep_df.train_loss, label="train"); ax[0].plot(ep_df.epoch, ep_df.val_loss, label="val (scenes)")
        ax[0].axhline(base_val_loss, color="gray", ls="--", label="constant baseline"); ax[0].set_xlabel("epoch")
        ax[0].set_ylabel("1 - cos^2(diff)"); ax[0].legend()
        for nm, S, p, c in (("val", VA, pv, "tab:blue"), ("test", TE, pt, "tab:red")):
            ax[1].scatter(S["th"] % 180, p % 180, s=8, alpha=.6, c=c, label=nm)
        ax[1].plot([0, 180], [0, 180], "k--", lw=.8); ax[1].set_xlabel("Copernicus wind axis (deg mod 180)")
        ax[1].set_ylabel("ResNet axis (deg mod 180)"); ax[1].legend()
        plt.tight_layout(); plt.savefig(os.path.join(a.out_dir, "results.png"), dpi=150)
    except Exception as e:
        print("plot skipped:", e)
    print(f"\nOutputs: {a.out_dir}")
    return tab


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out_dir", required=True)
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k}", type=type(v), default=v)
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
