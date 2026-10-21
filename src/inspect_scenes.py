"""
inspect_scenes.py - montage of patches of selected scenes for visual inspection.
Each patch is labelled with its id and the Copernicus wind direction (from) and speed, to check wind streaks,
fronts, ship wakes and other features.

  python inspect_scenes.py --npz .../stageA/dataset_A.npz --out_dir .../cv_results --folders durban,Chennai --n 24
"""
import argparse, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--npz", required=True); ap.add_argument("--out_dir", required=True)
ap.add_argument("--folders", required=True); ap.add_argument("--n", type=int, default=24)
a = ap.parse_args()
d = np.load(a.npz, allow_pickle=False)
os.makedirs(a.out_dir, exist_ok=True)
rng = np.random.default_rng(0)
for fol in [f.strip() for f in a.folders.split(",")]:
    idx = np.where(np.char.lower(d["folder"]) == fol.lower())[0]
    if len(idx) == 0:
        print("folder not found:", fol, "| available:", sorted(set(d["folder"]))); continue
    pick = np.sort(rng.choice(idx, size=min(a.n, len(idx)), replace=False))
    cols = 6; rows = int(np.ceil(len(pick) / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(cols * 2.1, rows * 2.3), squeeze=False)
    for axx in ax.ravel(): axx.axis("off")
    for axx, i in zip(ax.ravel(), pick):
        axx.imshow(d["X"][i], cmap="gray", vmin=-2.5, vmax=2.5)
        axx.set_title(f"{d['uid'][i].split('_')[-1]}  {d['theta_from'][i]:.0f}deg {d['wind_speed'][i]:.1f}m/s", fontsize=7)
    fig.suptitle(f"{fol}  ({len(idx)} patches, z-score, north-up)  - gray scale -2.5..+2.5 sigma")
    plt.tight_layout(); out = os.path.join(a.out_dir, f"montage_{fol}.png"); plt.savefig(out, dpi=120); plt.close()
    print("saved", out)
