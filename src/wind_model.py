"""
wind_model.py - loads the trained ResNet ensemble and predicts the wind axis of new patches.
Used by the inference and the API so that the preprocessing is identical to training.

    from wind_model import load_ensemble, predict_axis
    models, cfg = load_ensemble(".../final_model")
    axis_deg, conf = predict_axis(models, cfg, X)      # X: (N,50,50) per-patch z-score of VV (as in build_dataset)

axis_deg : wind axis, degrees clockwise from north, [0, 180); the direction is axis or axis + 180
conf     : 0..1, agreement of the predictions of all models x 8 transforms
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

import train_resnet as T
from train_resnet_v2 import D4, axial_mean_R, predict_tta, preprocess  # noqa: F401  (re-export)


def load_ensemble(model_dir, device=None):
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    dev = device or T.DEVICE
    models = []
    for f in cfg["model_files"]:
        m = T.WindDirNet(dropout=cfg.get("dropout", 0.0)).to(dev)
        m.load_state_dict(torch.load(os.path.join(model_dir, f), map_location=dev))
        m.eval()
        models.append(m)
    return models, cfg


def predict_axis(models, cfg, X, bs=256):
    Xp = preprocess(np.asarray(X, np.float32), cfg["lowpass_px"])
    P = np.concatenate([predict_tta(m, Xp, bs) for m in models], axis=1)     # (N, n_models*8)
    return axial_mean_R(P)
