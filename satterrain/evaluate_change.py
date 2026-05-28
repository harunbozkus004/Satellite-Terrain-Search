"""Evaluate change detection: OSCD benchmark + robustness to seasonal variation.

Three detectors are compared, each producing a score for a (before, after) pair:
  * pixel_difference  - mean absolute RGB difference (naive baseline)
  * baseline_imagenet - embedding drift with an ImageNet ResNet
  * seco              - embedding drift with our SeCo-trained ResNet

Part A - OSCD (urban change, co-registered pairs, Daudt et al. 2018).
  64x64 tiles (stride 32); "changed" if >= m of pixels are in the change mask,
  "unchanged" if 0% (in-between ignored). Reported for m=5% and m=30%.

Part B - change vs. seasonal variation (the real-world monitoring problem).
  Positives: OSCD tiles with >= 30% changed pixels.
  Negatives: held-out Sentinel-2 locations, same place on different dates
             (seasons/years differ, land cover overwhelmingly unchanged).
  Every detector's threshold is calibrated the SAME way: the 99th percentile
  of its score on calibration-split seasonal pairs (no OSCD labels used).
  We report AUROC, and recall on real change / false-alarm rate on seasonal
  pairs at that threshold.

Caveat: OSCD is L1C (top of atmosphere) while our time series is L2A
harmonised to an L1C-like look, so Part B mixes two (closely matched) sources.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from satterrain.change import drift, location_pairs, split_locations
from satterrain.config import TIMESERIES_DIR, device
from satterrain.data import harmonize, oscd
from satterrain.models import load_encoder
from satterrain.transforms import embed_arrays, spatial_drift


def oscd_tiles():
    t1, t2, fr = [], [], []
    for pair in oscd.iter_pairs("all"):
        t = oscd.tile_pairs(pair)
        t1.append(t["img1"])
        t2.append(t["img2"])
        fr.append(t["frac"])
    return np.concatenate(t1), np.concatenate(t2), np.concatenate(fr)


def seasonal_pairs(npz: Path = TIMESERIES_DIR / "timeseries.npz", heldout_frac: float = 0.2, seed: int = 0):
    """Same split as change.calibrate: returns images + (i, j) pairs for calibration and held-out locations."""
    z = np.load(npz)
    imgs = harmonize.apply(z["images"])
    cal, held = split_locations(z["loc_ids"], heldout_frac, seed)
    return imgs, location_pairs(z["loc_ids"], cal), location_pairs(z["loc_ids"], held)


def _pixel_score(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(a.astype(np.float32) - b.astype(np.float32)).mean(axis=(1, 2, 3))


def _detectors(checkpoint: Path, img_size: int) -> Dict[str, Callable]:
    dev = device()
    encs = {"baseline_imagenet": load_encoder(None), "seco": load_encoder(checkpoint)}

    def emb_detector(enc):
        cache: Dict[int, np.ndarray] = {}

        def embed(x: np.ndarray) -> np.ndarray:
            key = id(x)
            if key not in cache:
                cache[key] = embed_arrays(enc, x, img_size, dev)
            return cache[key]

        def score(x1, idx1, x2, idx2):
            return drift(embed(x1)[idx1], embed(x2)[idx2])

        return score

    def spatial_detector(enc):
        def score(x1, idx1, x2, idx2):
            return spatial_drift(enc, x1[idx1], x2[idx2], img_size, dev)

        return score

    def pixel(x1, idx1, x2, idx2):
        return _pixel_score(x1[idx1], x2[idx2])

    dets = {"pixel_difference": pixel}
    for k, e in encs.items():
        dets[k] = emb_detector(e)
        dets[k + "_spatial"] = spatial_detector(e)
    return dets


def _auroc(pos: np.ndarray, neg: np.ndarray) -> Tuple[float, float]:
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    s = np.r_[pos, neg]
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def compare(checkpoint: Path, img_size: int = 128, quantile: float = 0.99,
            out: Optional[Path] = None) -> Dict[str, Dict[str, float]]:
    o1, o2, frac = oscd_tiles()
    ts, cal_pairs, held_pairs = seasonal_pairs()
    all_o = np.arange(len(o1))
    report: Dict[str, Dict[str, float]] = {}
    for name, score in _detectors(checkpoint, img_size).items():
        s_oscd = score(o1, all_o, o2, all_o)
        s_cal = score(ts, cal_pairs[:, 0], ts, cal_pairs[:, 1])
        s_held = score(ts, held_pairs[:, 0], ts, held_pairs[:, 1])
        thr = float(np.quantile(s_cal, quantile))
        unchanged = s_oscd[frac == 0]
        r: Dict[str, float] = {}
        for m in (0.05, 0.30):
            r[f"oscd_auroc_ge{int(m * 100)}"], r[f"oscd_ap_ge{int(m * 100)}"] = _auroc(s_oscd[frac >= m], unchanged)
        big = s_oscd[frac >= 0.30]
        r["vs_seasonal_auroc"], r["vs_seasonal_ap"] = _auroc(big, s_held)
        r["threshold"] = thr
        r["recall_on_oscd_ge30"] = float((big > thr).mean())
        r["false_alarm_rate_seasonal_heldout"] = float((s_held > thr).mean())
        r["false_alarm_rate_oscd_unchanged"] = float((unchanged > thr).mean())
        report[name] = r
    report["_counts"] = {"oscd_changed_ge5": int((frac >= 0.05).sum()), "oscd_changed_ge30": int((frac >= 0.30).sum()),
                         "oscd_unchanged": int((frac == 0).sum()), "seasonal_cal_pairs": int(len(cal_pairs)),
                         "seasonal_heldout_pairs": int(len(held_pairs)), "quantile": quantile, "img_size": img_size}
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1))
    return report
