"""Embedding-drift change detection with a threshold calibrated on seasonal variation.

drift(a, b) = 1 - cos(e_a, e_b) for embeddings of the same location at two dates.

Calibration: across many random locations, every pair of dates differs by
season and year, but the overwhelming majority of locations do not change
land-cover. The distribution of drift over those pairs is therefore an
empirical null of "seasonal / atmospheric variation only". The threshold is a
high quantile of that null (default 99th percentile -> ~1% false-positive rate
on unchanged land by construction), checked on held-out locations.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from satterrain.config import ARTIFACTS_DIR

CALIBRATION_PATH = ARTIFACTS_DIR / "change_calibration.json"


def drift(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine distance between L2-normalised embeddings (row-wise)."""
    return 1.0 - np.sum(np.atleast_2d(a) * np.atleast_2d(b), axis=1)


def pairwise_location_drifts(emb: np.ndarray, loc_ids: np.ndarray, locs: Optional[np.ndarray] = None) -> np.ndarray:
    """All within-location, cross-date drifts."""
    out: List[float] = []
    keep = None if locs is None else set(int(l) for l in locs)
    groups: Dict[int, List[int]] = {}
    for i, l in enumerate(loc_ids):
        l = int(l)
        if keep is None or l in keep:
            groups.setdefault(l, []).append(i)
    for idx in groups.values():
        for i, j in combinations(idx, 2):
            out.append(float(drift(emb[i], emb[j])[0]))
    return np.array(out, dtype=np.float64)


@dataclass
class Calibration:
    threshold: float
    quantile: float
    null_mean: float
    null_std: float
    n_pairs_calibration: int
    n_locations_calibration: int
    heldout_false_positive_rate: float
    n_pairs_heldout: int

    def save(self, path: Path = CALIBRATION_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=1))

    @classmethod
    def load(cls, path: Path = CALIBRATION_PATH) -> "Calibration":
        return cls(**json.loads(Path(path).read_text()))

    def score(self, d: np.ndarray) -> np.ndarray:
        """Drift expressed in standard deviations above the seasonal null mean."""
        return (np.asarray(d) - self.null_mean) / max(self.null_std, 1e-9)

    def flag(self, d: np.ndarray) -> np.ndarray:
        return np.asarray(d) > self.threshold


def split_locations(loc_ids: np.ndarray, heldout_frac: float = 0.2, seed: int = 0):
    """-> (calibration_locs, heldout_locs); the same split everywhere."""
    locs = np.unique(loc_ids)
    rng = np.random.RandomState(seed)
    rng.shuffle(locs)
    n_held = int(len(locs) * heldout_frac)
    return locs[n_held:], locs[:n_held]


def location_pairs(loc_ids: np.ndarray, locs) -> np.ndarray:
    """All (i, j) index pairs of different dates at the same location, for `locs`."""
    keep = set(int(l) for l in locs)
    groups: Dict[int, List[int]] = {}
    for i, l in enumerate(loc_ids):
        if int(l) in keep:
            groups.setdefault(int(l), []).append(i)
    pairs = [p for idx in groups.values() for p in combinations(idx, 2)]
    return np.array(pairs, dtype=np.int64).reshape(-1, 2)


def calibrate(emb: np.ndarray, loc_ids: np.ndarray, quantile: float = 0.99,
              heldout_frac: float = 0.2, seed: int = 0) -> Calibration:
    cal, held = split_locations(loc_ids, heldout_frac, seed)
    d_cal = pairwise_location_drifts(emb, loc_ids, cal)
    d_held = pairwise_location_drifts(emb, loc_ids, held)
    return calibrate_scores(d_cal, d_held, quantile, n_locations=len(cal))


def calibrate_scores(d_cal: np.ndarray, d_held: np.ndarray, quantile: float = 0.99,
                     n_locations: int = 0) -> Calibration:
    """Threshold = `quantile` of calibration-pair scores; FPR checked on held-out pairs."""
    if len(d_cal) == 0:
        raise ValueError("no multi-date locations to calibrate on")
    thr = float(np.quantile(d_cal, quantile))
    return Calibration(
        threshold=thr, quantile=quantile, null_mean=float(d_cal.mean()), null_std=float(d_cal.std()),
        n_pairs_calibration=int(len(d_cal)), n_locations_calibration=int(n_locations),
        heldout_false_positive_rate=float((d_held > thr).mean()) if len(d_held) else float("nan"),
        n_pairs_heldout=int(len(d_held)),
    )


def timeline(emb: np.ndarray, dates: List[str], cal: Calibration) -> List[dict]:
    """Drift of every date relative to the earliest date for one location."""
    order = np.argsort(dates)
    ref = emb[order[0]]
    rows = []
    for i in order:
        d = float(drift(ref, emb[i])[0])
        rows.append({"date": dates[i], "drift_from_first": d, "z": float(cal.score(d)),
                     "changed": bool(cal.flag(d))})
    return rows
