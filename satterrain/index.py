"""Embed the tile catalogue (EuroSAT + Sentinel-2 time series) into a FAISS index."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from satterrain.faiss_compat import faiss
import numpy as np
from PIL import Image

from satterrain.change import calibrate, calibrate_scores, location_pairs, split_locations
from satterrain.config import ARTIFACTS_DIR, TIMESERIES_DIR, device
from satterrain.data import eurosat, harmonize
from satterrain.models import load_encoder
from satterrain.transforms import embed_arrays, spatial_drift

log = logging.getLogger(__name__)

INDEX_DIR = ARTIFACTS_DIR / "index"


def _build_faiss(emb: np.ndarray, kind: str = "flat") -> faiss.Index:
    d = emb.shape[1]
    if kind == "flat":
        index = faiss.IndexFlatIP(d)
    elif kind == "hnsw":
        index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efSearch = 64
    else:
        raise ValueError(kind)
    index.add(np.ascontiguousarray(emb, dtype=np.float32))
    return index


def build(checkpoint: Optional[Path], out_dir: Path = INDEX_DIR, img_size: int = 128,
          timeseries: Path = TIMESERIES_DIR / "timeseries.npz", include_eurosat: bool = True,
          kind: str = "flat", quantile: float = 0.99) -> Path:
    """Embed every tile, write index.faiss + metadata, and calibrate the change threshold."""
    out_dir.mkdir(parents=True, exist_ok=True)
    enc = load_encoder(checkpoint)
    dev = device()
    embs: List[np.ndarray] = []
    meta: Dict[str, list] = {k: [] for k in ("source", "ref", "label", "loc_id", "date", "lat", "lon")}

    if timeseries.exists():
        z = np.load(timeseries)
        imgs = harmonize.apply(z["images"])
        e = embed_arrays(enc, imgs, img_size, dev)
        embs.append(e)
        n = len(imgs)
        meta["source"] += ["sentinel2"] * n
        meta["ref"] += list(range(n))
        meta["label"] += [None] * n
        meta["loc_id"] += [int(x) for x in z["loc_ids"]]
        meta["date"] += [str(x) for x in z["dates"]]
        meta["lat"] += [float(x) for x in z["lats"]]
        meta["lon"] += [float(x) for x in z["lons"]]
        cal = calibrate(e, z["loc_ids"], quantile=quantile)
        cal.save(out_dir / "change_calibration.json")
        log.info("global change threshold %.4f (heldout FPR %.3f)", cal.threshold, cal.heldout_false_positive_rate)

        # Spatial drift (the served change detector): calibrate on the same
        # location split, and precompute each tile's drift from its location's
        # earliest acquisition for the catalogue endpoints.
        loc_ids = z["loc_ids"]
        cal_locs, held_locs = split_locations(loc_ids)
        pc, ph = location_pairs(loc_ids, cal_locs), location_pairs(loc_ids, held_locs)
        s_cal = spatial_drift(enc, imgs[pc[:, 0]], imgs[pc[:, 1]], img_size, dev)
        s_held = spatial_drift(enc, imgs[ph[:, 0]], imgs[ph[:, 1]], img_size, dev)
        scal = calibrate_scores(s_cal, s_held, quantile, n_locations=len(cal_locs))
        scal.save(out_dir / "change_calibration_spatial.json")
        log.info("spatial change threshold %.4f (heldout FPR %.3f)", scal.threshold, scal.heldout_false_positive_rate)

        dates = z["dates"]
        first = np.empty(n, dtype=np.int64)
        for l in np.unique(loc_ids):
            idx = np.where(loc_ids == l)[0]
            first[idx] = idx[np.argmin(dates[idx])]
        np.save(out_dir / "spatial_from_first.npy", spatial_drift(enc, imgs[first], imgs, img_size, dev))

        # Persistent change (catalogue ranking): clouds, snow and low-sun shadow
        # that Sen2Cor misses are transient, while land-cover change persists.
        # Score a location by the MIN spatial drift over {2 earliest} x {2 latest}
        # acquisitions, so a single bad image cannot trigger a flag. Calibrated
        # the same way, on the same location split.
        pair_score = {}
        for pairs, scores in ((pc, s_cal), (ph, s_held)):
            for (i, j), s in zip(pairs, scores):
                pair_score[(int(i), int(j))] = pair_score[(int(j), int(i))] = float(s)
        held_set = set(int(l) for l in held_locs)
        persistent, p_cal, p_held = {}, [], []
        for l in np.unique(loc_ids):
            idx = np.where(loc_ids == l)[0]
            idx = idx[np.argsort(dates[idx])]
            if len(idx) < 4:
                continue
            s = min(pair_score[(int(i), int(j))] for i in idx[:2] for j in idx[-2:])
            persistent[int(l)] = s
            (p_held if int(l) in held_set else p_cal).append(s)
        pcal = calibrate_scores(np.array(p_cal), np.array(p_held), quantile, n_locations=len(p_cal))
        pcal.save(out_dir / "change_calibration_persistent.json")
        (out_dir / "persistent_drift.json").write_text(json.dumps(persistent))
        log.info("persistent change threshold %.4f (heldout FPR %.3f, %d locations)",
                 pcal.threshold, pcal.heldout_false_positive_rate, len(persistent))

    if include_eurosat:
        samples = eurosat.list_samples()
        imgs = np.stack([np.asarray(Image.open(p).convert("RGB")) for p, _ in samples])
        embs.append(embed_arrays(enc, imgs, img_size, dev))
        n = len(samples)
        meta["source"] += ["eurosat"] * n
        meta["ref"] += [str(Path(p).relative_to(eurosat.image_root())) for p, _ in samples]
        meta["label"] += [eurosat.CLASSES[l] for _, l in samples]
        meta["loc_id"] += [None] * n
        meta["date"] += [None] * n
        meta["lat"] += [None] * n
        meta["lon"] += [None] * n

    emb = np.concatenate(embs)
    faiss.write_index(_build_faiss(emb, kind), str(out_dir / "index.faiss"))
    np.save(out_dir / "embeddings.npy", emb)
    (out_dir / "meta.json").write_text(json.dumps(meta))
    (out_dir / "config.json").write_text(json.dumps({
        "checkpoint": str(Path(checkpoint).resolve()) if checkpoint else None, "img_size": img_size,
        "timeseries": str(Path(timeseries).resolve()), "kind": kind, "n": int(len(emb)), "dim": int(emb.shape[1]),
    }, indent=1))
    log.info("indexed %d tiles -> %s", len(emb), out_dir)
    return out_dir
