"""FastAPI service: terrain similarity search + change detection.

Run:  python -m satterrain.cli serve       (or: uvicorn satterrain.api:app)
Env:  SATTERRAIN_INDEX_DIR (default artifacts/index)

Change detection uses *spatial* embedding drift (top-10% most-changed cells of
the layer-3 feature map) with a threshold calibrated on seasonal variation;
global (pooled-embedding) drift is reported alongside for reference.
"""
from __future__ import annotations

import base64
import io
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, Response
from PIL import Image
from pydantic import BaseModel

from satterrain.change import Calibration, drift
from satterrain.config import TILE_SIZE, device
from satterrain.data import eurosat, harmonize
from satterrain.faiss_compat import faiss
from satterrain.index import INDEX_DIR
from satterrain.models import load_encoder
from satterrain.transforms import embed_arrays, spatial_drift


def _load_cal(path: Path) -> Optional[Calibration]:
    return Calibration.load(path) if path.exists() else None


class Engine:
    def __init__(self, index_dir: Path):
        cfg = json.loads((index_dir / "config.json").read_text())
        self.cfg = cfg
        self.img_size = int(cfg["img_size"])
        self.index = faiss.read_index(str(index_dir / "index.faiss"))
        self.emb = np.load(index_dir / "embeddings.npy")
        self.meta = json.loads((index_dir / "meta.json").read_text())
        ckpt = cfg.get("checkpoint")
        self.encoder = load_encoder(Path(ckpt) if ckpt else None)
        self.device = device()
        self.global_cal = _load_cal(index_dir / "change_calibration.json")
        self.spatial_cal = _load_cal(index_dir / "change_calibration_spatial.json")
        sff = index_dir / "spatial_from_first.npy"
        self.spatial_from_first = np.load(sff) if sff.exists() else None
        self.persistent_cal = _load_cal(index_dir / "change_calibration_persistent.json")
        pd_path = index_dir / "persistent_drift.json"
        self.persistent = {int(k): v for k, v in json.loads(pd_path.read_text()).items()} \
            if pd_path.exists() else {}
        ts = Path(cfg["timeseries"])
        self.ts_images = harmonize.apply(np.load(ts)["images"]) if ts.exists() else None
        self.loc_to_ids = {}
        for i, (src, loc) in enumerate(zip(self.meta["source"], self.meta["loc_id"])):
            if src == "sentinel2":
                self.loc_to_ids.setdefault(int(loc), []).append(i)
        for ids in self.loc_to_ids.values():
            ids.sort(key=lambda i: self.meta["date"][i])

    def __len__(self) -> int:
        return self.index.ntotal

    def record(self, i: int, score: Optional[float] = None) -> dict:
        r = {k: self.meta[k][i] for k in ("source", "label", "loc_id", "date", "lat", "lon")}
        r["tile_id"] = i
        r["image_url"] = f"/tiles/{i}.png"
        if score is not None:
            r["similarity"] = float(score)
        return r

    def tile_image(self, i: int) -> np.ndarray:
        src, ref = self.meta["source"][i], self.meta["ref"][i]
        if src == "sentinel2":
            return self.ts_images[int(ref)]
        return np.asarray(Image.open(eurosat.image_root() / ref).convert("RGB"))

    def embed(self, images: np.ndarray) -> np.ndarray:
        return embed_arrays(self.encoder, images, self.img_size, self.device)

    def search(self, q: np.ndarray, k: int, exclude: Optional[int] = None, source: Optional[str] = None):
        fetch = min(len(self), k * (4 if source else 1) + 1)
        scores, ids = self.index.search(np.ascontiguousarray(q[None], dtype=np.float32), fetch)
        out = []
        for s, i in zip(scores[0], ids[0]):
            if i < 0 or i == exclude or (source and self.meta["source"][i] != source):
                continue
            out.append(self.record(int(i), float(s)))
            if len(out) == k:
                break
        return out

    def assess(self, spatial: float, global_drift: Optional[float] = None) -> dict:
        out = {"spatial_drift": float(spatial)}
        if self.spatial_cal:
            c = self.spatial_cal
            out.update({"threshold": c.threshold, "z_score": float(c.score(spatial)),
                        "changed": bool(c.flag(spatial))})
        if global_drift is not None:
            out["global_drift"] = float(global_drift)
            if self.global_cal:
                out["global_changed"] = bool(self.global_cal.flag(global_drift))
        return out

    def compare(self, before: np.ndarray, after: np.ndarray) -> dict:
        s = float(spatial_drift(self.encoder, before[None], after[None], self.img_size, self.device)[0])
        e = self.embed(np.stack([before, after]))
        return self.assess(s, float(drift(e[0], e[1])[0]))


@lru_cache(maxsize=1)
def engine() -> Engine:
    d = Path(os.environ.get("SATTERRAIN_INDEX_DIR", INDEX_DIR))
    if not (d / "index.faiss").exists():
        raise RuntimeError(f"no index at {d}; run `python -m satterrain.cli build-index` first")
    return Engine(d)


def _read_upload(data: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(400, "could not decode image")
    if img.size != (TILE_SIZE, TILE_SIZE):
        img = img.resize((TILE_SIZE, TILE_SIZE), Image.BICUBIC)
    return np.asarray(img)


def _png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


app = FastAPI(title="Satellite Terrain Search & Change Detection", version="0.1.0")


class SearchResponse(BaseModel):
    query: dict
    results: List[dict]


@app.get("/health")
def health():
    e = engine()
    return {"status": "ok", "n_tiles": len(e), "dim": int(e.emb.shape[1]), "checkpoint": e.cfg.get("checkpoint"),
            "n_locations": len(e.loc_to_ids),
            "change_threshold": e.spatial_cal.threshold if e.spatial_cal else None}


@app.get("/tiles/{tile_id}.png")
def tile_png(tile_id: int):
    e = engine()
    if not 0 <= tile_id < len(e):
        raise HTTPException(404, "unknown tile")
    return Response(_png(e.tile_image(tile_id)), media_type="image/png")


@app.get("/search/{tile_id}", response_model=SearchResponse)
def search_by_id(tile_id: int, k: int = Query(10, ge=1, le=100), source: Optional[str] = None):
    """Nearest neighbours of a tile already in the catalogue."""
    e = engine()
    if not 0 <= tile_id < len(e):
        raise HTTPException(404, "unknown tile")
    return {"query": e.record(tile_id), "results": e.search(e.emb[tile_id], k, exclude=tile_id, source=source)}


@app.post("/search", response_model=SearchResponse)
async def search_by_image(file: UploadFile = File(...), k: int = Query(10, ge=1, le=100),
                          source: Optional[str] = None):
    """Upload an RGB tile at ~10 m/px (resized to 64x64)."""
    e = engine()
    q = e.embed(_read_upload(await file.read())[None])[0]
    return {"query": {"uploaded": file.filename}, "results": e.search(q, k, source=source)}


@app.get("/locations/{loc_id}/changes")
def location_changes(loc_id: int):
    """Drift of each acquisition relative to the earliest one, with change flags."""
    e = engine()
    ids = e.loc_to_ids.get(loc_id)
    if not ids:
        raise HTTPException(404, "unknown location")
    if e.spatial_from_first is None:
        raise HTTPException(503, "no change calibration available")
    rows = []
    for i in ids:
        g = float(drift(e.emb[ids[0]], e.emb[i])[0])
        rows.append({"date": e.meta["date"][i], "tile_id": i, "image_url": f"/tiles/{i}.png",
                     **e.assess(float(e.spatial_from_first[e.meta["ref"][i]]), g)})
    return {"loc_id": loc_id, "lat": e.meta["lat"][ids[0]], "lon": e.meta["lon"][ids[0]], "timeline": rows}


@app.get("/changes/top")
def top_changes(n: int = Query(20, ge=1, le=500), changed_only: bool = False):
    """Catalogue locations ranked by *persistent* change: the minimum spatial
    drift between either of the two earliest and either of the two latest
    acquisitions, so a single cloudy/snowy/shadowed image cannot rank a location."""
    e = engine()
    if not e.persistent or e.persistent_cal is None:
        raise HTTPException(503, "no persistent-change calibration available")
    cal = e.persistent_cal
    rows = []
    for loc, score in e.persistent.items():
        ids = e.loc_to_ids[loc]
        first, last = ids[0], ids[-1]
        flagged = bool(cal.flag(score))
        if changed_only and not flagged:
            continue
        rows.append({"loc_id": loc, "from": e.meta["date"][first], "to": e.meta["date"][last],
                     "lat": e.meta["lat"][first], "lon": e.meta["lon"][first],
                     "before_url": f"/tiles/{first}.png", "after_url": f"/tiles/{last}.png",
                     "persistent_drift": float(score), "threshold": cal.threshold,
                     "z_score": float(cal.score(score)), "changed": flagged})
    rows.sort(key=lambda r: -r["persistent_drift"])
    return rows[:n]


@app.post("/change")
async def change_from_images(before: UploadFile = File(...), after: UploadFile = File(...)):
    """Compare two co-registered tiles of the same place."""
    e = engine()
    return e.compare(_read_upload(await before.read()), _read_upload(await after.read()))


@app.get("/change/live")
def change_live(lat: float = Query(..., ge=-90, le=90), lon: float = Query(..., ge=-180, le=180),
                before: str = Query(..., description="YYYY-MM-DD/YYYY-MM-DD"),
                after: str = Query(..., description="YYYY-MM-DD/YYYY-MM-DD")):
    """Fetch the least-cloudy Sentinel-2 tile in each window from Planetary Computer and compare."""
    from satterrain.data.sentinel import fetch_best_tile

    e = engine()
    tiles = []
    for rng in (before, after):
        try:
            start, end = rng.split("/")
        except ValueError:
            raise HTTPException(400, "date ranges must look like 2019-05-01/2019-09-30")
        t = fetch_best_tile(lat, lon, start, end)
        if t is None:
            raise HTTPException(404, f"no cloud-free Sentinel-2 tile for {rng}")
        tiles.append(t)
    imgs = harmonize.apply(np.stack([t.image for t in tiles]))
    res = e.compare(imgs[0], imgs[1])
    for name, t, img in zip(("before", "after"), tiles, imgs):
        res[name] = {"date": t.date, "item_id": t.item_id,
                     "image": "data:image/png;base64," + base64.b64encode(_png(img)).decode()}
    res["similar_to_after"] = e.search(e.embed(imgs[1:])[0], 5)
    return res


@app.get("/", response_class=HTMLResponse)
def ui():
    return (Path(__file__).parent / "static" / "index.html").read_text()
