"""API tests against a built index. Skipped if no index exists.

    SATTERRAIN_INDEX_DIR=artifacts/index pytest tests/test_api.py
"""
import io
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import satterrain  # noqa: F401
from satterrain.config import ARTIFACTS_DIR

INDEX = Path(os.environ.get("SATTERRAIN_INDEX_DIR", ARTIFACTS_DIR / "index"))
pytestmark = pytest.mark.skipif(not (INDEX / "index.faiss").exists(), reason="no index built")


@pytest.fixture(scope="module")
def client():
    os.environ["SATTERRAIN_INDEX_DIR"] = str(INDEX)
    from fastapi.testclient import TestClient

    from satterrain import api

    api.engine.cache_clear()
    return TestClient(api.app)


def _png(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def test_health(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["n_tiles"] > 0


def test_search_by_id_returns_self_similar(client):
    r = client.get("/search/0?k=5").json()
    assert len(r["results"]) == 5
    sims = [x["similarity"] for x in r["results"]]
    assert sims == sorted(sims, reverse=True)
    assert all(x["tile_id"] != 0 for x in r["results"])


def test_search_by_upload_finds_exact_tile(client):
    png = client.get("/tiles/0.png").content
    r = client.post("/search?k=1", files={"file": ("q.png", png, "image/png")}).json()
    assert r["results"][0]["tile_id"] == 0
    assert r["results"][0]["similarity"] > 0.999


def test_source_filter(client):
    r = client.get("/search/0?k=5&source=eurosat").json()
    assert all(x["source"] == "eurosat" for x in r["results"])


def test_change_identical_images_not_flagged(client):
    png = client.get("/tiles/0.png").content
    r = client.post("/change", files={"before": ("a.png", png), "after": ("b.png", png)}).json()
    assert r["spatial_drift"] < 1e-4 and r["global_drift"] < 1e-4 and r["changed"] is False


def test_change_lake_to_industrial_flagged(client):
    from satterrain import api

    labels = api.engine().meta["label"]
    lake, industry = labels.index("SeaLake"), labels.index("Industrial")
    before = client.get(f"/tiles/{lake}.png").content
    after = client.get(f"/tiles/{industry}.png").content
    r = client.post("/change", files={"before": ("a.png", before), "after": ("b.png", after)}).json()
    assert r["changed"] is True and r["spatial_drift"] > r["threshold"]


def test_location_timeline_and_top_changes(client):
    top = client.get("/changes/top?n=3").json()
    assert len(top) == 3 and top[0]["persistent_drift"] >= top[1]["persistent_drift"]
    tl = client.get(f"/locations/{top[0]['loc_id']}/changes").json()["timeline"]
    assert tl[0]["spatial_drift"] < 1e-4
    flagged = client.get("/changes/top?n=500&changed_only=true").json()
    assert all(r["changed"] for r in flagged)


def test_bad_upload(client):
    r = client.post("/search", files={"file": ("x.png", b"not an image")})
    assert r.status_code == 400
