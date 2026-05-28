"""Shared paths and constants."""
from __future__ import annotations

import os
from pathlib import Path

import torch

ROOT = Path(os.environ.get("SATTERRAIN_ROOT", Path(__file__).resolve().parent.parent))
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"

EUROSAT_DIR = DATA_DIR / "eurosat"
TIMESERIES_DIR = DATA_DIR / "timeseries"
OSCD_DIR = DATA_DIR / "oscd"

# Tiles are 64x64 RGB at 10 m/px, matching EuroSAT.
TILE_SIZE = 64

# Sentinel-2 surface reflectance is stored as uint16 DN = reflectance * 10000.
# EuroSAT RGB JPEGs are roughly a linear stretch of 0..~3000 DN to 0..255, so we
# use the same clip when converting Sentinel-2 reads into 8-bit tiles.
REFLECTANCE_CLIP = 3000.0

# ImageNet normalisation (both the baseline and our encoder see the same input
# statistics so the comparison is apples to apples).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dirs() -> None:
    for d in (DATA_DIR, ARTIFACTS_DIR, EUROSAT_DIR, TIMESERIES_DIR, OSCD_DIR):
        d.mkdir(parents=True, exist_ok=True)
