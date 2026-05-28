"""SeCo-style contrastive pretraining on multi-date Sentinel-2 tiles.

Positive pair = two different acquisition dates of the same location (each
independently augmented). Negatives = other locations in the MoCo queue.
"""
from __future__ import annotations

import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from satterrain.config import ARTIFACTS_DIR, TIMESERIES_DIR, device
from satterrain.data import harmonize
from satterrain.models import MoCo
from satterrain.transforms import to_pil, train_transform

log = logging.getLogger(__name__)


class TemporalPairDataset(Dataset):
    """One item per location: (aug(tile @ date i), aug(tile @ date j)), i != j."""

    def __init__(self, npz_path: Path, transform, loc_subset: Optional[np.ndarray] = None):
        z = np.load(npz_path)
        self.images = harmonize.apply(z["images"])
        loc_ids = z["loc_ids"]
        groups = {}
        for idx, loc in enumerate(loc_ids):
            groups.setdefault(int(loc), []).append(idx)
        locs = sorted(l for l, g in groups.items() if len(g) >= 2)
        if loc_subset is not None:
            keep = set(int(x) for x in loc_subset)
            locs = [l for l in locs if l in keep]
        self.groups = [groups[l] for l in locs]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, i):
        a, b = random.sample(self.groups[i], 2)
        return self.transform(to_pil(self.images[a])), self.transform(to_pil(self.images[b]))


class SameImagePairDataset(Dataset):
    """Classic instance discrimination on single images (optional extra data)."""

    def __init__(self, paths, transform):
        self.paths = list(paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image

        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), self.transform(img)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(
    npz_path: Path = TIMESERIES_DIR / "timeseries.npz",
    out_dir: Path = ARTIFACTS_DIR / "seco",
    arch: str = "resnet18",
    init: str = "imagenet",
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 0.03,
    weight_decay: float = 1e-4,
    img_size: int = 128,
    queue_size: int = 4096,
    temperature: float = 0.2,
    momentum: float = 0.99,
    workers: int = 4,
    extra_image_paths=None,
    eval_every: int = 10,
    seed: int = 0,
) -> Path:
    """Train and save `out_dir/encoder.pt` (backbone weights only)."""
    from satterrain.evaluate import precision_at_k_split

    _set_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = device()
    tf = train_transform(img_size)

    ds: Dataset = TemporalPairDataset(npz_path, tf)
    n_locs = len(ds)
    if extra_image_paths:
        ds = ConcatDataset([ds, SameImagePairDataset(extra_image_paths, tf)])
    # The queue must be much smaller than the number of distinct locations or
    # it fills up with stale keys from the positive's own location.
    queue_size = int(min(queue_size, max(batch_size, (len(ds) // 2 // batch_size) * batch_size)))
    log.info("training on %d locations (%d items), queue=%d, device=%s", n_locs, len(ds), queue_size, dev)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=workers,
                        persistent_workers=workers > 0)
    if len(loader) == 0:
        raise RuntimeError(f"not enough data ({len(ds)} items) for batch size {batch_size}")
    model = MoCo(arch, pretrained=(init == "imagenet"), queue_size=queue_size,
                 temperature=temperature, momentum=momentum).to(dev)
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=lr * batch_size / 256, momentum=0.9, weight_decay=weight_decay)
    total = epochs * len(loader)
    warmup = min(len(loader) * 2, total // 10)

    def lr_at(step: int) -> float:
        base = lr * batch_size / 256
        if step < warmup:
            return base * (step + 1) / warmup
        return base * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))

    history = []
    best = -1.0
    step = 0
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        losses, accs = [], []
        for x_q, x_k in loader:
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            x_q, x_k = x_q.to(dev, non_blocking=True), x_k.to(dev, non_blocking=True)
            loss, acc = model(x_q, x_k)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            accs.append(acc.item())
            step += 1
        rec = {"epoch": epoch + 1, "loss": float(np.mean(losses)), "inst_acc": float(np.mean(accs)),
               "lr": lr_at(step - 1), "elapsed_s": round(time.time() - t0, 1)}

        state = {"arch": arch, "backbone": model.backbone.state_dict(), "epoch": epoch + 1,
                 "img_size": img_size, "init": init}
        torch.save(state, out_dir / "last.pt")

        if eval_every and ((epoch + 1) % eval_every == 0 or epoch + 1 == epochs):
            # Model selection on a validation split carved out of EuroSAT *train*;
            # the EuroSAT test split is never touched during training.
            model.eval()
            val = precision_at_k_split(out_dir / "last.pt", split_name="val", img_size=img_size)
            rec["val_p@10"] = val
            if val > best:
                best = val
                torch.save(state, out_dir / "encoder.pt")
        log.info(json.dumps(rec))
        history.append(rec)
        (out_dir / "history.json").write_text(json.dumps(history, indent=1))

    if not (out_dir / "encoder.pt").exists():
        torch.save(state, out_dir / "encoder.pt")
    return out_dir / "encoder.pt"
