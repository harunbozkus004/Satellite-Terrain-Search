"""Retrieval evaluation on EuroSAT: precision@k of nearest neighbours.

Protocol
--------
* EuroSAT RGB, fixed stratified 80/20 split (seed 0).
* "test": every test image is a query against an index of all train images.
* "val":  10% of train images are queries against the other 90% of train
          (used only for checkpoint selection during training).
* precision@k = fraction of the k nearest neighbours (cosine similarity on
  L2-normalised embeddings) whose land-cover class matches the query's.
  Labels are used for scoring only; no model ever trains on them.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

from satterrain.faiss_compat import faiss
import numpy as np
from PIL import Image

from satterrain.config import device
from satterrain.data import eurosat
from satterrain.models import load_encoder
from satterrain.transforms import embed_arrays


@lru_cache(maxsize=1)
def _load_eurosat() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train, test = eurosat.split(eurosat.list_samples(), test_frac=0.2, seed=0)

    def load(samples):
        imgs = np.stack([np.asarray(Image.open(p).convert("RGB")) for p, _ in samples])
        return imgs, np.array([l for _, l in samples])

    xtr, ytr = load(train)
    xte, yte = load(test)
    return xtr, ytr, xte, yte


def split_arrays(split_name: str):
    """-> (index_imgs, index_labels, query_imgs, query_labels)."""
    xtr, ytr, xte, yte = _load_eurosat()
    if split_name == "test":
        return xtr, ytr, xte, yte
    if split_name == "val":
        rng = np.random.RandomState(1)
        perm = rng.permutation(len(xtr))
        n_val = len(xtr) // 10
        q, idx = perm[:n_val], perm[n_val:]
        return xtr[idx], ytr[idx], xtr[q], ytr[q]
    raise ValueError(split_name)


def precision_at_k(index_emb: np.ndarray, index_labels: np.ndarray, query_emb: np.ndarray,
                   query_labels: np.ndarray, k: int = 10) -> Dict[str, object]:
    index = faiss.IndexFlatIP(index_emb.shape[1])
    index.add(np.ascontiguousarray(index_emb, dtype=np.float32))
    _, nn = index.search(np.ascontiguousarray(query_emb, dtype=np.float32), k)
    hits = index_labels[nn] == query_labels[:, None]
    per_query = hits.mean(1)
    per_class = {eurosat.CLASSES[c]: float(per_query[query_labels == c].mean()) for c in np.unique(query_labels)}
    return {"p@k": float(per_query.mean()), "p@1": float(hits[:, 0].mean()), "k": k, "per_class": per_class,
            "n_queries": int(len(query_labels)), "n_index": int(len(index_labels))}


def evaluate_checkpoint(checkpoint: Optional[Path], split_name: str = "test", img_size: int = 128,
                        k: int = 10, arch: str = "resnet18") -> Dict[str, object]:
    enc = load_encoder(checkpoint, arch=arch)
    xi, yi, xq, yq = split_arrays(split_name)
    dev = device()
    ei = embed_arrays(enc, xi, img_size, dev)
    eq = embed_arrays(enc, xq, img_size, dev)
    res = precision_at_k(ei, yi, eq, yq, k=k)
    res.update({"checkpoint": str(checkpoint) if checkpoint else "imagenet", "split": split_name,
                "img_size": img_size})
    return res


def precision_at_k_split(checkpoint: Path, split_name: str = "val", img_size: int = 128, k: int = 10) -> float:
    return float(evaluate_checkpoint(checkpoint, split_name, img_size, k)["p@k"])


def compare(checkpoint: Path, img_size: int = 128, k: int = 10, out: Optional[Path] = None) -> Dict[str, object]:
    """Trained encoder vs ImageNet-pretrained baseline on the EuroSAT test split."""
    ckpt_arch = "resnet18"
    try:
        import torch

        ckpt_arch = torch.load(checkpoint, map_location="cpu").get("arch", "resnet18")
    except Exception:
        pass
    baseline = evaluate_checkpoint(None, "test", img_size, k, arch=ckpt_arch)
    ours = evaluate_checkpoint(checkpoint, "test", img_size, k, arch=ckpt_arch)
    report = {"baseline_imagenet": baseline, "seco": ours}
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1))
    return report
