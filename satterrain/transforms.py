"""Image transforms shared by training, evaluation and serving."""
from __future__ import annotations

import random

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as TF

from satterrain.config import IMAGENET_MEAN, IMAGENET_STD


class RandomRot90:
    """Overhead imagery has no canonical 'up', so 90-degree rotations are free."""

    def __call__(self, img):
        return TF.rotate(img, 90 * random.randint(0, 3))


def eval_transform(img_size: int = 128):
    return T.Compose([
        T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def train_transform(img_size: int = 128):
    # Colour jitter is deliberately mild: seasonal appearance change is supplied
    # by the temporal positive pairs themselves.
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.4, 1.0), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        RandomRot90(),
        T.RandomApply([T.ColorJitter(0.3, 0.3, 0.2, 0.05)], p=0.8),
        T.RandomGrayscale(p=0.1),
        T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.3),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.asarray(arr, dtype=np.uint8))


@torch.no_grad()
def embed_arrays(encoder, images: np.ndarray, img_size: int, device, batch_size: int = 256) -> np.ndarray:
    """Embed a (N, H, W, 3) uint8 array -> (N, D) float32, L2-normalised."""
    tf = eval_transform(img_size)
    out = []
    encoder = encoder.to(device).eval()
    for i in range(0, len(images), batch_size):
        x = torch.stack([tf(to_pil(a)) for a in images[i:i + batch_size]]).to(device)
        out.append(encoder(x).float().cpu())
    if not out:
        return np.zeros((0, encoder.dim), np.float32)
    return torch.cat(out).numpy().astype(np.float32)


@torch.no_grad()
def spatial_drift(encoder, a: np.ndarray, b: np.ndarray, img_size: int, device, layer: str = "layer3",
                  top_frac: float = 0.1, batch_size: int = 128) -> np.ndarray:
    """Per-cell cosine distance between feature maps of paired tiles, summarised
    as the mean of the top `top_frac` most-changed cells. (N,) float."""
    tf = eval_transform(img_size)
    encoder = encoder.to(device).eval()
    out = []
    for i in range(0, len(a), batch_size):
        xa = torch.stack([tf(to_pil(x)) for x in a[i:i + batch_size]]).to(device)
        xb = torch.stack([tf(to_pil(x)) for x in b[i:i + batch_size]]).to(device)
        d = 1.0 - (encoder.feature_map(xa, layer) * encoder.feature_map(xb, layer)).sum(1)  # (B, h, w)
        d = d.flatten(1)
        k = max(1, int(round(d.shape[1] * top_frac)))
        out.append(d.topk(k, dim=1).values.mean(1).float().cpu())
    return torch.cat(out).numpy() if out else np.zeros(0, np.float32)
