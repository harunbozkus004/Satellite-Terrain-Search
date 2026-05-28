"""ResNet encoder, MoCo-v2 wrapper used for SeCo-style training, and loaders."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def resnet_backbone(arch: str = "resnet18", pretrained: bool = True) -> nn.Module:
    """ResNet with the classifier removed; returns pooled features (B, D)."""
    weights = None
    if pretrained:
        weights = {"resnet18": "IMAGENET1K_V1", "resnet50": "IMAGENET1K_V2"}[arch]
    net = getattr(torchvision.models, arch)(weights=weights)
    net.fc = nn.Identity()
    return net


def feature_dim(arch: str) -> int:
    return {"resnet18": 512, "resnet50": 2048}[arch]


class Encoder(nn.Module):
    """Backbone + L2 normalisation. This is what retrieval / change detection use."""

    def __init__(self, arch: str = "resnet18", pretrained: bool = True):
        super().__init__()
        self.arch = arch
        self.backbone = resnet_backbone(arch, pretrained)

    @property
    def dim(self) -> int:
        return feature_dim(self.arch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.backbone(x), dim=1)

    def feature_map(self, x: torch.Tensor, layer: str = "layer3") -> torch.Tensor:
        """Per-cell L2-normalised features (B, C, h, w) from an intermediate stage.

        Used for spatial change detection: a small new building changes a few
        cells a lot but barely moves the globally pooled embedding.
        """
        b = self.backbone
        x = b.maxpool(b.relu(b.bn1(b.conv1(x))))
        for name in ("layer1", "layer2", "layer3", "layer4"):
            x = getattr(b, name)(x)
            if name == layer:
                break
        return F.normalize(x, dim=1)


def _mlp(d_in: int, d_hidden: int, d_out: int) -> nn.Module:
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(inplace=True), nn.Linear(d_hidden, d_out))


class MoCo(nn.Module):
    """MoCo-v2 (He et al. 2020) as used by SeCo (Mañas et al. 2021).

    The SeCo twist lives in the data: query and key are crops of the SAME
    location on DIFFERENT dates, so the encoder learns features that are stable
    under seasonal appearance change but still discriminate between places.
    """

    def __init__(self, arch: str = "resnet18", pretrained: bool = True, proj_dim: int = 128,
                 queue_size: int = 4096, momentum: float = 0.99, temperature: float = 0.2):
        super().__init__()
        d = feature_dim(arch)
        self.encoder_q = nn.Sequential(resnet_backbone(arch, pretrained), _mlp(d, d, proj_dim))
        self.encoder_k = copy.deepcopy(self.encoder_q)
        for p in self.encoder_k.parameters():
            p.requires_grad = False
        self.m = momentum
        self.t = temperature
        self.register_buffer("queue", F.normalize(torch.randn(queue_size, proj_dim), dim=1))
        self.register_buffer("queue_ptr", torch.zeros((), dtype=torch.long))

    @property
    def backbone(self) -> nn.Module:
        return self.encoder_q[0]

    @torch.no_grad()
    def _momentum_update(self) -> None:
        for pq, pk in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            pk.data.mul_(self.m).add_(pq.data, alpha=1.0 - self.m)

    @torch.no_grad()
    def _enqueue(self, k: torch.Tensor) -> None:
        n = k.shape[0]
        size = self.queue.shape[0]
        ptr = int(self.queue_ptr)
        idx = (torch.arange(n, device=k.device) + ptr) % size
        self.queue[idx] = k
        self.queue_ptr.fill_((ptr + n) % size)

    def forward(self, x_q: torch.Tensor, x_k: torch.Tensor):
        q = F.normalize(self.encoder_q(x_q), dim=1)
        with torch.no_grad():
            self._momentum_update()
            k = F.normalize(self.encoder_k(x_k), dim=1)
        l_pos = (q * k).sum(dim=1, keepdim=True)
        l_neg = q @ self.queue.clone().detach().T
        logits = torch.cat([l_pos, l_neg], dim=1) / self.t
        labels = torch.zeros(q.shape[0], dtype=torch.long, device=q.device)
        loss = F.cross_entropy(logits, labels)
        acc = (logits.argmax(dim=1) == 0).float().mean()
        self._enqueue(k)
        return loss, acc


def load_encoder(checkpoint: Optional[Path], arch: str = "resnet18") -> Encoder:
    """`checkpoint=None` -> ImageNet-pretrained baseline; otherwise our trained weights."""
    if checkpoint is None:
        return Encoder(arch, pretrained=True).eval()
    ckpt = torch.load(checkpoint, map_location="cpu")
    enc = Encoder(ckpt.get("arch", arch), pretrained=False)
    enc.backbone.load_state_dict(ckpt["backbone"])
    return enc.eval()
