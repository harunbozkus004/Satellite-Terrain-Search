"""Fast unit tests (no network, no datasets)."""
import numpy as np
import torch

import satterrain  # noqa: F401  (sets macOS OpenMP workaround)
from satterrain.change import calibrate, drift, pairwise_location_drifts
from satterrain.data.sentinel import _offset, to_uint8
from satterrain.evaluate import precision_at_k
from satterrain.models import MoCo


def _unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_drift_is_cosine_distance():
    a = _unit(np.array([[1.0, 0.0], [1.0, 1.0]]))
    b = _unit(np.array([[1.0, 0.0], [-1.0, -1.0]]))
    np.testing.assert_allclose(drift(a, b), [0.0, 2.0], atol=1e-6)


def test_precision_at_k_perfect_clusters():
    rng = np.random.RandomState(0)
    centers = _unit(rng.randn(3, 16))
    labels = np.repeat(np.arange(3), 20)
    emb = _unit(centers[labels] + 0.01 * rng.randn(60, 16)).astype(np.float32)
    r = precision_at_k(emb, labels, emb[::5], labels[::5], k=5)
    assert r["p@k"] == 1.0 and r["p@1"] == 1.0


def test_calibration_controls_false_positives():
    rng = np.random.RandomState(0)
    n_locs, n_dates = 400, 4
    base = _unit(rng.randn(n_locs, 32))
    emb = _unit(np.repeat(base, n_dates, 0) + 0.15 * rng.randn(n_locs * n_dates, 32))
    loc_ids = np.repeat(np.arange(n_locs), n_dates)
    cal = calibrate(emb, loc_ids, quantile=0.95)
    assert abs(cal.heldout_false_positive_rate - 0.05) < 0.03
    # A genuinely different place should be flagged.
    assert cal.flag(drift(base[0], base[1]))[0]
    assert len(pairwise_location_drifts(emb, loc_ids)) == n_locs * 6


def test_reflectance_scaling_and_baseline_offset():
    class Item:
        properties = {"s2:processing_baseline": "05.09"}

    assert _offset(Item()) == 1000.0
    Item.properties = {"s2:processing_baseline": "02.14"}
    assert _offset(Item()) == 0.0
    out = to_uint8(np.array([0, 1000, 1500, 4000, 9000]), offset=1000)
    assert out.tolist() == [0, 0, 42, 255, 255]


def test_spatial_drift_localised_change():
    from satterrain.models import Encoder
    from satterrain.transforms import spatial_drift

    rng = np.random.RandomState(0)
    a = (rng.rand(2, 64, 64, 3) * 255).astype(np.uint8)
    b = a.copy()
    b[1, :16, :16] = 255  # paint a small "new building" in one corner of the second pair
    enc = Encoder("resnet18", pretrained=False)
    d = spatial_drift(enc, a, b, img_size=64, device=torch.device("cpu"))
    assert d[0] < 1e-4 and d[1] > d[0]


def test_scl_mask_rules():
    from satterrain.data.sentinel import scl_rejects

    veg = np.full((32, 32), 4, np.uint8)
    roofs = veg.copy(); roofs[:9] = 9            # ~28% "cloud" = bright factory roofs
    assert not scl_rejects(roofs, scene_cloud_pct=0.1)[0]   # clear scene: trust it's roofs
    assert scl_rejects(roofs, scene_cloud_pct=20.0)[0]      # cloudy scene: it's cloud
    city = veg.copy(); city[:7] = 2               # ~22% dark area = building shadows
    assert not scl_rejects(city, scene_cloud_pct=0.1)[0]
    winter = veg.copy(); winter[:20] = 2          # ~62% dark = low-sun terrain shadow
    assert scl_rejects(winter, scene_cloud_pct=0.1)[0]
    snow = veg.copy(); snow[:4] = 11              # snow is rejected even in clear scenes
    assert scl_rejects(snow, scene_cloud_pct=0.1)[0]


def test_moco_forward_backward():
    m = MoCo("resnet18", pretrained=False, queue_size=16)
    x = torch.randn(4, 3, 64, 64)
    loss, acc = m(x, x.clone())
    loss.backward()
    assert torch.isfinite(loss) and 0 <= acc.item() <= 1
    assert int(m.queue_ptr) == 4
