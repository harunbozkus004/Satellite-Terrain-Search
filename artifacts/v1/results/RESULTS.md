## Results

### Terrain retrieval - EuroSAT test split (5400 queries vs 21600 indexed)

| Encoder (ResNet-18, 128 px input) | precision@10 | precision@1 |
|---|---|---|
| ImageNet-pretrained baseline | 84.7% | 88.2% |
| SeCo-style temporal contrastive (ours, 3 seeds) | **89.3% ± 0.2** | 91.5% ± 0.3 |

Per-class precision@10 (seed 0):

| Class | ImageNet | SeCo |
|---|---|---|
| AnnualCrop | 85.0 | 93.2 |
| Forest | 96.5 | 97.1 |
| HerbaceousVegetation | 82.1 | 82.1 |
| Highway | 63.2 | 78.2 |
| Industrial | 91.0 | 94.1 |
| Pasture | 86.2 | 88.8 |
| PermanentCrop | 84.4 | 80.3 |
| Residential | 91.8 | 96.6 |
| River | 69.8 | 79.5 |
| SeaLake | 92.3 | 98.3 |

### Change detection

OSCD: 1466 changed (>=5% px) / 136 heavily changed (>=30% px) / 3796 unchanged tiles. Seasonal null: 6000 held-out same-place/different-date pairs. Thresholds = 99% quantile of each detector's score on 23902 calibration pairs.

| Detector | OSCD AUROC (>=5%) | OSCD AUROC (>=30%) | Change vs. season AUROC | Recall (>=30%) @ thr | False alarms, seasonal | False alarms, OSCD unchanged |
|---|---|---|---|---|---|---|
| Pixel difference | 0.682 | 0.812 | 0.711 | 0.0% | 1.1% | 0.0% |
| ImageNet, global drift | 0.601 | 0.758 | 0.503 | 0.0% | 1.3% | 0.0% |
| ImageNet, spatial drift | 0.755 | 0.912 | 0.750 | 7.4% | 1.1% | 0.0% |
| SeCo, global drift | 0.591 | 0.803 | 0.728 | 2.2% | 1.1% | 0.2% |
| **SeCo, spatial drift (served)** | 0.732 | 0.911 | 0.850 | 5.9% | 1.2% | 0.0% |
