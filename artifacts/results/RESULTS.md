## Results

### Terrain retrieval - EuroSAT test split (5400 queries vs 21600 indexed)

| Encoder (ResNet-18, 128 px input) | precision@10 | precision@1 |
|---|---|---|
| ImageNet-pretrained baseline | 84.7% | 88.2% |
| SeCo-style temporal contrastive (ours, 3 seeds) | **89.4% ± 0.3** | 91.6% ± 0.1 |

Per-class precision@10 (seed 0):

| Class | ImageNet | SeCo |
|---|---|---|
| AnnualCrop | 85.0 | 93.0 |
| Forest | 96.5 | 97.1 |
| HerbaceousVegetation | 82.1 | 82.2 |
| Highway | 63.2 | 76.6 |
| Industrial | 91.0 | 95.4 |
| Pasture | 86.2 | 88.8 |
| PermanentCrop | 84.4 | 79.7 |
| Residential | 91.8 | 96.6 |
| River | 69.8 | 78.6 |
| SeaLake | 92.3 | 98.6 |

### Change detection

OSCD: 1466 changed (>=5% px) / 136 heavily changed (>=30% px) / 3796 unchanged tiles. Seasonal null: 6000 held-out same-place/different-date pairs. Thresholds = 99% quantile of each detector's score on 23892 calibration pairs.

| Detector | OSCD AUROC (>=5%) | OSCD AUROC (>=30%) | Change vs. season AUROC | Recall (>=30%) @ thr | False alarms, seasonal | False alarms, OSCD unchanged |
|---|---|---|---|---|---|---|
| Pixel difference | 0.682 | 0.812 | 0.688 | 0.7% | 1.2% | 0.0% |
| ImageNet, global drift | 0.601 | 0.758 | 0.487 | 0.0% | 1.0% | 0.0% |
| ImageNet, spatial drift | 0.755 | 0.912 | 0.760 | 10.3% | 0.9% | 0.0% |
| SeCo, global drift | 0.603 | 0.821 | 0.809 | 8.1% | 1.1% | 1.3% |
| **SeCo, spatial drift (served)** | 0.742 | 0.912 | 0.877 | 6.6% | 0.7% | 0.2% |

Operating points (threshold = quantile of the seasonal null; recall on OSCD tiles with >=30% changed pixels / false alarms on held-out seasonal pairs):

| Detector | 99% quantile: recall / false alarms | 95% quantile: recall / false alarms |
|---|---|---|
| Pixel difference | 0.7% / 1.2% | 2.9% / 5.1% |
| ImageNet, global drift | 0.0% / 1.0% | 2.2% / 5.4% |
| ImageNet, spatial drift | 10.3% / 0.9% | 22.8% / 5.3% |
| SeCo, global drift | 8.1% / 1.1% | 24.3% / 5.4% |
| **SeCo, spatial drift (served)** | 6.6% / 0.7% | 33.8% / 5.4% |
