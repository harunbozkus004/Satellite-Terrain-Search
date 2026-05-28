## Results

### Terrain retrieval - EuroSAT test split (5400 queries vs 21600 indexed)

| Encoder (ResNet-18, 128 px input) | precision@10 | precision@1 |
|---|---|---|
| ImageNet-pretrained baseline | 84.7% | 88.2% |
| SeCo-style temporal contrastive (ours, 3 seeds) | **89.3% ± 0.2** | 91.5% ± 0.1 |

Per-class precision@10 (seed 0):

| Class | ImageNet | SeCo |
|---|---|---|
| AnnualCrop | 85.0 | 93.5 |
| Forest | 96.5 | 96.7 |
| HerbaceousVegetation | 82.1 | 81.6 |
| Highway | 63.2 | 78.8 |
| Industrial | 91.0 | 95.1 |
| Pasture | 86.2 | 89.1 |
| PermanentCrop | 84.4 | 80.5 |
| Residential | 91.8 | 95.6 |
| River | 69.8 | 78.6 |
| SeaLake | 92.3 | 98.0 |

### Change detection

OSCD: 1466 changed (>=5% px) / 136 heavily changed (>=30% px) / 3796 unchanged tiles. Seasonal null: 5978 held-out same-place/different-date pairs. Thresholds = 99% quantile of each detector's score on 23851 calibration pairs.

| Detector | OSCD AUROC (>=5%) | OSCD AUROC (>=30%) | Change vs. season AUROC | Recall (>=30%) @ thr | False alarms, seasonal | False alarms, OSCD unchanged |
|---|---|---|---|---|---|---|
| Pixel difference | 0.682 | 0.812 | 0.694 | 0.7% | 1.0% | 0.0% |
| ImageNet, global drift | 0.601 | 0.758 | 0.493 | 0.0% | 1.1% | 0.0% |
| ImageNet, spatial drift | 0.755 | 0.912 | 0.774 | 10.3% | 1.0% | 0.0% |
| SeCo, global drift | 0.600 | 0.813 | 0.808 | 5.1% | 0.9% | 0.9% |
| **SeCo, spatial drift (served)** | 0.738 | 0.910 | 0.884 | 7.4% | 1.1% | 0.2% |

Operating points (threshold = quantile of the seasonal null; recall on OSCD tiles with >=30% changed pixels / false alarms on held-out seasonal pairs):

| Detector | 99% quantile: recall / false alarms | 95% quantile: recall / false alarms |
|---|---|---|
| Pixel difference | 0.7% / 1.0% | 2.9% / 5.0% |
| ImageNet, global drift | 0.0% / 1.1% | 2.2% / 5.6% |
| ImageNet, spatial drift | 10.3% / 1.0% | 21.3% / 4.8% |
| SeCo, global drift | 5.1% / 0.9% | 27.2% / 4.8% |
| **SeCo, spatial drift (served)** | 7.4% / 1.1% | 36.0% / 4.9% |
