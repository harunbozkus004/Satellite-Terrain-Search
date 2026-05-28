"""Command-line entry points.  python -m satterrain.cli <command> --help"""
from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path

from satterrain.config import ARTIFACTS_DIR, TIMESERIES_DIR, ensure_dirs


def cmd_download_eurosat(a):
    from satterrain.data.eurosat import download_eurosat, list_samples

    download_eurosat()
    print(f"EuroSAT ready: {len(list_samples())} images")


def cmd_download_oscd(a):
    from satterrain.data.oscd import download

    download()
    print("OSCD ready")


def cmd_fetch(a):
    from satterrain.data.sentinel import build_timeseries

    years = list(range(a.start_year, a.end_year + 1))
    print(build_timeseries(a.locations, n_dates=a.dates, years=years, workers=a.workers, seed=a.seed))


def cmd_harmonize(a):
    import numpy as np
    from PIL import Image

    from satterrain.data import eurosat, harmonize

    rng = np.random.RandomState(0)
    src = np.load(TIMESERIES_DIR / "timeseries.npz")["images"]
    src = src[rng.choice(len(src), min(len(src), 5000), replace=False)]
    samples = eurosat.list_samples()
    tgt = np.stack([np.asarray(Image.open(samples[i][0]).convert("RGB"))
                    for i in rng.choice(len(samples), 5000, replace=False)])
    print(json.dumps(harmonize.fit(src, tgt), indent=1))


def cmd_train(a):
    from satterrain.train import train

    extra = None
    if a.extra_eurosat:
        from satterrain.data import eurosat

        train_split, _ = eurosat.split(eurosat.list_samples())
        extra = [p for p, _ in train_split]
    print(train(out_dir=Path(a.out), arch=a.arch, init=a.init, epochs=a.epochs, batch_size=a.batch_size,
                lr=a.lr, img_size=a.img_size, queue_size=a.queue_size, temperature=a.temperature,
                workers=a.workers, extra_image_paths=extra, eval_every=a.eval_every, seed=a.seed))


def cmd_evaluate(a):
    from satterrain.evaluate import compare

    rep = compare(Path(a.checkpoint), img_size=a.img_size, k=a.k, out=Path(a.out))
    for name, r in rep.items():
        print(f"{name:20s} precision@{r['k']} = {r['p@k']:.4f}   p@1 = {r['p@1']:.4f}")


def cmd_build_index(a):
    from satterrain.index import build

    print(build(Path(a.checkpoint) if a.checkpoint else None, out_dir=Path(a.out), img_size=a.img_size,
                kind=a.kind, quantile=a.quantile))


def cmd_evaluate_change(a):
    from satterrain.evaluate_change import compare

    rep = compare(Path(a.checkpoint), img_size=a.img_size, quantile=a.quantile, out=Path(a.out))
    print(f"{'detector':26s} {'OSCD>=5%':>9s} {'OSCD>=30%':>9s} {'vsSeason':>9s} "
          f"{'recall':>7s} {'FA-seas':>7s} {'FA-oscd':>7s}")
    for name in (k for k in rep if not k.startswith("_")):
        r = rep[name]
        print(f"{name:26s} {r['oscd_auroc_ge5']:9.3f} {r['oscd_auroc_ge30']:9.3f} {r['vs_seasonal_auroc']:9.3f} "
              f"{r['recall_on_oscd_ge30']:7.3f} {r['false_alarm_rate_seasonal_heldout']:7.3f} "
              f"{r['false_alarm_rate_oscd_unchanged']:7.3f}")


def cmd_report(a):
    """Aggregate artifacts/results/*.json into a Markdown results table."""
    import glob

    import numpy as np

    res_dir = Path(a.results)
    runs = [json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(res_dir / "retrieval_seed*.json")))]
    lines = ["## Results", ""]
    if runs:
        base = runs[0]["baseline_imagenet"]
        p10 = np.array([r["seco"]["p@k"] for r in runs]) * 100
        p1 = np.array([r["seco"]["p@1"] for r in runs]) * 100
        lines += [
            f"### Terrain retrieval - EuroSAT test split ({base['n_queries']} queries vs {base['n_index']} indexed)",
            "",
            "| Encoder (ResNet-18, 128 px input) | precision@10 | precision@1 |",
            "|---|---|---|",
            f"| ImageNet-pretrained baseline | {base['p@k'] * 100:.1f}% | {base['p@1'] * 100:.1f}% |",
            f"| SeCo-style temporal contrastive (ours, {len(runs)} seeds) | **{p10.mean():.1f}% ± {p10.std():.1f}** "
            f"| {p1.mean():.1f}% ± {p1.std():.1f} |",
            "",
            "Per-class precision@10 (seed 0):",
            "",
            "| Class | ImageNet | SeCo |",
            "|---|---|---|",
        ]
        for c, v in runs[0]["seco"]["per_class"].items():
            lines.append(f"| {c} | {base['per_class'][c] * 100:.1f} | {v * 100:.1f} |")
        lines.append("")
    ch = res_dir / "change.json"
    if ch.exists():
        r = json.loads(ch.read_text())
        n = r["_counts"]
        lines += [
            "### Change detection",
            "",
            f"OSCD: {n['oscd_changed_ge5']} changed (>=5% px) / {n['oscd_changed_ge30']} heavily changed (>=30% px) / "
            f"{n['oscd_unchanged']} unchanged tiles. Seasonal null: {n['seasonal_heldout_pairs']} held-out "
            f"same-place/different-date pairs. Thresholds = {n['quantile']:.0%} quantile of each detector's "
            f"score on {n['seasonal_cal_pairs']} calibration pairs.",
            "",
            "| Detector | OSCD AUROC (>=5%) | OSCD AUROC (>=30%) | Change vs. season AUROC | Recall (>=30%) @ thr | "
            "False alarms, seasonal | False alarms, OSCD unchanged |",
            "|---|---|---|---|---|---|---|",
        ]
        names = {"pixel_difference": "Pixel difference", "baseline_imagenet": "ImageNet, global drift",
                 "baseline_imagenet_spatial": "ImageNet, spatial drift", "seco": "SeCo, global drift",
                 "seco_spatial": "**SeCo, spatial drift (served)**"}
        for k, label in names.items():
            if k in r:
                x = r[k]
                lines.append(f"| {label} | {x['oscd_auroc_ge5']:.3f} | {x['oscd_auroc_ge30']:.3f} | "
                             f"{x['vs_seasonal_auroc']:.3f} | {x['recall_on_oscd_ge30']:.1%} | "
                             f"{x['false_alarm_rate_seasonal_heldout']:.1%} | {x['false_alarm_rate_oscd_unchanged']:.1%} |")
        lines.append("")
        q95 = res_dir / "change_q95.json"
        if q95.exists():
            r2 = json.loads(q95.read_text())
            lines += [f"Operating points (threshold = quantile of the seasonal null; recall on OSCD tiles with "
                      f">=30% changed pixels / false alarms on held-out seasonal pairs):", "",
                      "| Detector | 99% quantile: recall / false alarms | 95% quantile: recall / false alarms |",
                      "|---|---|---|"]
            for k, label in names.items():
                if k in r and k in r2:
                    lines.append(f"| {label} | {r[k]['recall_on_oscd_ge30']:.1%} / "
                                 f"{r[k]['false_alarm_rate_seasonal_heldout']:.1%} | "
                                 f"{r2[k]['recall_on_oscd_ge30']:.1%} / {r2[k]['false_alarm_rate_seasonal_heldout']:.1%} |")
            lines.append("")
    out = res_dir / "RESULTS.md"
    out.write_text("\n".join(lines))
    print(out.read_text())


def cmd_serve(a):
    import os

    import uvicorn

    os.environ["SATTERRAIN_INDEX_DIR"] = a.index
    uvicorn.run("satterrain.api:app", host=a.host, port=a.port)


def main(argv=None):
    warnings.filterwarnings("ignore", message=".*OpenSSL.*")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("rasterio").setLevel(logging.WARNING)
    ensure_dirs()
    p = argparse.ArgumentParser(prog="satterrain")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("download-eurosat").set_defaults(fn=cmd_download_eurosat)
    sub.add_parser("download-oscd").set_defaults(fn=cmd_download_oscd)

    s = sub.add_parser("fetch", help="fetch multi-date Sentinel-2 tiles from Planetary Computer")
    s.add_argument("--locations", type=int, default=3000)
    s.add_argument("--dates", type=int, default=5)
    s.add_argument("--start-year", type=int, default=2019)
    s.add_argument("--end-year", type=int, default=2023)
    s.add_argument("--workers", type=int, default=16)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_fetch)

    sub.add_parser("harmonize", help="fit L2A->EuroSAT colour harmonisation").set_defaults(fn=cmd_harmonize)

    s = sub.add_parser("train", help="SeCo-style contrastive pretraining")
    s.add_argument("--out", default=str(ARTIFACTS_DIR / "seco"))
    s.add_argument("--arch", default="resnet18", choices=["resnet18", "resnet50"])
    s.add_argument("--init", default="imagenet", choices=["imagenet", "random"])
    s.add_argument("--epochs", type=int, default=100)
    s.add_argument("--batch-size", type=int, default=128)
    s.add_argument("--lr", type=float, default=0.03)
    s.add_argument("--img-size", type=int, default=128)
    s.add_argument("--queue-size", type=int, default=4096)
    s.add_argument("--temperature", type=float, default=0.2)
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--eval-every", type=int, default=10)
    s.add_argument("--extra-eurosat", action="store_true",
                   help="also use unlabeled EuroSAT train images as single-date positive pairs")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("evaluate", help="EuroSAT precision@k vs ImageNet baseline")
    s.add_argument("--checkpoint", default=str(ARTIFACTS_DIR / "seco" / "encoder.pt"))
    s.add_argument("--img-size", type=int, default=128)
    s.add_argument("--k", type=int, default=10)
    s.add_argument("--out", default=str(ARTIFACTS_DIR / "results" / "retrieval.json"))
    s.set_defaults(fn=cmd_evaluate)

    s = sub.add_parser("evaluate-change", help="change detection on OSCD")
    s.add_argument("--checkpoint", default=str(ARTIFACTS_DIR / "seco" / "encoder.pt"))
    s.add_argument("--img-size", type=int, default=128)
    s.add_argument("--quantile", type=float, default=0.99)
    s.add_argument("--out", default=str(ARTIFACTS_DIR / "results" / "change.json"))
    s.set_defaults(fn=cmd_evaluate_change)

    s = sub.add_parser("build-index", help="embed catalogue into FAISS + calibrate change threshold")
    s.add_argument("--checkpoint", default=str(ARTIFACTS_DIR / "seco" / "encoder.pt"),
                   help="empty string -> ImageNet baseline")
    s.add_argument("--out", default=str(ARTIFACTS_DIR / "index"))
    s.add_argument("--img-size", type=int, default=128)
    s.add_argument("--kind", default="flat", choices=["flat", "hnsw"])
    s.add_argument("--quantile", type=float, default=0.99)
    s.set_defaults(fn=cmd_build_index)

    s = sub.add_parser("report", help="write artifacts/results/RESULTS.md from result JSONs")
    s.add_argument("--results", default=str(ARTIFACTS_DIR / "results"))
    s.set_defaults(fn=cmd_report)

    s = sub.add_parser("serve")
    s.add_argument("--index", default=str(ARTIFACTS_DIR / "index"))
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_serve)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
