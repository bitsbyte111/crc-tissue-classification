#!/usr/bin/env python
"""Exploratory figures for the report: class balance, example patches, colour shift train vs test.

    python scripts/eda.py --platform local_cpu --out reports/eda
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import _bootstrap  # noqa: F401,E402

from crc_tissuenet.config import platform_overrides  # noqa: E402
from crc_tissuenet.data import CLASS_DESCRIPTIONS, CLASS_NAMES, SPLITS, load_split, prepare_data  # noqa: E402
from crc_tissuenet.utils import get_logger, save_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--platform", default="local_cpu")
    ap.add_argument("--size", type=int, default=28)
    ap.add_argument("--out", default="reports/eda")
    args = ap.parse_args()
    plat = platform_overrides(args.platform)
    folder = prepare_data(plat.get("data.root", "data"), args.size, download=bool(plat.get("data.download", False)),
                          source_dirs=plat.get("data.source_dirs"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    arrays = {s: load_split(folder, s, in_memory=False) for s in SPLITS}

    counts = pd.DataFrame({s: np.bincount(a.labels, minlength=9) for s, a in arrays.items()}, index=CLASS_NAMES)
    counts.to_csv(out / "class_counts.csv")
    fig, ax = plt.subplots(figsize=(9, 3.8))
    (counts / counts.sum()).plot.bar(ax=ax)
    ax.set(title="Class proportions per split", ylabel="share of split")
    fig.tight_layout()
    fig.savefig(out / "class_distribution.png", dpi=150)
    plt.close(fig)

    rng = np.random.default_rng(0)
    train = arrays["train"]
    fig, axes = plt.subplots(3, 9, figsize=(15, 5.6))
    for c, name in enumerate(CLASS_NAMES):
        for r, i in enumerate(rng.choice(np.flatnonzero(train.labels == c), 3, replace=False)):
            axes[r, c].imshow(train.images[i], interpolation="nearest")
            axes[r, c].axis("off")
        axes[0, c].set_title(f"{name}\n{CLASS_DESCRIPTIONS[name]}", fontsize=8)
    fig.suptitle(f"Example training patches ({args.size} px)")
    fig.tight_layout()
    fig.savefig(out / "example_patches.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.8))
    stats = {}
    for split, color in (("train", "tab:blue"), ("test", "tab:red")):
        a = arrays[split]
        x = np.asarray(a.images[np.sort(rng.choice(len(a), min(3000, len(a)), replace=False))], float) / 255
        stats[split] = {"mean_rgb": x.mean((0, 1, 2)).round(4).tolist(), "std_rgb": x.std((0, 1, 2)).round(4).tolist()}
        for c, ls in zip(range(3), ("-", "--", ":")):
            ax.hist(x[..., c].ravel(), bins=60, histtype="step", density=True, color=color, ls=ls,
                    label=f"{split} {'RGB'[c]}")
    ax.set(title="Pixel intensities: training slides vs unseen test patients", xlabel="intensity", ylabel="density")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "colour_shift_train_vs_test.png", dpi=150)
    plt.close(fig)
    save_json({"counts": counts.to_dict(), "colour": stats}, out / "eda_summary.json")
    get_logger().info("EDA figures written to %s", out)


if __name__ == "__main__":
    main()
