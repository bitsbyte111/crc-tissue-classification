#!/usr/bin/env python
"""Run the project on a laptop / desktop CPU.

    python scripts/run_local.py --data-dir "C:/Users/me/Downloads/pathmnist" --phases check estimate
    python scripts/run_local.py --data-dir "C:/Users/me/Downloads/pathmnist"            # capstone run

Profiles:
  capstone  (default) one baseline CNN tuned over 6 hyperparameters (12 runs) + two lightweight
            pretrained models (MobileNetV3-Small, EfficientNet-B0), each with a 2-question transfer study
  standard  the extensive version: 18 CNN stages, ResNet-18 transfer study (5 stages), 6 final models
  quick     the extensive version on smaller budgets

Phases (in order; default = all except estimate):
  check      find + validate the data (28 px), cache it in ./data, report hardware
  estimate   measure training speed on THIS machine and predict each phase's runtime
  eda        class balance, example patches, colour shift train vs test -> reports/eda
  cnn        baseline CNN hyperparameter study (validation data only)
  transfer   transfer-learning study for the pretrained model(s) (validation data only)
  benchmark  final models on the 7,180-patch test cohort, with 95 % CIs and paired tests
  explain    SHAP (pixel + H&E stain space) and Grad-CAM for the best model and the tuned CNN
  summary    reports/PRESENTATION_SUMMARY.md + reports/presentation/ (figures for the slides)

Everything is resumable: stop with Ctrl+C at any time and run the same command again -
finished runs are skipped.
"""
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
PHASES = ["check", "estimate", "eda", "cnn", "transfer", "benchmark", "explain", "summary"]
QUICK_CNN_STAGES = ["optimizer", "learning_rate", "batch_size", "activation", "kernel_size", "width", "depth",
                    "dropout", "augmentation"]
QUICK_TRANSFER_STAGES = ["finetune_strategy", "learning_rate", "input_resolution"]

# Each profile: which plans/suite to run and with what budget overrides. Budgets only change how much
# training data / how many epochs the searches use; validation and test sets are always complete.
PROFILES = {
    "capstone": {
        "cnn_plan": "capstone_cnn.yaml", "cnn_stages": None, "cnn": [],
        "transfer_plan": "capstone_transfer.yaml", "transfer_stages": None, "transfer": [],
        "transfer_models": ["l3_mobilenetv3_small_pt", "l4_efficientnet_b0_pt"],
        "suite": "capstone_benchmark.yaml",
        "bench_groups": [(None, [])],
    },
    "standard": {
        "cnn_plan": "cnn_ablation_cpu.yaml", "cnn_stages": None, "cnn": [],
        "transfer_plan": "transfer_ablation_cpu.yaml", "transfer_stages": None, "transfer": [],
        "transfer_models": ["l1_resnet18_pt"],
        "suite": "local_cpu_benchmark.yaml",
        "bench_groups": [(["B0_mlp", "B1_cnn", "B1_cnn_tuned"], []),
                         (["L1_resnet18_pt", "L1_resnet18_tuned", "L2_deit_tiny_pt"], [])],
    },
    "quick": {
        "cnn_plan": "cnn_ablation_cpu.yaml", "cnn_stages": QUICK_CNN_STAGES,
        "cnn": ["data.subset_fraction=0.1", "train.epochs=5"],
        "transfer_plan": "transfer_ablation_cpu.yaml", "transfer_stages": QUICK_TRANSFER_STAGES,
        "transfer": ["data.subset_fraction=0.05", "train.epochs=2"],
        "transfer_models": ["l1_resnet18_pt"],
        "suite": "local_cpu_benchmark.yaml",
        "bench_groups": [(["B0_mlp", "B1_cnn", "B1_cnn_tuned"], ["train.epochs=8"]),
                         (["L1_resnet18_pt", "L1_resnet18_tuned", "L2_deit_tiny_pt"],
                          ["data.subset_fraction=0.25", "train.epochs=2"])],
    },
}


# Lightweight pretrained models available to the capstone profile: --models name -> (config, final-run id)
PRETRAINED_CHOICES = {
    "mobilenetv3_small": ("l3_mobilenetv3_small_pt", "MobileNetV3_small_pretrained"),
    "efficientnet_b0": ("l4_efficientnet_b0_pt", "EfficientNet_B0_pretrained"),
    "resnet18": ("l1_resnet18_pt", "ResNet18_pretrained"),
}


def _plan_name(plan_file: str) -> str:
    from crc_tissuenet.ablation import load_plan

    return load_plan(ROOT / "configs/ablation" / plan_file)["name"]


def _suite_name(suite_file: str) -> str:
    from crc_tissuenet.suite import load_suite

    return load_suite(ROOT / "configs/suites" / suite_file)["name"]


def _transfer_dir(prof: dict, model: str) -> Path:
    name = _plan_name(prof["transfer_plan"])
    tag = model.split("_", 1)[1].removesuffix("_pt")  # l3_mobilenetv3_small_pt -> mobilenetv3_small
    return ROOT / "runs" / (name if prof["cnn_plan"] != "capstone_cnn.yaml" else f"{name}_{tag}")


def log(msg: str) -> None:
    print(f"\n{'=' * 78}\n{time.strftime('%H:%M:%S')} | {msg}\n{'=' * 78}", flush=True)


def run(args: list[str], extra_env: dict | None = None) -> int:
    cmd = [sys.executable, *args]
    print(">>", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT, env={**os.environ, **(extra_env or {})}).returncode


# ----------------------------------------------------------------------------- phases
def phase_check(a) -> int:
    import torch

    from crc_tissuenet.data import ARRAY_KEYS, find_sources, prepare_data

    print(f"Python {platform.python_version()} | torch {torch.__version__} | CPU threads {torch.get_num_threads()} "
          f"| CUDA GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none (CPU training)'}")
    sources = [a.data_dir] if a.data_dir else []
    for f in find_sources([ROOT / "data", *sources]):
        print(f"  found {f['size']}px {f['kind']}: {f['counts']} {f['path']} "
              f"[{'OK' if f['complete'] and f['official_counts'] else f['problems'] or 'non-official sizes'}]")
    folder = prepare_data(ROOT / "data", 28, download=True, source_dirs=sources)
    cache = ROOT / "data" / "pathmnist_28_mmap"
    if folder.resolve() != cache.resolve():  # unpacked .npy folder found elsewhere: copy (0.3 GB) into ./data
        cache.mkdir(parents=True, exist_ok=True)
        for k in ARRAY_KEYS:
            shutil.copy2(folder / f"{k}.npy", cache / f"{k}.npy")
    print(f"28 px data ready in {cache}")
    return 0


def _speed(model, size, train=True, bs=64, steps=6):
    import torch

    x = torch.rand(bs, 3, size, size)
    y = torch.randint(0, 9, (bs,))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], 1e-4)
    model.train(train)
    t0 = None
    for i in range(steps + 2):
        if i == 2:
            t0 = time.perf_counter()
        if train:
            loss = torch.nn.functional.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        else:
            with torch.no_grad():
                model(x)
    return bs * steps / (time.perf_counter() - t0)


def phase_estimate(a) -> int:
    """Measure images/second for each model on this machine and predict runtimes per phase."""
    import timm

    from crc_tissuenet.ablation import load_plan
    from crc_tissuenet.config import load_config
    from crc_tissuenet.models import MLP, ConfigurableCNN
    from crc_tissuenet.models.pretrained import TIMM_ALIASES
    from crc_tissuenet.suite import load_suite

    prof = PROFILES[a.profile]
    n_train, n_val, n_test = 89_996, 10_004, 7_180
    print("Measuring training speed on this machine (1-3 minutes)...")
    cnn_tr = _speed(ConfigurableCNN(28, 9), 28, bs=128)
    cnn_ev = _speed(ConfigurableCNN(28, 9), 28, train=False, bs=256)
    cnn_big = _speed(ConfigurableCNN(28, 9, kernel_size=5), 28, bs=128, steps=4)
    mlp_tr = _speed(MLP(28, 9), 28, bs=128)
    speeds = {"cnn": (cnn_tr, cnn_ev), "mlp": (mlp_tr, cnn_ev)}

    def pretrained_speed(cfg_file):
        cfg = load_config(ROOT / "configs/models" / f"{cfg_file}.yaml")
        name, size = cfg["model"]["name"], cfg["model"].get("input_size") or 64
        if name not in speeds:
            kw = (cfg["model"].get("params") or {}).get("timm_kwargs") or {}
            m = timm.create_model(TIMM_ALIASES.get(name, name).split(".")[0], num_classes=9, **kw)
            speeds[name] = (_speed(m, size, steps=3), _speed(m, size, train=False, bs=128, steps=3))
        return speeds[name]

    def opt(lst, key, default):
        for o in lst:
            if o.startswith(key + "="):
                return float(o.split("=")[1])
        return default

    def study(plan_file, stages, overrides, speed, mix=1.0):
        plan = load_plan(ROOT / "configs/ablation" / plan_file)
        st = [s for s in plan["stages"] if not stages or s["name"] in stages]
        runs = sum(len(s["candidates"]) for s in st) - len(st) + 1  # identical incumbents are reused
        frac = opt(overrides, "data.subset_fraction", plan["common_overrides"]["data.subset_fraction"])
        ep = opt(overrides, "train.epochs", plan["common_overrides"]["train.epochs"])
        return runs, runs * ep * (n_train * frac / speed[0] + n_val / speed[1]) * mix, plan

    if a.profile == "capstone":  # measure every lightweight candidate so the choice is informed
        print("\nLightweight pretrained candidates on this machine (train images/second, higher = faster):")
        for key, (cfg_file, _) in PRETRAINED_CHOICES.items():
            tr, ev = pretrained_speed(cfg_file)
            chosen = "  <- selected" if cfg_file in prof["transfer_models"] else ""
            print(f"  {key:<18} train {tr:6.0f} | evaluate {ev:6.0f}{chosen}")
        print("  Choose with:  --models <name> <name>   (e.g. --models mobilenetv3_small resnet18)\n")
    rows = []
    mix = 1 + 0.4 * (cnn_tr / cnn_big - 1) / 2  # some candidates (bigger filters/networks) are slower
    runs, sec, cnn_plan = study(prof["cnn_plan"], prof["cnn_stages"], prof["cnn"], speeds["cnn"], mix)
    rows.append(("baseline CNN study", runs, sec))
    tplan = None
    for m in prof["transfer_models"]:
        runs, sec, tplan = study(prof["transfer_plan"], prof["transfer_stages"], prof["transfer"],
                                 pretrained_speed(m), 1.3 if a.profile != "capstone" else 1.0)
        rows.append((f"transfer study: {m}", runs, sec))
    suite = load_suite(ROOT / "configs/suites" / prof["suite"])
    bench = 0.0
    n_bench = 0
    for ids, ovs in prof["bench_groups"]:
        for run in suite["runs"]:
            if ids and run["id"] not in ids:
                continue
            n_bench += 1
            cfg_path = (suite["_dir"] / run["config"]).resolve()
            final = {}
            if not cfg_path.exists():  # tuned config written by a study: estimate from its base config
                stem = cfg_path.stem.split("_tuned")[0]
                cfg_path = cfg_path.with_name(stem + ".yaml")
                final = (cnn_plan if stem.startswith("b1_cnn") else tplan or {}).get("final_overrides", {})
            cfg = load_config(cfg_path, {**final, **run.get("overrides", {})})
            name = cfg["model"]["name"]
            sp = speeds["mlp" if name == "mlp" else "cnn"] if name in ("mlp", "simple_cnn") else \
                pretrained_speed(cfg_path.stem)
            frac = opt(ovs, "data.subset_fraction", cfg["data"].get("subset_fraction", 1.0))
            ep = opt(ovs, "train.epochs", cfg["train"]["epochs"])
            bench += ep * (n_train * frac / sp[0] + n_val / sp[1]) * (mix if name == "simple_cnn" else 1) + n_test / sp[1]
    rows.append(("final comparison (test)", n_bench, bench))
    rows.append(("eda + explain + summary", "-", 12 * 60))
    total = sum(r[2] for r in rows)
    print("images/second (train | evaluate): " + ", ".join(f"{k} {v[0]:.0f}|{v[1]:.0f}" for k, v in speeds.items()))
    print(f"\nEstimated runtime on this machine, profile '{a.profile}' (rough: +/- 40 %, upper bound if early "
          "stopping triggers):")
    for name, n, sec in rows:
        print(f"  {name:<44} runs {str(n):>3}   ~{sec / 3600:4.1f} h")
    print(f"  {'TOTAL':<44}            ~{total / 3600:4.1f} h")
    with open(ROOT / "runs" / "time_estimate.json", "w", encoding="utf-8") as f:
        json.dump({"speed": speeds, "profile": a.profile, "estimate_h": {r[0]: r[2] / 3600 for r in rows}}, f, indent=2)
    return 0


def phase_eda(a) -> int:
    return run(["scripts/eda.py", "--platform", "local_cpu", "--out", "reports/eda"])


def phase_cnn(a) -> int:
    p = PROFILES[a.profile]
    stages = ["--stages", *p["cnn_stages"]] if p["cnn_stages"] else []
    return run(["scripts/run_ablation.py", "--plan", f"configs/ablation/{p['cnn_plan']}", *stages,
                "--gpus", "none", "--platform", "local_cpu", "-o", *p["cnn"], *a.override])


def phase_transfer(a) -> int:
    p = PROFILES[a.profile]
    stages = ["--stages", *p["transfer_stages"]] if p["transfer_stages"] else []
    rc = 0
    for m in p["transfer_models"]:
        print(f"\n--- transfer study: {m}")
        rc |= run(["scripts/run_ablation.py", "--plan", f"configs/ablation/{p['transfer_plan']}", *stages,
                   "--base", f"configs/models/{m}.yaml", "--output-dir", str(_transfer_dir(p, m).relative_to(ROOT)),
                   "--best-config-out", f"configs/models/{m}_tuned.yaml",
                   "--gpus", "none", "--platform", "local_cpu", "-o", *p["transfer"], *a.override])
    return rc


def phase_benchmark(a) -> int:
    p = PROFILES[a.profile]
    suite = f"configs/suites/{p['suite']}"
    rc = 0
    for ids, ovs in p["bench_groups"]:
        only = ["--only", *ids] if ids else []
        rc |= run(["scripts/run_suite.py", "--suite", suite, *only, "--gpus", "none",
                   "--platform", "local_cpu", "-o", *ovs, *a.override])
    rc |= run(["scripts/report.py", "--suite", suite])
    return rc


def phase_explain(a) -> int:
    import pandas as pd

    name = _suite_name(PROFILES[a.profile]["suite"])
    summary = ROOT / "reports" / name / "summary.csv"
    if not summary.exists():
        print("No final results yet - run the benchmark phase first.")
        return 1
    s = pd.read_csv(summary).sort_values("test_macro_f1_mean", ascending=False)
    baseline = next((i for i in s.id if "tuned" in i.lower() and "cnn" in i.lower()), None)
    targets = list(dict.fromkeys([s.id.iloc[0]] + ([baseline] if baseline else [])))
    rc = 0
    for rid in targets:
        rd = f"runs/{name}/{rid}/seed42"
        rc |= run(["scripts/explain.py", "--run", rd, "--per-class", "5", "--nsamples", "100"])
        rc |= run(["scripts/explain.py", "--run", rd, "--per-class", "2", "--stain-space", "--no-sanity",
                   "--nsamples", "100"])
    return rc


def phase_summary(a) -> int:
    import pandas as pd

    rep = ROOT / "reports"
    pres = rep / "presentation"
    pres.mkdir(parents=True, exist_ok=True)
    lines = ["# CRC-TissueNet - results summary (laptop CPU run)", "",
             f"Generated {time.strftime('%Y-%m-%d %H:%M')} | profile `{a.profile}`. Hyperparameters were selected on the "
             "validation split only; the test cohort (7,180 patches from 50 unseen patients) was used once per final "
             "model. Studies used parts of the training data (see each study's settings); validation and test sets "
             "were always complete.", ""]
    picks = []

    def pick(src: Path, name: str):
        if src.exists():
            shutil.copy2(src, pres / name)
            picks.append(name)
            return f"![{name}](presentation/{name})"
        return ""

    lines += ["## 1. Data", ""]
    for f in ("class_distribution.png", "example_patches.png", "colour_shift_train_vs_test.png"):
        lines.append(pick(rep / "eda" / f, f"eda_{f}"))
    prof = PROFILES[a.profile]
    studies = [("2. Baseline CNN - hyperparameter study", ROOT / "runs" / _plan_name(prof["cnn_plan"]))]
    studies += [(f"3. Transfer learning - {m}", _transfer_dir(prof, m)) for m in prof["transfer_models"]]
    suite_name = _suite_name(prof["suite"])
    for title, d in studies:
        lines += ["", f"## {title}", ""]
        if (d / "ablation_results.csv").exists():
            rows = pd.read_csv(d / "ablation_results.csv")
            sel = rows[rows.selected.astype(str) == "True"].drop_duplicates("stage")
            lines += ["| stage | selected setting | val macro F1 |", "|---|---|---|"]
            lines += [f"| {r.stage} | {r.candidate} | {float(r.val_macro_f1):.4f} |" for r in sel.itertuples()]
            lines += ["", pick(d / "figures/progress.png", f"{d.name}_progress.png"), ""]
            for st in sel.stage:
                lines.append(pick(d / "figures" / f"stage_{st}.png", f"{d.name}_stage_{st}.png"))
            lines += ["", f"Full observations: `{d.relative_to(ROOT)}/OBSERVATIONS.md`"]
        else:
            lines.append("_Not run yet._")
    lines += ["", "## 4. Final models on the unseen test cohort", ""]
    bench = rep / suite_name
    if (bench / "REPORT.md").exists():
        body = (bench / "REPORT.md").read_text(encoding="utf-8").split("\n", 2)[-1]
        lines += [body.replace("](figures/", f"]({suite_name}/figures/"), ""]
        for f in ("accuracy_vs_params.png", "confusion_best.png", "reliability_best.png"):
            pick(bench / "figures" / f, f"benchmark_{f}")
    else:
        lines.append("_Not run yet._")
    lines += ["", "## 5. Explainability (SHAP)", ""]
    for rd in sorted((ROOT / "runs" / suite_name).glob("*/seed42/explain")):
        rid = rd.parent.parent.name
        for f in sorted(rd.glob("*.png")):
            lines.append(pick(f, f"shap_{rid}_{f.name}"))
        san = rd / "sanity_check.json"
        if san.exists():
            with open(san, encoding="utf-8") as fh:
                rho = json.load(fh)["spearman_mean"]
            lines.append(f"{rid}: randomisation sanity check, Spearman = {rho:.3f} "
                         "(low = explanations depend on the learned weights)")
    (rep / "PRESENTATION_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Summary: {rep / 'PRESENTATION_SUMMARY.md'} | {len(picks)} figures in {pres}")
    return 0


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phases", nargs="+", default=[p for p in PHASES if p != "estimate"], choices=PHASES)
    ap.add_argument("--profile", default="capstone", choices=sorted(PROFILES))
    ap.add_argument("--data-dir", default=None, help="folder containing your downloaded PathMNIST files")
    ap.add_argument("--threads", type=int, default=None, help="CPU threads for PyTorch (default: all cores)")
    ap.add_argument("--models", nargs="+", default=["mobilenetv3_small", "efficientnet_b0"],
                    choices=sorted(PRETRAINED_CHOICES),
                    help="capstone profile: lightweight pretrained models to study and compare")
    ap.add_argument("-o", "--override", nargs="*", default=[], help="extra overrides for every run (advanced)")
    a = ap.parse_args()
    os.chdir(ROOT)
    if a.profile == "capstone":
        PROFILES["capstone"]["transfer_models"] = [PRETRAINED_CHOICES[m][0] for m in a.models]
        PROFILES["capstone"]["bench_groups"] = [(["CNN_baseline_default", "CNN_baseline_tuned"]
                                                 + [PRETRAINED_CHOICES[m][1] for m in a.models], [])]
    if a.threads:
        os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = str(a.threads)
        import torch

        torch.set_num_threads(a.threads)
    (ROOT / "runs").mkdir(exist_ok=True)
    times_file = ROOT / "runs" / "phase_times.json"
    times = json.load(open(times_file, encoding="utf-8")) if times_file.exists() else {}
    failed = []
    for ph in [p for p in PHASES if p in a.phases]:
        log(f"PHASE {ph}  (profile {a.profile})")
        t0 = time.perf_counter()
        rc = globals()[f"phase_{ph}"](a)
        times[ph] = round((time.perf_counter() - t0) / 60, 1)
        json.dump(times, open(times_file, "w", encoding="utf-8"), indent=2)
        print(f"-- phase {ph} finished in {times[ph]} min (exit code {rc})")
        if rc:
            failed.append(ph)
    log("DONE" + (f" - phases with errors: {', '.join(failed)} (re-run the same command to retry)" if failed else ""))
    print("Open reports/PRESENTATION_SUMMARY.md for the results.")


if __name__ == "__main__":
    main()
