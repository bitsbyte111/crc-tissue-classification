# Running the project on a laptop (CPU, no NVIDIA GPU)

Designed for a 4-core Core i7 with 16 GB RAM. Everything runs from one command and finishes in one working day.

## The capstone profile (default, about 1–2 h on a 4-core laptop)

| Part | What runs | Data used |
|---|---|---|
| **Baseline CNN study** (`configs/ablation/capstone_cnn.yaml`) | 6 hyperparameters, 2–3 settings each, 12 runs: gradient-descent variant (SGD / momentum / Adam), learning rate (×0.1 / ×1 / ×10), batch size (32 / 128 / 512), activation (ReLU / tanh / sigmoid), filter size (3×3 / 5×5), dropout (0 / 0.25 / 0.5) | 30 % of training data, 8 epochs; full validation set |
| **Two lightweight pretrained models** (`configs/ablation/capstone_transfer.yaml`) | **MobileNetV3-Small** (1.5 M parameters) and **EfficientNet-B0** (4.0 M), both ImageNet-pretrained, 28 px patches up-sampled to 64 px. Each answers two questions (4 runs): freeze the backbone (feature extraction) or fine-tune it, and which learning rate | 15 % of training data, 3 epochs; full validation set |
| **Final comparison** (`configs/suites/capstone_benchmark.yaml`) | Baseline CNN before and after tuning (all training data, 15 epochs) vs the two tuned pretrained models (half the training data, 4 epochs) | Test cohort: all 7,180 patches from 50 unseen patients, 95 % CIs, paired tests vs the tuned CNN |
| **Explanations** | SHAP (pixel and H&E stain space), Grad-CAM, for the best model and the tuned CNN | Test patches |

Every hyperparameter is chosen on validation data. The test cohort is used once, for the final comparison.

The profiles `--profile standard` and `--profile quick` run the extensive version: 18 CNN hyperparameters, a ResNet-18 transfer study and a DeiT-Tiny transformer. The rest of this guide applies to all profiles.

## What changes compared with the GPU version

| | Kaggle / GPU | Laptop CPU |
|---|---|---|
| Images | 28 px (scratch) and 224 px (pretrained) | **28 px only** (pretrained models get them up-sampled to 64–112 px inside the model) |
| Study 1: CNN, 18 stages | 20 % … 100 % of training data | 20 % of the training split (≈18,000 patches), 8 epochs per setting |
| Study 2: transfer learning | ResNet-50 / DeiT-S at 224 px | **ResNet-18** at 64 px on 10 % of training data, 3 epochs; adds an *input-resolution* stage |
| Study 4: final models | 12 models, 3 seeds | MLP, CNN (default + tuned) on **all** training data; ResNet-18 (default + tuned) and DeiT-Tiny on half; 1 seed |
| Validation / test | full | **full** (10,004 / 7,180 patches); all reported numbers are on complete sets |

Measured on one server core: the small CNN trains at ~900 images/s, ResNet-18 at 64 px at ~84 images/s, ResNet-50 at 224 px at far below that. A 4-core laptop is typically 2.5–4× faster. That is why the laptop plan uses 28 px data, small pretrained models and training subsets for the searches.

## Setup (30 min, Windows)

1. **Python 3.10–3.12** from python.org; tick *Add python.exe to PATH* during installation.
2. In VS Code, open the project folder and a terminal (**Ctrl + `**). Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
   (Mac/Linux: `source .venv/bin/activate`.)
3. Install the CPU build of PyTorch first (~200 MB instead of ~2.5 GB), then the project:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   pip install -e ".[dev]"
   ```
4. Check it works: `pytest -q` should end with all tests passed.
5. **Power settings.** Plug the laptop in. Set *Settings → System → Power → Screen and sleep → When plugged in, put my device to sleep after* → **Never**. A sleeping laptop stops training.

## The data

Put your downloaded PathMNIST file(s) in one folder, e.g. `C:\Users\you\Downloads\pathmnist`.
- MedMNIST `.npz` files work under any name.
- So do folders of `train_images.npy`, `train_labels.npy`, …
- If you downloaded a Kaggle `archive.zip`, unzip it first.

Only the **28 px** file is needed. If no usable file is found, the official 28 px file (~200 MB) is downloaded from Zenodo automatically.

## Run

```bash
# 1. how long will it take on THIS laptop? (2 minutes)
python scripts/run_local.py --data-dir "C:/Users/you/Downloads/pathmnist" --phases check estimate

# 2. run the capstone profile (default)
python scripts/run_local.py --data-dir "C:/Users/you/Downloads/pathmnist"
```

Use forward slashes in the path, and quotes if it contains spaces.

**Stopping and resuming:** press **Ctrl + C** at any time and run the same command again. Finished runs are skipped, and an interrupted run restarts from its beginning.

**Running one phase at a time:** `--phases cnn`, `--phases benchmark explain summary`, and so on. Phases run in this order: `check eda cnn transfer benchmark explain summary`. The benchmark needs the two studies first, because it uses their tuned settings.

**Keeping the laptop usable meanwhile:** `--threads 3` leaves one core free (training is slower).

## What you get

| File | Content |
|---|---|
| `reports/PRESENTATION_SUMMARY.md` | Everything in one page: data figures, the winner of every stage, the test-cohort table with CIs and paired tests, SHAP |
| `reports/presentation/*.png` | All figures, ready to drop into slides |
| `runs/capstone_cnn/OBSERVATIONS.md` | Baseline CNN study: hypothesis, measurements, automatic observations and space for your interpretation, for each of the 6 hyperparameters |
| `runs/capstone_transfer_mobilenetv3_small/OBSERVATIONS.md`, `runs/capstone_transfer_efficientnet_b0/OBSERVATIONS.md` | The same for each pretrained model's two questions |
| `reports/capstone_benchmark/REPORT.md` | Final models on 50 unseen patients: accuracy, macro F1/AUC with 95 % CIs, calibration, per-class F1, paired tests |
| `runs/capstone_benchmark/<model>/seed42/explain/` | SHAP (pixel and H&E stain space), Grad-CAM, sanity check |
| `configs/models/*_tuned*.yaml` | The winning settings, written by the studies |
| `runs/phase_times.json` | How long each phase took (worth one line in the presentation) |

## Suggested day plan

| Time | Task |
|---|---|
| 08:30 | Setup, `pytest`, `--phases check estimate` |
| 09:00 | Start the full run. Then outline the slides (motivation, data, methods) while it trains. |
| ~10:30 | `reports/eda/` is ready: data slides |
| ~12:00 | Study 1 is done: write the interpretations in `OBSERVATIONS.md` |
| afternoon | Study 2 and the benchmark finish: results slides |
| evening | SHAP and the summary: final slides and a rehearsal |

The times depend on your laptop's measured speed. When presenting, describe these as **CPU-budget runs**: searches on training subsets, one seed. The full GPU configuration is in the repository (`docs/KAGGLE.md`).
