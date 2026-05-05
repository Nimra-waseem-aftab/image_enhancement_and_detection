# Low-Light Image Enhancement for Robust Object Detection

> **AI335L Deep Learning Lab — Spring 2026 | Phase 2 Submission**
> From Darkness to Detection: Enhancing Visual Clarity for Robust Object Recognition

| Team Member | Registration ID |
|---|---|
| Muhammad Umair Naveed | S2024AI001 |
| Dayan Amjad | S2024AI007 |
| Nimra Waseem | S2024AI002 |

**Instructor:** Sir M. Haseeb | **Submission Tag:** `phase2-submission`

---

## Project Overview

A two-stage deep learning pipeline that addresses performance degradation of object detection systems under low-light conditions:

- **Stage 1:** U-Net CNN performs pixel-wise low-light image enhancement
- **Stage 2:** Pretrained YOLOv8 detector processes enhanced images for object detection

**Datasets:** LOL (485 pairs, primary) · VE-LOL (2,500 pairs, backup)  
**Framework:** PyTorch 2.2.0 · Ultralytics YOLOv8 8.1.x

---

## Repository Structure

```
low-light-enhancement/
├── data/                          # Raw + processed data (gitignored)
│   ├── raw/
│   │   ├── LOL/
│   │   │   ├── train/
│   │   │   │   ├── low/           # Low-light training images
│   │   │   │   └── high/          # Normal-light reference images
│   │   │   └── test/
│   │   │       ├── low/
│   │   │       └── high/
│   │   └── VE-LOL/                # Backup dataset
│   └── processed/                 # Preprocessed tensors (generated)
│       ├── train_data.pt
│       ├── val_data.pt
│       └── test_data.pt
├── notebooks/
│   ├── EDA_LowLight.ipynb         # Full EDA notebook (also exported as HTML)
│   ├── EDA_LowLight.html          # HTML export for submission
│   └── download_data.py           # Dataset download + verification script
├── src/
│   ├── data/
│   │   └── dataset.py             # LOLDataset class, pair loading, splits
│   ├── preprocessing/
│   │   ├── pipeline.py            # Preprocessor class (.fit / .transform)
│   │   └── augment.py             # PairedRandomFlip, PhotometricJitter
│   ├── models/
│   │   ├── baseline_mlp.py        # Baseline MLP (Phase 2)
│   │   ├── unet.py                # U-Net enhancement model (Phase 3)
│   │   └── yolov8_wrapper.py      # YOLOv8 inference wrapper (Phase 3)
│   ├── training/
│   │   └── train.py               # Training loop with early stopping + checkpointing
│   └── utils/
│       ├── seed.py                # seed_everything() utility
│       ├── metrics.py             # PSNR, SSIM computation
│       └── logger.py              # CSV logging helpers
├── experiments/
│   ├── baseline_config.yaml       # Baseline MLP hyperparameters
│   ├── unet_config.yaml           # U-Net hyperparameters (Phase 3)
│   ├── training_history.csv       # Per-epoch logs (auto-generated)
│   └── checkpoints/               # Saved model weights (gitignored)
├── reports/
│   ├── loss_curves.png
│   ├── sample_grid.png
│   ├── test_results.json
│   └── confusion_matrix.png
├── tests/
│   └── test_pipeline.py           # Smoke tests for preprocessing pipeline
├── requirements.txt               # Pinned Python dependencies
├── environment.yml                # Conda environment
├── DATA_CARD.md                   # Dataset documentation
├── README.md                      # This file
└── .gitignore
```

---

## Quickstart — Reproduce the Baseline in 5 Steps

### Prerequisites
- Python 3.10+
- pip or conda
- ~2 GB free disk space for LOL dataset

### Step 1 — Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/low-light-enhancement.git
cd low-light-enhancement
```

### Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

Or using conda:

```bash
conda env create -f environment.yml
conda activate lowlight
```

### Step 3 — Download the Dataset

```bash
# Primary dataset: LOL (485 pairs, ~90 MB)
python notebooks/download_data.py --dataset lol --output data/raw/

# Backup dataset: VE-LOL (2,500 pairs, ~500 MB)
python notebooks/download_data.py --dataset velol --output data/raw/
```

The script verifies download integrity automatically:
```
[VERIFIED] 400 training pairs and 85 test pairs loaded and matched correctly.
```

### Step 4 — Run Preprocessing (~5 minutes, CPU)

```bash
jupyter nbconvert --to notebook --execute notebooks/EDA_LowLight.ipynb
```

This generates `data/processed/train_data.pt`, `val_data.pt`, `test_data.pt`.

### Step 5 — Train the Baseline MLP (~20 minutes, CPU sufficient)

```bash
python src/training/train.py --config experiments/baseline_config.yaml
```

This single command:
- Calls `seed_everything(42)` before anything else
- Loads preprocessed data from `data/processed/`
- Trains the baseline MLP with early stopping (patience=10)
- Saves best checkpoint to `experiments/checkpoints/baseline_mlp.pt`
- Evaluates on the test set **exactly once** at the end
- Writes results to `reports/test_results.json`

Expected output:
```
[Epoch 25/50] Train L1: 0.0855 | Val L1: 0.0849 | Val PSNR: 18.5 dB | Saved checkpoint.
Early stopping triggered at epoch 38.
[TEST] PSNR: 18.2 dB | SSIM: 0.64 | L1: 0.0842
Results saved to reports/test_results.json
```

---

## Pretrained Weights (Skip Training)

Download pretrained baseline weights from Google Drive:

```bash
# Instructions in DATA_CARD.md — link in shared team Drive folder
```

---

## Evaluation Metrics

| Metric | Meaning | Baseline MLP | U-Net Target (Phase 3) |
|---|---|---|---|
| PSNR (dB) | Pixel-level fidelity | 18.2 dB | ≥ 20.0 dB |
| SSIM | Perceptual / structural quality | 0.64 | ≥ 0.75 |
| L1 Loss | Mean absolute pixel error | 0.0842 | Minimise |
| mAP@0.5 | YOLOv8 detection on ExDark | TBD Phase 3 | ≥ 10% improvement |

---

## Reproducing the EDA Notebook

```bash
# Run and export EDA notebook to HTML
jupyter nbconvert --to html --execute notebooks/EDA_LowLight.ipynb --output notebooks/EDA_LowLight.html
```

Open `notebooks/EDA_LowLight.html` in any browser — no Python required.

---

## Running Tests

```bash
pytest tests/ -v
```

Smoke tests verify:
- Preprocessor output shape: `(N, 3, 256, 256)`
- Pixel values in normalised range (mean ≈ 0, std ≈ 1)
- No NaN or Inf in output tensors
- Low and high tensors have identical spatial dimensions

---

## Configuration

All hyperparameters live in `experiments/baseline_config.yaml`. No hardcoded values exist in source code.

```yaml
# experiments/baseline_config.yaml
model:
  hidden_dims: [4096, 2048, 1024]
  dropout: 0.3
  weight_init: kaiming_uniform

training:
  seed: 42
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
  optimizer: adam
  early_stopping_patience: 10
  gradient_clip: 1.0

data:
  patch_size: 64
  image_size: 256
  train_split: 0.70
  val_split: 0.15
  test_split: 0.15
```

---

## Six-Week Project Timeline

| Week | Phase | Key Deliverable | Lead |
|---|---|---|---|
| 1 | Planning | SRS Document, repo skeleton | Umair |
| 2 | **Data & Baseline** | **EDA notebook, preprocessing pipeline, baseline MLP** | **Dayan** |
| 3 | Core Model | U-Net training, YOLOv8 integration tests | Umair |
| 4 | Tuning | Optimizer comparison, Optuna (30 trials) | Nimra |
| 5 | Evaluation | Ablation studies, mAP on ExDark, error analysis | All |
| 6 | Deployment | Streamlit/FastAPI demo, final report | Umair |

---

## Tech Stack

| Component | Tool | Version |
|---|---|---|
| Deep Learning | PyTorch | 2.2.0 |
| Detection Model | Ultralytics YOLOv8 | 8.1.x |
| Image Processing | OpenCV | 4.9.0 |
| Augmentation | albumentations | 1.3.x |
| Metrics | scikit-image | 0.22.0 |
| Hyperparameter Tuning | Optuna | 3.5.0 (Phase 4) |
| Experiment Tracking | TensorBoard | 2.16.0 (Phase 3+) |
| Deployment | Streamlit + FastAPI | 1.32.0 / 0.110.0 (Phase 6) |

---

## Submission

```bash
git tag phase2-submission
git push origin phase2-submission
```

Submit on LMS: PDF report + GitHub URL + commit hash of tagged release.
