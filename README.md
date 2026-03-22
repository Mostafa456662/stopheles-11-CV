# Siamese Image Comparator
 
A deep learning pipeline for image similarity comparison using a Siamese Network built on top of MobileNetV2, with automated segmentation preprocessing via SAM (Segment Anything Model).
 
---
 
## Overview
 
This project implements a two-phase Siamese Network that learns to determine whether two images are similar (positive pair) or dissimilar (negative pair). It is designed around a custom database of positive (`P`) and negative (`N`) image classes, with an optional segmentation stage that pre-processes raw images using SAM before training.
 
---
 
## Architecture
 
```
Input Pair (img1, img2)
        │
        ▼
 Embedding Model f(·)          ← MobileNetV2 (pretrained on ImageNet)
  f(img1) │ f(img2)
        │
   Concatenate
        │
        ▼
  Siamese Head g(·)
  Dense(128, relu) → Dropout(0.5) → BatchNorm → Dense(1, sigmoid)
        │
        ▼
   Similarity Score ∈ [0, 1]
```
 
Training follows a two-phase strategy:
 
1. **Phase 1 — Head training**: MobileNetV2 weights are frozen; only the siamese head is trained.
2. **Phase 2 — Fine-tuning**: All weights are unfrozen and the full model is trained end-to-end at a lower learning rate.
 
---
 
## Project Structure
 
```
.
├── axl/
│   └── requirements.txt
├── dbs/
│   ├── comparator_db/
│   │   └── raw/
│   │       ├── N/              # Negative class images
│   │       └── P/              # Positive class images
│   └── segmentor_db/
│       ├── raw/                # Raw input images for segmentation
│       └── cooked/             # SAM-generated .npz mask files
├── docs/
│   ├── Siamese_Network[1].pdf
│   ├── Vision_Transformer[1].pdf
│   ├── SAM[1].pdf
│   └── ...
└── src/
    ├── data/
    │   ├── siamese_data_pipeline.py
    │   └── display_masks.py
    ├── SAM.py
    ├── Siamese_helper.py
    └── siamese.py
```
 
---
 
## Data Pipeline
 
Defined in `src/data/siamese_data_pipeline.py`.
 
| Step | Description |
|------|-------------|
| **Load** | Reads images from `N/` and `P/` subdirectories, resizes to target shape, normalizes to `[0, 1]` |
| **Augment** | Applies random flip, rotation, zoom, translation, brightness, contrast, and Gaussian noise — repeated `n` times |
| **Pair generation** | Creates all positive pairs `(P_i, P_j)` and all negative pairs `(N_i, P_j)` with corresponding binary labels |
| **Split** | Shuffles and splits into train/test sets, with a retry loop that enforces a minimum positive-to-negative ratio |
 
---
 
## Segmentation (SAM)
 
Defined in `src/SAM.py`. Uses [Ultralytics SAM](https://docs.ultralytics.com/models/sam/) to generate instance segmentation masks for raw images.
 
- Input: images in `dbs/segmentor_db/raw/`
- Output: `.npz` files (shape `[num_masks, H, W]`) saved to `dbs/segmentor_db/cooked/`
 
Masks can be visually inspected using `src/data/display_masks.py`, which overlays colored masks on the original images using OpenCV.
 
---
 
## Installation
 
```bash
git clone <repo-url>
cd <repo-name>
pip install -r axl/requirements.txt
```
 
> Requires Python 3.10+ and a CUDA-capable GPU for reasonable training speed.
 
---
 
## Usage
 
### 1. (Optional) Generate segmentation masks
 
```bash
python src/SAM.py
```
 
### 2. Train the Siamese model
 
```bash
python src/siamese.py
```
 
Key hyperparameters (configured at the bottom of `siamese.py`):
 
| Parameter | Default | Description |
|-----------|---------|-------------|
| `X_shape` | `(224, 224, 3)` | Input image dimensions |
| `n` | `4` | Augmentation multiplier |
| `train_ratio` | `0.8` | Train/test split ratio |
| `P_to_N_ratio` | `0.2` | Minimum positive-to-negative ratio in training set |
| `epochs` | `10` | Total training epochs (split 30/70 between phases) |
 
### 3. Monitor training
 
At the end of each epoch, `PredictionCallback` prints a per-sample prediction table showing true labels, predicted labels, probabilities, and per-sample accuracy.
 
---
 
## Training Details
 
- **Loss**: Binary cross-entropy
- **Optimizer**: Adam with exponential learning rate decay
- **Class weights**: Positives weighted 5× to handle class imbalance
- **Callbacks**: Early stopping, learning rate reduction on plateau, best-model checkpointing to `src/models/model.keras`
 
---
 
## References
 
- [Siamese Neural Networks for One-shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) — Koch et al.
- [Segment Anything Model (SAM)](https://arxiv.org/abs/2304.02643) — Kirillov et al.
- [MobileNetV2](https://arxiv.org/abs/1801.04381) — Sandler et al.
- Additional references in `docs/`
