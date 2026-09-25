# Age Estimation from Facial Images

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C)
![License](https://img.shields.io/badge/License-MIT-green)

Predicting a person's age directly from a facial photograph, framed as a
computer-vision regression problem. The project compares a frozen-backbone
baseline against a fully fine-tuned model with face-detection preprocessing,
benchmarks a classification reframing of the same task, and includes a
Grad-CAM interpretability analysis of what the model actually looks at.

This is a standalone Data Science / ML Engineering portfolio project —
distinct from my physics research (GNN-based phonon dispersion prediction,
in a separate repository).

---

## Problem & Motivation

Age estimation from a single image is a classic but non-trivial computer
vision problem: age-related visual cues (skin texture, facial structure)
vary continuously and are confounded by pose, lighting, expression, and
demographic factors. It's a good testbed for demonstrating a full ML
workflow — EDA, preprocessing, baseline-vs-improved modeling, metric-driven
evaluation, and interpretability — on a task that resists a "throw a
model at it with defaults" shortcut.

## Dataset

- **Source:** [UTKFace](https://www.kaggle.com/datasets/jangedoo/utkface-new) (Kaggle, `jangedoo/utkface-new`)
- **Size:** 23,705 valid labeled face images (a handful of files with malformed
  filenames were dropped during parsing)
- **Labels:** encoded in the filename as `[age]_[gender]_[race]_[timestamp].jpg`
  - Age: continuous, 1–116 years (mean ≈ 33, heavily skewed toward younger adults)
  - Gender: binary (0 = male, 1 = female)
  - Race: 5 categories (White, Black, Asian, Indian, Other)
- **Target variable:** age (used as a continuous regression target; also
  re-binned into 8 age groups for the classification comparison)
- **Split:** 80% train / 10% validation / 10% test, stratified on a coarse
  age bucket so all three splits have a similar age distribution

## Methodology

### 1. Exploratory Data Analysis
- Age, gender, and race distributions (see `notebooks/01_data_eda_baseline.ipynb`)
- Visual sanity-check grid of sample images across the age range

### 2. Baseline model
- ResNet18 (ImageNet-pretrained), **backbone frozen**, small regression head
  trained on raw (unaligned) 128×128 crops
- Trained with L1 loss (MAE) directly, early stopping on validation MAE

### 3. Improved model (fine-tuning + face detection preprocessing)
- **Face detection & cropping:** OpenCV Haar Cascade detects the largest
  face in each image and crops with a margin before resizing, reducing
  background/pose noise. Chosen over a landmark-based aligner (e.g. MTCNN)
  because it ships with `opencv-python` (already present in the target
  environment) and introduces no additional, potentially conflicting
  dependencies — a deliberate engineering tradeoff, not an oversight.
- **Full backbone fine-tuning:** all ResNet18 layers unfrozen, trained with
  a lower learning rate on the backbone (1e-4) than the regression head
  (1e-3), plus stronger augmentation (horizontal flip, color jitter,
  rotation)

### 4. Classification reframing
- Age re-binned into 8 groups (0–10, 11–20, ..., 71+)
- Same backbone, trained as a classifier with class-weighted cross-entropy
  to counter the imbalance across bins
- Evaluated with accuracy, macro-F1, and a confusion matrix; also converted
  back to an implied age (bin midpoint) to get a comparable MAE against the
  regression model

### 5. Interpretability
- Grad-CAM on the fine-tuned regression model's final convolutional layer,
  to visualize which facial regions drive the age prediction

## Results

| Model | Test MAE (years) | Notes |
|---|---|---|
| Baseline: frozen ResNet18, no face detection | **9.68** | Notebook 01 |
| Fine-tuned ResNet18 + face-detected crops | **4.95** | Notebook 02 — full backbone unfrozen |
| Classification (8 age bins), bin-midpoint implied age | **6.29** | Test accuracy 0.579, macro-F1 0.550 |

**Fine-tuning the backbone together with face-detection preprocessing cut
test MAE from 9.68 to 4.95 years** (a 49% reduction), and clearly
outperformed the classification reframing of the same task on the same
data split. An MAE of ~5 years on UTKFace with a lightweight ResNet18
backbone is in a competitive range for this dataset without relying on a
heavier architecture or ensembling.

Classification per-class performance (see `notebooks/02_finetune_alignment_gradcam.ipynb`
for the full report and confusion matrix): the model handled well-represented
bins (0–10, 21–30) noticeably better than sparse bins (61–70, 71+), which is
consistent with the training-set class imbalance despite class weighting.

*Grad-CAM visualizations are saved in `results/gradcam_examples.png` — see
the Limitations section below for what they show.*

## Key Decisions & Tradeoffs

- **MAE-as-loss vs. a two-stage loss:** L1 loss was used directly during
  training rather than MSE, since MAE is also the evaluation metric and is
  less sensitive to the handful of extreme-age outliers (e.g. age 100+) in
  the dataset.
- **Backbone choice (ResNet18):** chosen for a fast, GPU-modest training
  loop appropriate for a single-person, 1–2 week project, rather than a
  larger backbone (ResNet50/EfficientNet) that would likely improve MAE
  further at a meaningfully higher compute/time cost.
- **Face detector choice (Haar Cascade over MTCNN):** MTCNN
  (`facenet-pytorch`) was tried first but its dependency pins conflicted
  with the training environment's numpy/torchvision versions badly enough
  to break unrelated imports. Haar Cascade was substituted as a
  zero-additional-dependency alternative; it lacks landmark-based
  alignment but was sufficient to materially improve MAE (see Results).
  This tradeoff — and the debugging process behind it — is documented
  because reasoning about dependency conflicts is itself part of the ML
  engineering work, not just the modeling.
- **Class weighting for the classification variant:** inverse-frequency
  class weights were applied in the cross-entropy loss to counter the
  dataset's skew toward younger age bins, rather than oversampling or
  discarding majority-class samples.

## Limitations

- **Dataset bias:** UTKFace is skewed toward younger adults and toward the
  "White" race category (see the EDA plots in Notebook 01 for exact
  counts). Reported metrics are aggregate; performance is not guaranteed
  to be uniform across age, gender, or race subgroups, and this project
  does not include a subgroup-disaggregated fairness evaluation.
- **Face detector limitations:** Haar Cascade is a simpler, older detector
  than modern deep-learning-based face detectors; a small fraction of
  images fall back to an unaligned center crop when no face is detected.
- **Single train/val/test split:** results are reported from one stratified
  split rather than k-fold cross-validation, appropriate for this project's
  scope but meaning reported MAE has some inherent variance not captured
  by a single point estimate.
- **No demographic-stratified interpretability check:** Grad-CAM was
  inspected on a small random sample rather than systematically across
  race/gender subgroups.

## Tech Stack

- Python 3.10, PyTorch + torchvision, OpenCV, scikit-learn, pandas/numpy,
  matplotlib, `grad-cam`
- Training run on Kaggle (T4 GPU)

## How to Reproduce

### On Kaggle (recommended — this is how the reported results were produced)
1. Create a new Kaggle notebook, add the dataset `jangedoo/utkface-new` as
   an input, enable GPU (T4).
2. Upload and run `notebooks/01_data_eda_baseline.ipynb` top to bottom.
3. Upload and run `notebooks/02_finetune_alignment_gradcam.ipynb` top to
   bottom (it re-derives the same split; no artifacts need to be carried
   over from notebook 01).

### Locally
```bash
git clone <this-repo-url>
cd age-estimation
pip install -r environment/requirements.txt
# Download UTKFace manually from Kaggle and point DATA_DIR in the notebooks
# at your local copy, then run the notebooks as above.
```

## Repository Structure

See [`docs/REPO_STRUCTURE.md`](docs/REPO_STRUCTURE.md) for the full file tree.

## License

MIT — see [`LICENSE`](LICENSE).
