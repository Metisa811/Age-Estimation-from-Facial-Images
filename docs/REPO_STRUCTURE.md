# Repository Structure

This is the exact folder/file layout to create in the GitHub repository.
Copy each file into the matching path.

```
age-estimation/
├── README.md
├── LICENSE
├── .gitignore
├── environment/
│   └── requirements.txt
├── notebooks/
│   ├── 01_data_eda_baseline.ipynb
│   └── 02_finetune_alignment_gradcam.ipynb
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── model.py
│   └── train.py
├── results/
│   ├── eda_distributions.png
│   ├── sample_grid.png
│   ├── baseline_training_curve.png
│   ├── baseline_pred_vs_true.png
│   ├── alignment_comparison.png
│   ├── finetune_training_curve.png
│   ├── classification_confusion_matrix.png
│   └── gradcam_examples.png
└── docs/
    └── REPO_STRUCTURE.md
```

## Notes

- **`results/*.png`** — these are the plots your two notebooks already save
  when run on Kaggle (e.g. `eda_distributions.png`, `gradcam_examples.png`,
  etc.). Download them from the Kaggle output panel after running each
  notebook and place them here so the README's embedded images resolve.
- **`environment/requirements.txt`** — used for local reproduction; Kaggle
  already has most of these preinstalled except `grad-cam`, which the
  notebook installs inline.
- **Model checkpoints** (`best_baseline_model.pt`, `best_finetuned_model.pt`,
  `best_classifier_model.pt`) are intentionally **not** committed to the
  repo (see `.gitignore`) — they're a few hundred MB combined, which is
  poor practice for a git repo. If you want them downloadable, attach them
  to a GitHub Release instead and link that release from the README.
- **`src/`** mirrors the notebook logic as importable, reusable modules
  (`data.py`, `model.py`, `train.py`) — this is what signals "software
  engineering practice" rather than "notebook-only" work to anyone
  reviewing the repo.
