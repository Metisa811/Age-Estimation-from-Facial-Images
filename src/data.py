"""
Data loading, parsing, and preprocessing utilities for the UTKFace age
estimation project.

Mirrors the logic used interactively in notebooks/01 and notebooks/02, kept
here as reusable, testable functions so the project isn't notebook-only.
"""

import os
import re
import glob

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch


AGE_BINS = [0, 10, 20, 30, 40, 50, 60, 70, 120]
AGE_BIN_LABELS = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71+']
NUM_CLASSES = len(AGE_BIN_LABELS)


def parse_utkface_filename(path):
    """Parse age/gender/race from a UTKFace filename.

    Filenames are formatted as `[age]_[gender]_[race]_[timestamp].jpg`.
    Returns None if the filename doesn't match the expected pattern.
    """
    fname = os.path.basename(path)
    m = re.match(r'^(\d+)_(\d+)_(\d+)_', fname)
    if not m:
        return None
    age, gender, race = map(int, m.groups())
    return {'path': path, 'age': age, 'gender': gender, 'race': race}


def load_utkface_dataframe(data_dir):
    """Scan a UTKFace directory and return a DataFrame of valid labeled images.

    Silently skips files with malformed filenames (a small number in the
    original dataset) rather than raising, since this is expected data noise.
    """
    all_paths = glob.glob(os.path.join(data_dir, '*.jpg'))
    records = [parse_utkface_filename(p) for p in all_paths]
    records = [r for r in records if r is not None]
    df = pd.DataFrame(records)
    df['age_bucket'] = pd.cut(df['age'], bins=AGE_BINS, labels=False)
    return df


def stratified_split(df, seed=42):
    """Split a UTKFace dataframe into 80/10/10 train/val/test, stratified
    on the coarse age bucket so all splits have a similar age distribution.
    """
    train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=df['age_bucket'], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df['age_bucket'], random_state=seed
    )
    return train_df, val_df, test_df


def add_classification_bins(df):
    """Add the 8-class age bucket column used by the classification variant."""
    df = df.copy()
    df['age_class_bucket'] = pd.cut(df['age'], bins=AGE_BINS, labels=False)
    return df


class FaceDetector:
    """Thin wrapper around OpenCV's Haar Cascade frontal-face detector.

    Chosen over a landmark-based aligner (e.g. MTCNN via facenet-pytorch)
    because it ships with opencv-python — already a project dependency —
    and introduces no additional packages that could conflict with the
    training environment's numpy/torch/torchvision versions. See the
    project README's "Key Decisions & Tradeoffs" section for the reasoning.
    """

    def __init__(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise RuntimeError('Failed to load Haar Cascade — check OpenCV installation')

    def detect_and_crop(self, path, margin=0.2, out_size=128):
        """Detect the largest face in an image, crop with a margin, resize.

        Falls back to a plain resize of the original image if no face is
        detected, so no data is discarded due to detector misses.
        """
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            return np.array(Image.open(path).convert('RGB').resize((out_size, out_size)))

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )

        if len(faces) == 0:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            return np.array(Image.fromarray(img_rgb).resize((out_size, out_size)))

        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        mx, my = int(w * margin), int(h * margin)
        x0, y0 = max(0, x - mx), max(0, y - my)
        x1, y1 = min(img_bgr.shape[1], x + w + mx), min(img_bgr.shape[0], y + h + my)

        crop_bgr = img_bgr[y0:y1, x0:x1]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        return np.array(Image.fromarray(crop_rgb).resize((out_size, out_size)))

    def detect_and_cache(self, path, cache_dir, margin=0.2, out_size=128):
        """Same as detect_and_crop, but caches the result to disk and
        returns the cached path instead of an array (used for the
        precompute-once-per-epoch workflow in notebooks/02).
        """
        os.makedirs(cache_dir, exist_ok=True)
        out_path = os.path.join(cache_dir, os.path.basename(path))
        if os.path.exists(out_path):
            return out_path
        cropped = self.detect_and_crop(path, margin=margin, out_size=out_size)
        Image.fromarray(cropped).save(out_path)
        return out_path


class UTKFaceDataset(Dataset):
    """PyTorch Dataset for UTKFace, supporting both regression (raw age)
    and classification (binned age) label modes.
    """

    def __init__(self, dataframe, transform, path_col='path', label_mode='regression'):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.path_col = path_col
        self.label_mode = label_mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        img = Image.open(row[self.path_col]).convert('RGB')
        img = self.transform(img)
        if self.label_mode == 'regression':
            label = torch.tensor(row['age'], dtype=torch.float32)
        else:
            label = torch.tensor(row['age_class_bucket'], dtype=torch.long)
        return img, label
