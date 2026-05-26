#!/usr/bin/env python3
"""
Train a ResNet50 + XGBoost crop-disease classifier on the PlantVillage dataset.

How it works
------------
  1. Walk the PlantVillage folder (one sub-folder per class).
  2. Load every image, resize to 224×224.
  3. Forward-pass through ResNet50 (ImageNet weights, no classification head).
     Each image → 2048-dim feature vector (GlobalAveragePooling output).
  4. Normalise features with StandardScaler.
  5. Train XGBoostClassifier (multi:softprob, 38 classes).
  6. Evaluate on a held-out test split.
  7. Save:
       model/xgb_model.pkl          ← trained XGBoost
       model/xgb_scaler.pkl         ← fitted StandardScaler
       model/xgb_label_encoder.pkl  ← LabelEncoder (index ↔ class name)

Dataset layout expected
-----------------------
  <data_dir>/
      Apple___Apple_scab/
          img001.jpg
          ...
      Apple___Black_rot/
          ...
      ...   (38 class directories total)

Download: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
          or  https://data.mendeley.com/datasets/tywbtsjrjv/1

Usage
-----
  # Minimal
  python train_resnet50_xgboost.py --data /path/to/PlantVillage

  # Full options
  python train_resnet50_xgboost.py \\
      --data      /path/to/PlantVillage \\
      --output    ./model              \\
      --batch-size 64                  \\
      --n-estimators 300               \\
      --max-depth    6                 \\
      --learning-rate 0.1              \\
      --test-size 0.15

After training set in krishi-backend/.env:
  XGB_MODEL_PATH=./model/xgb_model.pkl
Then restart uvicorn and the backend auto-switches to ResNet50+XGBoost.
"""

import argparse
import logging
import time
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm

import xgboost as xgb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Must match CLASS_NAMES in main.py exactly (same index order)
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


# ─── helpers ──────────────────────────────────────────────────────────────────

def collect_dataset(data_dir: str) -> tuple[list[str], list[str]]:
    """Walk data_dir, return (image_paths, class_name_per_image)."""
    data_path = Path(data_dir)
    if not data_path.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    paths, labels = [], []
    class_dirs = sorted(d for d in data_path.iterdir() if d.is_dir())
    log.info(f"Found {len(class_dirs)} class directories in {data_dir}")

    for cls_dir in class_dirs:
        class_name = cls_dir.name
        if class_name not in CLASS_NAMES:
            log.warning(f"  Skipping unknown class directory: {class_name}")
            continue
        imgs = (
            list(cls_dir.glob("*.jpg"))
            + list(cls_dir.glob("*.JPG"))
            + list(cls_dir.glob("*.jpeg"))
            + list(cls_dir.glob("*.png"))
            + list(cls_dir.glob("*.PNG"))
        )
        log.info(f"  {class_name:<55}  {len(imgs):>5} images")
        paths.extend(str(p) for p in imgs)
        labels.extend(class_name for _ in imgs)

    log.info(f"Total: {len(paths)} images across {len(set(labels))} classes")
    return paths, labels


def extract_features_batch(
    image_paths: list[str], batch_size: int = 64
) -> np.ndarray:
    """
    Run all images through ResNet50 (ImageNet, no top) and return feature matrix.
    Shape: (N, 2048)

    ResNet50 preprocessing:
      - Resize to 224×224
      - Convert to float32
      - Apply keras preprocess_input:  mean-subtract [103.94, 116.78, 123.68]
        (i.e. BGR channel means from ImageNet — this is the correct ResNet50 range)
    """
    import tensorflow as tf
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.applications.resnet50 import preprocess_input

    log.info("Loading ResNet50 backbone (ImageNet weights, include_top=False)…")
    backbone = ResNet50(
        weights="imagenet",
        include_top=False,
        pooling="avg",          # GlobalAveragePooling2D → (batch, 2048)
        input_shape=(224, 224, 3),
    )
    backbone.trainable = False
    log.info("ResNet50 loaded ✓  (backbone params: {:,})".format(backbone.count_params()))

    all_feats = []
    n = len(image_paths)
    n_batches = (n + batch_size - 1) // batch_size

    with tqdm(total=n, unit="img", desc="Feature extraction", ncols=90,
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
        for start in range(0, n, batch_size):
            batch_paths = image_paths[start : start + batch_size]
            batch_imgs  = []

            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB").resize((224, 224))
                    batch_imgs.append(np.array(img, dtype=np.float32))
                except Exception as exc:
                    log.warning(f"    Could not load {p}: {exc}  → zero-filled")
                    batch_imgs.append(np.zeros((224, 224, 3), dtype=np.float32))

            batch_arr = preprocess_input(np.array(batch_imgs, dtype=np.float32))
            feats     = backbone.predict(batch_arr, verbose=0)   # (batch, 2048)
            all_feats.append(feats)
            pbar.update(len(batch_paths))

    return np.vstack(all_feats)   # (N, 2048)


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train ResNet50 + XGBoost crop-disease classifier"
    )
    parser.add_argument("--data",          required=True,  help="PlantVillage dataset root directory")
    parser.add_argument("--output",        default="./model", help="Output directory (default: ./model)")
    parser.add_argument("--batch-size",    type=int,   default=64,   help="Feature extraction batch size")
    parser.add_argument("--n-estimators",  type=int,   default=300,  help="XGBoost n_estimators")
    parser.add_argument("--max-depth",     type=int,   default=6,    help="XGBoost max_depth")
    parser.add_argument("--learning-rate", type=float, default=0.1,  help="XGBoost learning_rate")
    parser.add_argument("--subsample",     type=float, default=0.8,  help="XGBoost row subsampling ratio")
    parser.add_argument("--colsample",     type=float, default=0.6,  help="XGBoost column subsampling ratio")
    parser.add_argument("--test-size",     type=float, default=0.15, help="Test split fraction")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: collect image paths ─────────────────────────────────────
    log.info("=" * 60)
    log.info("STEP 1 — Collecting dataset")
    log.info("=" * 60)
    paths, labels = collect_dataset(args.data)

    # ── Step 2: encode labels ────────────────────────────────────────────
    log.info("=" * 60)
    log.info("STEP 2 — Encoding labels")
    log.info("=" * 60)
    le = LabelEncoder()
    le.fit(CLASS_NAMES)          # fixed order: must match CLASS_NAMES in main.py
    y  = le.transform(labels)    # integer array, each value in [0, 37]
    log.info(f"Classes: {len(le.classes_)}")

    # ── Step 3: extract ResNet50 features (or load cached) ───────────────
    log.info("=" * 60)
    log.info("STEP 3 — Extracting ResNet50 features  (this may take a while)")
    log.info("=" * 60)
    cache_X = out_dir / "features_cache.npy"
    cache_y = out_dir / "labels_cache.npy"
    if cache_X.exists() and cache_y.exists():
        log.info(f"  *** Loading cached features from {cache_X} — skipping ResNet50 pass ***")
        X = np.load(str(cache_X))
        y = np.load(str(cache_y))
        log.info(f"Feature matrix shape: {X.shape}  (loaded from cache)")
    else:
        t0 = time.perf_counter()
        X  = extract_features_batch(paths, batch_size=args.batch_size)
        elapsed = time.perf_counter() - t0
        log.info(f"Feature matrix shape: {X.shape}   ({elapsed:.0f} s)")
        log.info(f"  Saving feature cache → {cache_X}")
        np.save(str(cache_X), X)
        np.save(str(cache_y), y)
        log.info("  Feature cache saved ✓")

    # ── Step 4: scale features ───────────────────────────────────────────
    log.info("=" * 60)
    log.info("STEP 4 — Fitting StandardScaler")
    log.info("=" * 60)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Step 5: train / test split ───────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=args.test_size,
        stratify=y,
        random_state=42,
    )
    log.info(f"Train: {len(X_train)}  Test: {len(X_test)}")

    # ── Step 6: train XGBoost ────────────────────────────────────────────
    log.info("=" * 60)
    log.info("STEP 5 — Training XGBoost")
    log.info("=" * 60)
    clf = xgb.XGBClassifier(
        n_estimators      = args.n_estimators,
        max_depth         = args.max_depth,
        learning_rate     = args.learning_rate,
        subsample         = args.subsample,
        colsample_bytree  = args.colsample,
        objective         = "multi:softprob",
        num_class         = len(CLASS_NAMES),
        n_jobs            = -1,                # use all CPU cores
        eval_metric       = "mlogloss",
        early_stopping_rounds = 20,
        random_state      = 42,
        verbosity         = 1,
    )
    t0 = time.perf_counter()
    clf.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )
    log.info(f"XGBoost training finished in {time.perf_counter()-t0:.0f} s")

    # ── Step 7: evaluate ─────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("STEP 6 — Evaluation")
    log.info("=" * 60)
    y_pred = clf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    log.info(f"Test accuracy: {acc:.4f}  ({acc*100:.2f} %)")
    print()
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # ── Step 8: save artefacts ───────────────────────────────────────────
    log.info("=" * 60)
    log.info("STEP 7 — Saving model artefacts")
    log.info("=" * 60)
    xgb_path    = out_dir / "xgb_model.pkl"
    scaler_path = out_dir / "xgb_scaler.pkl"
    le_path     = out_dir / "xgb_label_encoder.pkl"

    try:
        joblib.dump(clf,    xgb_path)
        log.info(f"Saved XGBoost model  →  {xgb_path}")
        joblib.dump(scaler, scaler_path)
        log.info(f"Saved StandardScaler →  {scaler_path}")
        joblib.dump(le,     le_path)
        log.info(f"Saved LabelEncoder   →  {le_path}")
    except Exception as save_err:
        log.error(f"SAVE FAILED: {save_err}")
        raise

    log.info("")
    log.info("Training complete!")
    log.info("To activate the new model, add to krishi-backend/.env:")
    log.info("  XGB_MODEL_PATH=./model/xgb_model.pkl")
    log.info("Then restart uvicorn.  The backend auto-detects and uses ResNet50+XGBoost.")


if __name__ == "__main__":
    main()
