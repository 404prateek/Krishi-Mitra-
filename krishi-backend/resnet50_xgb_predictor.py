"""
ResNet50 + XGBoost hybrid crop-disease predictor.

Architecture
------------
  Image (224×224×3)
       │
  ResNet50 backbone  (ImageNet pre-trained, include_top=False, pooling='avg')
       │   → 50 conv layers with residual / skip connections
       │   → final feature map: 7×7×2048  →  GlobalAveragePooling  →  2048-dim
       │
  StandardScaler  (zero-mean, unit-variance per feature)
       │
  XGBoostClassifier  (objective='multi:softprob', 38 classes)
       │
  Class index + confidence

Why ResNet50 instead of MobileNetV2 (current model)
----------------------------------------------------
  • Residual / skip connections  →  deeper network without vanishing gradients
  • 2048-dim features (vs 1280 in MobileNetV2)  →  richer representation
  • XGBoost as classifier  →  no GPU needed, interpretable, handles imbalance
  • Typical PlantVillage accuracy: 97-99 %  (vs 90-95 % for vanilla fine-tuning)
"""

import io
import logging
from pathlib import Path

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


class ResNetXGBPredictor:
    """Lazy-loads ResNet50 feature extractor + XGBoost classifier."""

    def __init__(self):
        self._extractor = None   # ResNet50 keras model (no classification head)
        self._xgb       = None   # XGBoostClassifier
        self._scaler    = None   # sklearn StandardScaler
        self._le        = None   # sklearn LabelEncoder (class index → name)

    # ── public API ──────────────────────────────────────────────────────────

    def load(self, xgb_path: str, scaler_path: str = None, le_path: str = None):
        """Load artefacts produced by train_resnet50_xgboost.py."""
        import joblib
        log.info(f"Loading XGBoost model from {xgb_path}")
        self._xgb = joblib.load(xgb_path)

        if scaler_path and Path(scaler_path).exists():
            self._scaler = joblib.load(scaler_path)
            log.info("StandardScaler loaded ✓")

        if le_path and Path(le_path).exists():
            self._le = joblib.load(le_path)
            log.info("LabelEncoder loaded ✓")

    def predict(self, image_bytes: bytes) -> tuple[int, float]:
        """
        Returns (class_index, confidence).
        class_index indexes into CLASS_NAMES in main.py (same 38-class order).
        """
        if self._xgb is None:
            raise RuntimeError("XGBoost model not loaded. Call .load() first.")

        features = self._extract_features(image_bytes)  # (1, 2048)

        if self._scaler is not None:
            features = self._scaler.transform(features)

        proba      = self._xgb.predict_proba(features)[0]   # (38,)
        idx        = int(np.argmax(proba))
        confidence = float(proba[idx])
        return idx, confidence

    def extract_features(self, image_bytes: bytes) -> np.ndarray:
        """Public: return raw 2048-dim feature vector (before scaling)."""
        return self._extract_features(image_bytes)

    # ── internals ───────────────────────────────────────────────────────────

    def _get_extractor(self):
        """Lazy-initialise ResNet50 once per process."""
        if self._extractor is None:
            from tensorflow.keras.applications import ResNet50
            log.info("Initialising ResNet50 feature extractor (ImageNet weights)…")
            self._extractor = ResNet50(
                weights="imagenet",
                include_top=False,
                pooling="avg",           # GlobalAveragePooling2D → 2048-dim vector
                input_shape=(224, 224, 3),
            )
            self._extractor.trainable = False
            log.info("ResNet50 feature extractor ready ✓")
        return self._extractor

    def _extract_features(self, image_bytes: bytes) -> np.ndarray:
        """Preprocess one image and run it through the ResNet50 backbone."""
        from tensorflow.keras.applications.resnet50 import preprocess_input

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
        arr = np.array(img, dtype=np.float32)
        arr = preprocess_input(arr)           # ResNet50 convention: mean-subtract BGR
        arr = np.expand_dims(arr, axis=0)     # → (1, 224, 224, 3)

        extractor = self._get_extractor()
        features  = extractor.predict(arr, verbose=0)   # → (1, 2048)
        return features
