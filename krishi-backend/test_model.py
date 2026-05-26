#!/usr/bin/env python3
"""
Quick test script — works with BOTH the existing MobileNetV2 model
and (after training finishes) the ResNet50 + XGBoost model.

Usage:
    # Test with a specific image
    python test_model.py --image path/to/leaf.jpg

    # Auto-pick one sample image per class from the dataset
    python test_model.py --dataset ./dataset/"New Plant Diseases Dataset(Augmented)"/"New Plant Diseases Dataset(Augmented)"/train --samples 5

    # Test via live HTTP API (server must be running on port 8000)
    python test_model.py --image path/to/leaf.jpg --api
"""

import argparse
import os
import sys
import time
from pathlib import Path

# ─── direct predictor test (no server needed) ──────────────────────────────

CLASS_NAMES = [
    "Apple___Apple_scab", "Apple___Black_rot",
    "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight",
    "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus", "Tomato___healthy",
]


def test_mobilenetv2(image_path: str):
    """Test using existing MobileNetV2 .h5 model directly."""
    import numpy as np
    import tensorflow as tf
    from PIL import Image

    model_path = Path(__file__).parent / "model" / "krishi_model.h5"
    if not model_path.exists():
        print(f"  ✗ MobileNetV2 model not found at {model_path}")
        return None

    print(f"  Loading MobileNetV2 from {model_path}...")
    model = tf.keras.models.load_model(str(model_path))

    img = Image.open(image_path).convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    t0 = time.perf_counter()
    preds = model.predict(arr, verbose=0)
    latency = (time.perf_counter() - t0) * 1000

    idx        = int(preds[0].argmax())
    confidence = float(preds[0].max())
    top3       = sorted(enumerate(preds[0]), key=lambda x: -x[1])[:3]

    return {
        "backend":    "MobileNetV2 (Keras)",
        "predicted":  CLASS_NAMES[idx],
        "confidence": confidence,
        "latency_ms": latency,
        "top3": [(CLASS_NAMES[i], float(p)) for i, p in top3],
    }


def test_resnet50_xgb(image_path: str):
    """Test using ResNet50 + XGBoost (only works after training is complete)."""
    xgb_path    = Path(__file__).parent / "model" / "xgb_model.pkl"
    scaler_path = Path(__file__).parent / "model" / "xgb_scaler.pkl"
    le_path     = Path(__file__).parent / "model" / "xgb_label_encoder.pkl"

    if not xgb_path.exists():
        print(f"  ⏳ XGBoost model not ready yet at {xgb_path}")
        print(f"     Training is still running — check the python terminal.")
        return None

    sys.path.insert(0, str(Path(__file__).parent))
    from resnet50_xgb_predictor import ResNetXGBPredictor

    pred = ResNetXGBPredictor()
    pred.load(str(xgb_path), str(scaler_path), str(le_path))

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    t0 = time.perf_counter()
    idx, confidence = pred.predict(image_bytes)
    latency = (time.perf_counter() - t0) * 1000

    return {
        "backend":    "ResNet50 + XGBoost",
        "predicted":  CLASS_NAMES[idx],
        "confidence": confidence,
        "latency_ms": latency,
        "top3": [(CLASS_NAMES[idx], confidence)],   # XGB full proba not needed here
    }


def test_via_api(image_path: str, url: str = "http://localhost:8000/analyze"):
    """Test via the live FastAPI server."""
    import requests
    with open(image_path, "rb") as f:
        files = {"file": (Path(image_path).name, f, "image/jpeg")}
        t0 = time.perf_counter()
        resp = requests.post(url, files=files, timeout=60)
        latency = (time.perf_counter() - t0) * 1000

    if resp.status_code != 200:
        print(f"  ✗ API error {resp.status_code}: {resp.text[:200]}")
        return None

    data = resp.json()
    return {
        "backend":    "HTTP API (/analyze)",
        "predicted":  f"{data.get('crop')} — {data.get('disease')}",
        "confidence": data.get("confidence", 0),
        "latency_ms": latency,
        "severity":   data.get("severity"),
        "healthy":    data.get("is_healthy"),
    }


def print_result(result: dict, image_path: str):
    if result is None:
        return
    print(f"\n  {'─'*54}")
    print(f"  Image    : {Path(image_path).name}")
    print(f"  Backend  : {result['backend']}")
    print(f"  Predicted: {result['predicted']}")
    print(f"  Confidence: {result['confidence']*100:.1f}%")
    print(f"  Latency  : {result['latency_ms']:.0f} ms")
    if "top3" in result and len(result["top3"]) > 1:
        print(f"  Top-3:")
        for cls, prob in result["top3"]:
            print(f"    {prob*100:5.1f}%  {cls}")
    if "severity" in result:
        print(f"  Severity : {result['severity']}")
    print(f"  {'─'*54}")


def collect_sample_images(dataset_dir: str, n: int = 5) -> list[str]:
    """Pick n random sample images from the dataset (one per class, cycling)."""
    import random
    base = Path(dataset_dir)
    images = []
    class_dirs = [d for d in sorted(base.iterdir()) if d.is_dir()]
    random.seed(42)
    selected = random.sample(class_dirs, min(n, len(class_dirs)))
    for cls_dir in selected:
        imgs = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.JPG")) + list(cls_dir.glob("*.png"))
        if imgs:
            images.append((str(random.choice(imgs)), cls_dir.name))
    return images


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test Krishi Mitra disease models")
    parser.add_argument("--image",   help="Path to a single leaf/crop image to test")
    parser.add_argument("--dataset", help="Dataset train/ directory — auto-picks sample images")
    parser.add_argument("--samples", type=int, default=5, help="Number of sample images from dataset")
    parser.add_argument("--api",     action="store_true", help="Also test via HTTP API (server must be running)")
    parser.add_argument("--xgb-only", action="store_true", help="Only test XGBoost model (skip MobileNetV2)")
    args = parser.parse_args()

    if not args.image and not args.dataset:
        # Auto-find a sample from the local dataset if it exists
        default_ds = Path("./dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train")
        if default_ds.exists():
            print(f"No --image specified. Auto-sampling from {default_ds}")
            args.dataset = str(default_ds)
            args.samples = 3
        else:
            print("Provide --image <path> or --dataset <train-dir>")
            sys.exit(1)

    # Collect images to test
    test_items = []
    if args.image:
        test_items.append((args.image, "manual"))
    if args.dataset:
        test_items.extend(collect_sample_images(args.dataset, args.samples))

    print(f"\n{'='*58}")
    print(f"  Krishi Mitra — Model Test  ({len(test_items)} image(s))")
    print(f"{'='*58}")

    for img_path, true_label in test_items:
        print(f"\nGround truth : {true_label}")

        if not args.xgb_only:
            result = test_mobilenetv2(img_path)
            correct = true_label.lower() in result["predicted"].lower() if result else False
            if result:
                result["correct"] = correct
            print_result(result, img_path)
            if result:
                print(f"  Correct? : {'✓ YES' if correct else '✗ NO'}")

        # XGBoost test (skipped with a message if model not ready)
        xgb_result = test_resnet50_xgb(img_path)
        if xgb_result:
            correct_xgb = true_label.lower() in xgb_result["predicted"].lower()
            print_result(xgb_result, img_path)
            print(f"  Correct? : {'✓ YES' if correct_xgb else '✗ NO'}")

        if args.api:
            api_result = test_via_api(img_path)
            print_result(api_result, img_path)

    print(f"\n{'='*58}\n")


if __name__ == "__main__":
    main()
