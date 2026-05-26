#!/usr/bin/env python3
"""
Quick smoke-test for the trained ResNet50 + XGBoost model.

Usage:
    python test_xgb_model.py
    python test_xgb_model.py --image path/to/leaf.jpg
    python test_xgb_model.py --check-only          # just verify files exist
"""
import argparse
import sys
from pathlib import Path

MODEL_DIR   = Path(__file__).parent / "model"
XGB_PATH    = MODEL_DIR / "xgb_model.pkl"
SCALER_PATH = MODEL_DIR / "xgb_scaler.pkl"
LE_PATH     = MODEL_DIR / "xgb_label_encoder.pkl"


def check_files() -> bool:
    ok = True
    for p in [XGB_PATH, SCALER_PATH, LE_PATH]:
        size_mb = p.stat().st_size / 1_048_576 if p.exists() else 0
        status  = f"✓  {size_mb:.1f} MB" if p.exists() else "✗  NOT FOUND"
        print(f"  {p.name:<30} {status}")
        if not p.exists():
            ok = False
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",      default=None,  help="Path to a leaf/crop image to predict")
    parser.add_argument("--check-only", action="store_true", help="Only verify file existence")
    args = parser.parse_args()

    print("\n── Artifact files ──────────────────────────────────────────")
    all_present = check_files()
    if not all_present:
        print("\n❌  One or more artifact files are MISSING.")
        print("   Re-run training:  python train_resnet50_xgboost.py --data /path/to/PlantVillage")
        print("   (Feature cache will be reused if model/features_cache.npy exists — fast re-run)")
        sys.exit(1)

    if args.check_only:
        print("\n✅  All artifact files present.")
        sys.exit(0)

    print("\n── Loading model ────────────────────────────────────────────")
    from resnet50_xgb_predictor import ResNetXGBPredictor
    pred = ResNetXGBPredictor()
    pred.load(str(XGB_PATH), str(SCALER_PATH), str(LE_PATH))
    print("Model loaded ✓")

    # ── use user-supplied image or synthesise a green patch ──────────────
    if args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            print(f"Image not found: {img_path}")
            sys.exit(1)
        image_bytes = img_path.read_bytes()
        print(f"\n── Predicting on: {img_path.name} ────────────────────────")
    else:
        # Synthetic 224×224 green image (roughly leaf-like)
        import numpy as np, io
        from PIL import Image
        arr = np.zeros((224, 224, 3), dtype=np.uint8)
        arr[..., 1] = 120   # green channel
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, "JPEG")
        image_bytes = buf.getvalue()
        print("\n── Predicting on: synthetic green image (no real image supplied) ────")

    idx, confidence = pred.predict(image_bytes)

    import joblib
    le = joblib.load(str(LE_PATH))
    class_name = le.inverse_transform([idx])[0] if le is not None else f"class_{idx}"

    print(f"\n  Class index : {idx}")
    print(f"  Class name  : {class_name}")
    print(f"  Confidence  : {confidence:.2%}")
    print("\n✅  Smoke-test passed — model is ready to use.\n")
    print("To activate in backend, add to krishi-backend/.env:")
    print("  XGB_MODEL_PATH=./model/xgb_model.pkl")
    print("Then restart uvicorn.")


if __name__ == "__main__":
    main()
