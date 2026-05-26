"""
Krishi Mitra — FastAPI Backend (Production)
============================================
Endpoints:
  POST /analyze          → ML model + Gemini advisory
  POST /advisory         → Gemini advisory only
  GET  /health           → Status check
  GET  /classes          → All 38 class names
  POST /api/chat         → Secure Gemini chat proxy
  GET  /api/weather      → OpenWeatherMap proxy (lat/lon or city)
  GET  /api/mandi        → Agmarknet data proxy

Run:
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import os, io, logging, time, asyncio, json, uuid, sqlite3
from pathlib import Path
from typing import Optional
from collections import deque
import urllib.request

import httpx
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from config_store import (
    get_all_config,
    get_api_key,
    get_config_status,
    init_db,
    sync_env_to_db,
    purge_stale_keys,
)

# ─── NEW SDK: google-genai (replaces deprecated google-generativeai) ──────────
try:
    from google import genai
    from google.genai import types as genai_types
    _GENAI_NEW_SDK = True
except ImportError:
    # graceful fallback — warn loudly
    _GENAI_NEW_SDK = False
    logging.warning("google-genai not installed. AI features disabled. Run: pip install google-genai")

load_dotenv()
init_db()
sync_env_to_db(dict(os.environ))
purge_stale_keys()   # remove obsolete keys (e.g. WHISPER_*) left from older runs

CONFIG = get_all_config()

# ─── Scan History (SQLite + local image files) ────────────────────────────────
_HISTORY_DB = Path(__file__).resolve().parent / "config.db"   # reuse same DB file
_UPLOADS_DIR = Path(__file__).resolve().parent / "uploads"
_UPLOADS_DIR.mkdir(exist_ok=True)

def _history_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_HISTORY_DB)
    conn.row_factory = sqlite3.Row
    return conn

def _init_history_db():
    conn = _history_conn()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scan_history (
                id            TEXT PRIMARY KEY,
                image_path    TEXT NOT NULL,
                disease       TEXT NOT NULL,
                crop          TEXT NOT NULL,
                confidence    REAL NOT NULL,
                severity      TEXT NOT NULL,
                is_healthy    INTEGER NOT NULL DEFAULT 0,
                advisory_short  TEXT,
                advisory_detail TEXT,
                economic_impact TEXT,
                created_at    TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
    finally:
        conn.close()

_init_history_db()

def _save_scan(image_bytes: bytes, result: dict) -> str:
    """Save image to disk and record metadata in DB. Returns the new scan ID."""
    scan_id = str(uuid.uuid4())
    img_path = _UPLOADS_DIR / f"{scan_id}.jpg"
    # Re-encode as JPEG for consistent storage
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img.thumbnail((800, 800))          # cap at 800px — saves disk space
    img.save(img_path, "JPEG", quality=85)

    conn = _history_conn()
    try:
        conn.execute("""
            INSERT INTO scan_history
              (id, image_path, disease, crop, confidence, severity,
               is_healthy, advisory_short, advisory_detail, economic_impact)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (
            scan_id, str(img_path),
            result.get("disease", ""),
            result.get("crop", ""),
            result.get("confidence", 0.0),
            result.get("severity", "medium"),
            1 if result.get("is_healthy") else 0,
            result.get("advisory_short", ""),
            result.get("advisory_detail", ""),
            result.get("economic_impact", ""),
        ))
        conn.commit()
    finally:
        conn.close()
    return scan_id

def _get_history(limit: int = 30) -> list[dict]:
    conn = _history_conn()
    try:
        rows = conn.execute("""
            SELECT id, disease, crop, confidence, severity, is_healthy,
                   advisory_short, advisory_detail, economic_impact, created_at
            FROM scan_history
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()

GEMINI_API_KEY    = get_api_key("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY", "")

# ─── Gemini key pool: rotate across keys when one hits quota ─────────────────
def _build_key_pool() -> list[str]:
    """Collect all non-empty Gemini API keys in priority order."""
    raw = [
        GEMINI_API_KEY,
        get_api_key("GEMINI_API_KEY_2") or os.getenv("GEMINI_API_KEY_2", ""),
        get_api_key("GEMINI_API_KEY_3") or os.getenv("GEMINI_API_KEY_3", ""),
        get_api_key("GEMINI_API_KEY_4") or os.getenv("GEMINI_API_KEY_4", ""),
    ]
    return [k.strip() for k in raw if k.strip()]

GEMINI_KEY_POOL: list[str] = _build_key_pool()

# Maps key → unix timestamp when it was marked exhausted. Resets after 1 hour.
_key_exhausted_at: dict[str, float] = {}
_KEY_COOLDOWN_SECS = 3600  # 1 hour — Gemini free-tier daily quota resets ~hourly

def _key_available(key: str) -> bool:
    """True if the key is not in the cooldown window."""
    t = _key_exhausted_at.get(key)
    if t is None:
        return True
    if time.time() - t >= _KEY_COOLDOWN_SECS:
        del _key_exhausted_at[key]  # auto-recover after cooldown
        return True
    return False

def _mark_key_exhausted(key: str):
    _key_exhausted_at[key] = time.time()
    remaining = sum(1 for k in GEMINI_KEY_POOL if _key_available(k))
    log.warning(f"Key ...{key[-6:]} marked exhausted. Keys still available: {remaining}/{len(GEMINI_KEY_POOL)}")
WEATHER_API_KEY   = get_api_key("WEATHER_API_KEY") or os.getenv("WEATHER_API_KEY", "")
AGMARKNET_KEY     = get_api_key("AGMARKNET_KEY") or os.getenv("AGMARKNET_KEY", "")
ELEVENLABS_API_KEY = get_api_key("ELEVENLABS_API_KEY") or os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_STT_URL = get_api_key("ELEVENLABS_STT_URL") or os.getenv(
    "ELEVENLABS_STT_URL",
    "https://api.elevenlabs.io/v1/speech-to-text",
)
ELEVENLABS_STT_MODEL = get_api_key("ELEVENLABS_STT_MODEL") or os.getenv("ELEVENLABS_STT_MODEL", "scribe_v1")
GOV_SCHEMES_URL   = get_api_key("GOV_SCHEMES_URL") or os.getenv("GOV_SCHEMES_URL", "")
MODEL_PATH        = get_api_key("MODEL_PATH") or os.getenv("MODEL_PATH", "./model/krishi_model.h5")
USE_MOCK_MODEL    = (get_api_key("USE_MOCK_MODEL") or os.getenv("USE_MOCK_MODEL", "false")).lower() == "true"
# ResNet50 + XGBoost hybrid model (optional — set XGB_MODEL_PATH in .env to activate)
XGB_MODEL_PATH    = os.getenv("XGB_MODEL_PATH", "")
ALLOWED_ORIGINS   = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://localhost:3000"
).split(",")

DEFAULT_SCHEMES = [
    {
        "id": "pm-kisan",
        "title": "PM-KISAN",
        "fullName": "Pradhan Mantri Kisan Samman Nidhi",
        "category": "income",
        "states": "all",
        "crops": "all",
        "benefit": "₹6,000/year direct transfer (₹2,000 every 4 months)",
        "eligibility": "Land-owning farmer families with cultivable land",
        "landLimit": None,
        "documents": ["Aadhar Card", "Land Records (Khasra/Khatauni)", "Bank Account"],
        "deadline": "Ongoing",
        "link": "https://pmkisan.gov.in",
        "tags": ["income support", "all farmers", "central scheme"],
    },
    {
        "id": "pmfby",
        "title": "PMFBY",
        "fullName": "Pradhan Mantri Fasal Bima Yojana",
        "category": "insurance",
        "states": "all",
        "crops": ["rice", "wheat", "cotton", "oilseeds", "pulses", "maize"],
        "benefit": "Full crop insurance coverage against natural calamities",
        "eligibility": "Farmers growing notified crops in notified areas",
        "documents": ["Aadhar", "Land Records", "Bank Account", "Sowing Certificate"],
        "deadline": "Before sowing season",
        "link": "https://pmfby.gov.in",
        "tags": ["insurance", "crop loss", "natural disaster"],
    },
    {
        "id": "kcc",
        "title": "Kisan Credit Card",
        "fullName": "KCC - Kisan Credit Card Scheme",
        "category": "credit",
        "states": "all",
        "crops": "all",
        "benefit": "Credit up to ₹3 lakh at 4% interest (after 2% subvention)",
        "eligibility": "All farmers, sharecroppers, tenant farmers",
        "documents": ["Aadhar", "Land Records / Tenancy Agreement", "Passport Photo"],
        "deadline": "Ongoing",
        "link": "https://rbi.org.in",
        "tags": ["credit", "loan", "low interest"],
    },
    {
        "id": "pkvy",
        "title": "PKVY",
        "fullName": "Paramparagat Krishi Vikas Yojana",
        "category": "organic",
        "states": "all",
        "crops": "all",
        "benefit": "₹50,000/ha over 3 years for organic farming transition",
        "eligibility": "Farmer groups (minimum 50 farmers, 50 acres) adopting organic",
        "documents": ["Group Registration", "Land Records", "Bank Account"],
        "deadline": "Seasonal",
        "link": "https://agricoop.nic.in",
        "tags": ["organic", "group farming", "subsidy"],
    },
    {
        "id": "mgnregs-agri",
        "title": "MGNREGS (Agriculture)",
        "fullName": "Mahatma Gandhi NREGS - Agricultural Works",
        "category": "employment",
        "states": "all",
        "crops": "all",
        "benefit": "100 days guaranteed wage employment for farm-related work",
        "eligibility": "Rural households with job cards",
        "documents": ["Job Card", "Aadhar", "Bank Account"],
        "deadline": "Ongoing",
        "link": "https://nrega.nic.in",
        "tags": ["employment", "rural", "wage"],
    },
    {
        "id": "rkvy",
        "title": "RKVY",
        "fullName": "Rashtriya Krishi Vikas Yojana",
        "category": "infrastructure",
        "states": "all",
        "crops": "all",
        "benefit": "State-specific: farm infrastructure, machinery, storage",
        "eligibility": "Farmers via state government projects",
        "documents": ["State-specific"],
        "deadline": "Varies by state",
        "link": "https://rkvy.nic.in",
        "tags": ["infrastructure", "machinery", "state scheme"],
    },
    {
        "id": "maha-shetkari",
        "title": "Shetkari Sanman Yojana",
        "fullName": "Maharashtra Shetkari Sanman Yojana",
        "category": "income",
        "states": ["Maharashtra"],
        "crops": "all",
        "benefit": "₹12,000/year additional state support (on top of PM-KISAN)",
        "eligibility": "Maharashtra farmers registered under PM-KISAN",
        "documents": ["PM-KISAN registration", "Aadhar", "7/12 Extract"],
        "deadline": "Ongoing",
        "link": "https://krishi.maharashtra.gov.in",
        "tags": ["Maharashtra", "income support", "state scheme"],
    },
    {
        "id": "karnataka-ryothaseva",
        "title": "Raitha Siri",
        "fullName": "Karnataka Raitha Siri Yojane",
        "category": "income",
        "states": ["Karnataka"],
        "crops": "all",
        "benefit": "₹4,000/year state income support",
        "eligibility": "Karnataka farmers with less than 5 acres land",
        "documents": ["Aadhar", "RTC (Record of Rights)", "Bank Account"],
        "deadline": "Ongoing",
        "link": "https://raitamitra.karnataka.gov.in",
        "tags": ["Karnataka", "income support", "small farmers"],
    },
    {
        "id": "kalia-odisha",
        "title": "KALIA",
        "fullName": "Krushak Assistance for Livelihood and Income Augmentation",
        "category": "income",
        "states": ["Odisha"],
        "crops": "all",
        "benefit": "₹10,000/year + crop insurance + life insurance",
        "eligibility": "Small and marginal farmers in Odisha",
        "documents": ["Aadhar", "Land Records", "Bank Account"],
        "deadline": "Ongoing",
        "link": "https://kalia.odisha.gov.in",
        "tags": ["Odisha", "income support", "insurance"],
    },
    {
        "id": "rythu-bandhu",
        "title": "Rythu Bandhu",
        "fullName": "Telangana Rythu Bandhu Scheme",
        "category": "income",
        "states": ["Telangana"],
        "crops": "all",
        "benefit": "₹10,000/acre/year (₹5,000 per season)",
        "eligibility": "Land-owning farmers in Telangana (not tenant farmers)",
        "documents": ["Aadhar", "Pattadar Passbook / Land Records"],
        "deadline": "Before each sowing season",
        "link": "https://rythubandhu.telangana.gov.in",
        "tags": ["Telangana", "investment support", "per acre"],
    },
]

# ─── Model chain (fallback order when primary is overloaded) ─────────────────
# gemini-2.0-flash is quota-exhausted on this project — excluded from chain
PRIMARY_MODEL    = "gemini-2.5-flash"
FALLBACK_MODEL   = "gemini-2.5-flash-lite"
FALLBACK_MODEL_2 = "gemini-flash-latest"

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("krishi")

app = FastAPI(title="Krishi Mitra API", version="3.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # dev — lock to ALLOWED_ORIGINS in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Rate Limiter (global, in-memory) ─────────────────────────────────────────
# Gemini 2.5 Flash free tier: 10 RPM, 500 RPD
_rate_window   = deque()          # timestamps of recent AI calls
RATE_LIMIT_RPM = 8                # safe below the 10 RPM free-tier ceiling
RATE_LIMIT_RPD = 450              # safe below the 500 RPD free-tier ceiling
_daily_count   = 0
_daily_reset   = time.time()

def _check_rate_limit() -> tuple[bool, str]:
    """Returns (allowed: bool, reason: str)."""
    global _daily_count, _daily_reset
    now = time.time()

    # Reset daily counter every 24h
    if now - _daily_reset > 86400:
        _daily_count = 0
        _daily_reset = now

    if _daily_count >= RATE_LIMIT_RPD:
        log.warning(f"Daily AI quota reached ({_daily_count}/{RATE_LIMIT_RPD})")
        return False, "daily_quota"

    # Sliding-window RPM check (keep only last 60 seconds)
    while _rate_window and now - _rate_window[0] > 60:
        _rate_window.popleft()

    if len(_rate_window) >= RATE_LIMIT_RPM:
        log.warning(f"Minute rate limit hit ({len(_rate_window)}/{RATE_LIMIT_RPM})")
        return False, "minute_quota"

    return True, "ok"

def _record_ai_call():
    global _daily_count
    _rate_window.append(time.time())
    _daily_count += 1
    log.info(f"AI call recorded — daily: {_daily_count}/{RATE_LIMIT_RPD}, rpm: {len(_rate_window)}/{RATE_LIMIT_RPM}")

# ─── Simple in-memory cache ────────────────────────────────────────────────────
_cache: dict[str, tuple[dict, float]] = {}

def cache_get(key: str, ttl: int = 300) -> Optional[dict]:
    if key in _cache:
        value, ts = _cache[key]
        if time.time() - ts < ttl:
            return value
        del _cache[key]
    return None

def cache_set(key: str, value: dict):
    _cache[key] = (value, time.time())

# ─── 38 PlantVillage class names ──────────────────────────────────────────────
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

def parse_class_name(raw: str) -> tuple[str, str, bool]:
    parts   = raw.split("___")
    crop    = parts[0].replace("_", " ").strip()
    disease = parts[1].replace("_", " ").replace("  ", " ").strip().title() if len(parts) > 1 else "Unknown"
    healthy = "healthy" in disease.lower()
    return crop, disease, healthy

# ─── ML model ─────────────────────────────────────────────────────────────────
_ml_model    = None   # Keras / MobileNetV2 model (legacy path)
_xgb_pred    = None   # ResNet50 + XGBoost predictor (activated via XGB_MODEL_PATH)

def _get_xgb_predictor():
    """Lazy-load ResNet50+XGBoost predictor. Returns None when not configured."""
    global _xgb_pred
    if _xgb_pred is not None:
        return _xgb_pred
    if not XGB_MODEL_PATH:
        return None
    xgb_file = Path(XGB_MODEL_PATH)
    if not xgb_file.exists():
        log.warning(f"XGB_MODEL_PATH set but file not found: {XGB_MODEL_PATH}")
        return None
    from resnet50_xgb_predictor import ResNetXGBPredictor
    _xgb_pred = ResNetXGBPredictor()
    scaler_path = str(xgb_file.parent / "xgb_scaler.pkl")
    le_path     = str(xgb_file.parent / "xgb_label_encoder.pkl")
    _xgb_pred.load(str(xgb_file), scaler_path, le_path)
    log.info("ResNet50+XGBoost predictor loaded ✓")
    return _xgb_pred

def get_ml_model():
    """Return Keras model (MobileNetV2). Used only when XGB predictor is absent."""
    global _ml_model
    if _ml_model is not None:
        return _ml_model
    if USE_MOCK_MODEL:
        log.warning("MOCK MODE — random predictions")
        return None
    model_file = Path(MODEL_PATH)
    if not model_file.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Set USE_MOCK_MODEL=true in .env for dev mode."
        )
    import tensorflow as tf
    log.info(f"Loading MobileNetV2 model from {MODEL_PATH} …")
    _ml_model = tf.keras.models.load_model(MODEL_PATH)
    log.info("MobileNetV2 model loaded ✓")
    return _ml_model

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def validate_crop_image(image_bytes: bytes) -> tuple[bool, str]:
    """Reject obvious non-crop uploads before running disease prediction."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.asarray(img, dtype=np.float32) / 255.0

    r = arr[..., 0]
    g = arr[..., 1]
    b = arr[..., 2]

    max_rgb = np.maximum(np.maximum(r, g), b)
    min_rgb = np.minimum(np.minimum(r, g), b)
    delta = max_rgb - min_rgb

    hue = np.zeros_like(max_rgb)
    nonzero = delta > 1e-6
    hue[nonzero & (max_rgb == r)] = ((g[nonzero & (max_rgb == r)] - b[nonzero & (max_rgb == r)]) / delta[nonzero & (max_rgb == r)]) % 6
    hue[nonzero & (max_rgb == g)] = ((b[nonzero & (max_rgb == g)] - r[nonzero & (max_rgb == g)]) / delta[nonzero & (max_rgb == g)]) + 2
    hue[nonzero & (max_rgb == b)] = ((r[nonzero & (max_rgb == b)] - g[nonzero & (max_rgb == b)]) / delta[nonzero & (max_rgb == b)]) + 4
    hue = hue / 6.0

    saturation = np.where(max_rgb == 0, 0.0, delta / max_rgb)
    value = max_rgb

    green_ratio = float(np.mean((hue >= 0.22) & (hue <= 0.45) & (saturation >= 0.18) & (value >= 0.12)))
    saturation_mean = float(np.mean(saturation))
    color_variance = float(np.var(arr))
    colorfulness = float(np.std(np.abs(r - g)) + np.std(np.abs(r - b)) + np.std(np.abs(g - b)))

    gray = np.mean(arr, axis=2)
    gx = np.diff(gray, axis=1, append=gray[:, -1:])
    gy = np.diff(gray, axis=0, append=gray[-1:, :])
    edge_density = float(np.mean(np.sqrt(gx * gx + gy * gy) > 0.03))

    # Reject flat, low-structure images that do not contain plant-like green cues.
    if green_ratio < 0.02 and color_variance < 0.02 and edge_density < 0.01:
        return False, "Please upload a crop or leaf image for disease detection."

    # Reject strongly saturated non-green solids that do not show any leaf-like structure.
    if green_ratio < 0.03 and saturation_mean > 0.55 and colorfulness < 0.10 and edge_density < 0.02:
        return False, "Please upload a crop or leaf image for disease detection."

    # Reject very weak plant-like images that are almost uniform and lack visible texture.
    if green_ratio < 0.05 and color_variance < 0.01 and colorfulness < 0.05:
        return False, "Please upload a crop or leaf image for disease detection."

    return True, ""


def predict_disease(image_bytes: bytes):
    # ── 1. Mock mode ──────────────────────────────────────────────────────
    if USE_MOCK_MODEL:
        idx        = np.random.randint(0, len(CLASS_NAMES))
        confidence = float(np.random.uniform(0.72, 0.98))
        raw_class  = CLASS_NAMES[idx]
        crop, disease, healthy = parse_class_name(raw_class)
        log.info(f"[MOCK] {raw_class} ({confidence:.1%})")
        return raw_class, confidence, crop, disease, healthy

    # ── 2. ResNet50 + XGBoost (preferred when XGB_MODEL_PATH is set) ──────
    # Wrapped in try/except: any XGB failure automatically falls back to MobileNetV2
    try:
        xgb_pred = _get_xgb_predictor()
        if xgb_pred is not None:
            idx, confidence = xgb_pred.predict(image_bytes)
            raw_class       = CLASS_NAMES[idx]
            crop, disease, healthy = parse_class_name(raw_class)
            log.info(f"[ResNet50+XGB] {raw_class} ({confidence:.1%})")
            return raw_class, confidence, crop, disease, healthy
    except Exception as xgb_err:
        log.error(f"[ResNet50+XGB] failed ({xgb_err}), falling back to MobileNetV2")

    # ── 3. MobileNetV2 Keras model (legacy / default) ─────────────────────
    model = get_ml_model()
    arr         = preprocess_image(image_bytes)
    preds       = model.predict(arr, verbose=0)
    idx         = int(np.argmax(preds[0]))
    confidence  = float(np.max(preds[0]))
    raw_class   = CLASS_NAMES[idx]
    crop, disease, healthy = parse_class_name(raw_class)
    log.info(f"[MobileNetV2] {raw_class} ({confidence:.1%})")
    return raw_class, confidence, crop, disease, healthy

# ─── Gemini prompts ───────────────────────────────────────────────────────────

def _estimate_severity_from_confidence(confidence: float) -> str:
    """Pre-estimate severity from model confidence for use in Gemini prompt."""
    if confidence >= 0.85: return "Severe"
    if confidence >= 0.60: return "Moderate"
    return "Low"


def build_detailed_prompt(disease: str, confidence: float, crop: str,
                           region: str, season: str, language: str,
                           severity: str = "Moderate", land_acres: float = 2.0) -> str:
    lang_name = {"en": "English", "hi": "Hindi", "ml": "Malayalam"}.get(language, "English")

    return f"""You are an expert agricultural advisor for Indian farmers. Respond ONLY in the exact format below. Do not add any extra text, markdown, or explanation outside these keys.

A farmer's crop photo was analyzed by an AI model:
- Crop: {crop}
- Disease Detected: {disease}
- Model Confidence: {confidence:.0%}
- Farmer's Region: {region}
- Current Season: {season}
- Land Size: {land_acres} acres

DISEASE_SUMMARY: [2-3 sentences explaining what this disease is, how it spreads, in simple language]
SEVERITY: [{severity}]
VISIBLE_SYMPTOMS:
- [symptom 1]
- [symptom 2]
- [symptom 3]
IMMEDIATE_ACTION_48H:
- [specific action 1]
- [specific action 2]
- [specific action 3]
CHEMICAL_TREATMENT: [specific chemical name, dosage, application method. Mention cheapest generic option, not branded]
ORGANIC_ALTERNATIVE: [organic/home remedy option that is low cost]
PREVENTION_NEXT_SEASON:
- [prevention tip 1]
- [prevention tip 2]
ECONOMIC_IMPACT: [Estimate rupee loss if untreated for {land_acres} acres. Format: "Potential loss: ₹X,XXX - ₹X,XXX if not treated within 48 hours"]
CALL_OFFICER_IF: [specific condition when farmer should escalate to expert]
FARMER_TIP: [one practical memorable tip in simple Hindi-friendly language]

Disease: {disease}
Crop: {crop}
Severity: {severity}
Land size: {land_acres} acres
Respond in: {"Hindi" if language == "hi" else "Malayalam" if language == "ml" else "English"}

FORMATTING RULES:
- Keep ALL section LABELS in English — do not translate them
- NO markdown, NO asterisks, NO bold, NO headers with #
- Each bullet point starts with a hyphen (-)
- Be specific with amounts, timings, and percentages
- Tone: expert but warm, like a trusted village doctor"""


def build_short_prompt(disease: str, confidence: float, crop: str,
                        region: str, season: str, language: str) -> str:
    lang_name = {"en": "English", "hi": "Hindi", "ml": "Malayalam",
                 "bn": "Bengali", "te": "Telugu", "mr": "Marathi"}.get(language, "Hindi")
    return f"""You are a helpful farm advisor. Write a SHORT WhatsApp message for a farmer.
STRICT: Only agriculture topics. Do NOT hallucinate or guess location.

Disease: {disease} on {crop}
Confidence: {confidence:.0%}
Region: {region}

Rules:
- MAXIMUM 3 sentences total
- First sentence: what the farmer should do TODAY (one specific action)
- Second sentence: what product to use (generic name, not brand)
- Third sentence: one prevention tip
- Write in {lang_name} only
- Conversational tone, like texting a friend
- NO greetings, NO "Hello", start directly with the advice"""


def build_chat_prompt(question: str, state: str, weather_summary: str,
                       active_crops: str, language: str) -> str:
    lang_name = {
        "en": "English", "hi": "Hindi", "ml": "Malayalam",
        "bn": "Bengali", "te": "Telugu", "mr": "Marathi",
    }.get(language, "Hindi")

    system_prompt = f"""You are Krishi Mitra, an expert AI farming assistant for Indian farmers.

LANGUAGE RULE: You MUST respond ONLY in {lang_name}. Never switch languages mid-response.

YOUR EXPERTISE:
- Crop diseases: diagnosis, treatment, prevention
- Fertilizers: which to use, dosage, timing
- Weather: how to act on weather changes
- Market prices: when to sell, which mandi
- Government schemes: PM-KISAN, Fasal Bima, state schemes
- Soil health: NPK, pH, improvement methods
- Irrigation: scheduling, water conservation

RESPONSE RULES:
1. Keep responses SHORT — 3-5 sentences max for simple questions
2. Always give ACTIONABLE advice, not just information
3. Always mention RUPEE COST impact where relevant ("Isse aapko ₹500-1000 ka faida hoga")
4. For disease questions, always ask: "Kya aap photo upload kar sakte hain for exact diagnosis?"
5. Never recommend expensive branded products — always mention generic/cheaper alternatives
6. If question is unclear, ask ONE clarifying question only

PERSONALITY: You are like a trusted friend who happens to be an agriculture expert. Warm, simple language, no jargon.

### CONTEXT
- Farmer location: {state}, India
- Current weather: {weather_summary or 'not provided'}
- Crops mentioned: {active_crops or 'not specified'}

### STRICT RULES
- ONLY answer about: Crops, Plant diseases, Fertilizers, Pesticides, Weather impact on crops, Market prices, Government agricultural schemes
- DO NOT hallucinate or invent details
- DO NOT say things like "Nice to connect from [random place]"
- If unsure: say "I'm not fully certain. Please consult your local KVK or agriculture officer."

Farmer's question: {question}"""

    return system_prompt


# ─── Gemini AI helper (new SDK) ───────────────────────────────────────────────

def _is_quota_error(e: Exception) -> bool:
    """Detect 429 / quota exceeded errors only — NOT 503 service unavailable."""
    msg = str(e).lower()
    return "429" in msg or "quota" in msg or "resource_exhausted" in msg

def _call_model(client, model_name: str, prompt: str, max_tokens: int) -> str:
    """Single model call — raises on error."""
    log.info(f"Gemini call — model: {model_name}")
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=max_tokens,
        ),
    )
    return response.text.strip()


def _call_gemini_sync(prompt: str, max_tokens: int = 1200) -> Optional[str]:
    """
    Call Gemini using new google-genai SDK.
    Iterates the key pool first, then the model chain per key.
    Keys that hit 429/quota are marked exhausted for 1 hour and skipped.
    Returns None only if ALL keys AND models are exhausted.
    """
    if not _GENAI_NEW_SDK:
        raise RuntimeError("google-genai SDK not installed")
    if not GEMINI_KEY_POOL:
        raise RuntimeError("No GEMINI_API_KEY configured")

    allowed, reason = _check_rate_limit()
    if not allowed:
        log.warning(f"Internal rate limit ({reason}) — skipping AI call")
        return None

    models_to_try = [PRIMARY_MODEL, FALLBACK_MODEL, FALLBACK_MODEL_2]

    for key in GEMINI_KEY_POOL:
        if not _key_available(key):
            log.info(f"Skipping exhausted key ...{key[-6:]}")
            continue

        client = genai.Client(api_key=key)
        log.info(f"Trying key ...{key[-6:]}")

        for model_name in models_to_try:
            for attempt in range(2):  # retry once per model on transient errors
                try:
                    result = _call_model(client, model_name, prompt, max_tokens)
                    _record_ai_call()
                    log.info(f"Gemini success — key: ...{key[-6:]}, model: {model_name}")
                    return result

                except Exception as e:
                    log.error(f"Gemini error [key ...{key[-6:]}][{model_name}] attempt {attempt+1}: {type(e).__name__}: {e}")

                    if _is_quota_error(e):
                        # This model's quota exhausted for this key — try next model
                        log.warning(f"Quota/429 on {model_name} (key ...{key[-6:]}) — trying next model")
                        break  # move to next model

                    if attempt == 0:
                        log.info(f"Transient error — retrying {model_name} in 2s")
                        time.sleep(2.0)
                    else:
                        log.warning(f"Transient error persisted on {model_name} — trying next model")
                        break
        else:
            # All models for this key returned quota errors
            _mark_key_exhausted(key)
            log.warning(f"All models exhausted for key ...{key[-6:]} — trying next key in pool")
            continue

    log.error("All keys and models exhausted — returning None for static fallback")
    return None


async def _call_gemini_async(prompt: str, max_tokens: int = 800) -> Optional[str]:
    """Async wrapper for chat endpoint — runs sync call in thread pool."""
    loop = asyncio.get_event_loop()
    return await asyncio.wait_for(
        loop.run_in_executor(None, lambda: _call_gemini_sync(prompt, max_tokens)),
        timeout=30.0
    )


def get_gemini_advisory(disease: str, confidence: float, crop: str,
                         region: str, season: str, language: str, mode: str,
                         severity: str = "Moderate", land_acres: float = 2.0) -> str:
    """
    Get advisory from Gemini with proper fallback.
    ALWAYS returns a non-empty string — never fails visibly to the user.
    """
    if not GEMINI_KEY_POOL or not _GENAI_NEW_SDK:
        log.warning("AI unavailable — using static fallback advisory")
        return _fallback_advisory(disease, crop, mode)

    if mode == "short":
        prompt = build_short_prompt(disease, confidence, crop, region, season, language)
    else:
        prompt = build_detailed_prompt(disease, confidence, crop, region, season, language,
                                       severity=severity, land_acres=land_acres)

    try:
        result = _call_gemini_sync(prompt, max_tokens=1200 if mode == "detailed" else 300)
        if result is None:
            log.info("Gemini returned None (quota/rate limit) — using structured fallback")
            return _fallback_advisory(disease, crop, mode)
        if mode == "detailed":
            return _normalize_detailed_advisory(
                result,
                disease=disease,
                crop=crop,
                severity=severity,
                land_acres=land_acres,
            )
        return result
    except Exception as e:
        log.error(f"Advisory generation failed: {e}")
        return _fallback_advisory(disease, crop, mode)


def _fallback_advisory(disease: str, crop: str, mode: str) -> str:
    """
    Structured fallback advisory — disease-specific where possible.
    NEVER returns an empty string.
    """
    # Disease-specific fallbacks for common diseases
    disease_lower = disease.lower()

    disease_tips = {
        "early blight": {
            "summary": f"Early Blight is a fungal disease caused by Alternaria solani that creates dark brown spots with yellow rings on {crop} leaves. It spreads through infected soil, water splash, and wind. It can cause up to 50% yield loss if not treated early.",
            "chemical": "Apply Mancozeb 75% WP at 2.5 grams per litre of water. Spray every 7 days for 3 consecutive weeks.",
            "organic": "Mix 10ml neem oil with 1 litre water and a few drops of soap. Spray every 5 days on all leaf surfaces.",
        },
        "late blight": {
            "summary": f"Late Blight is caused by Phytophthora infestans — a highly destructive water mold that can destroy a {crop} field within days. It spreads rapidly in cool, wet conditions.",
            "chemical": "Apply Cymoxanil + Mancozeb (72% WP) at 3 grams per litre. Spray every 5-7 days. Start at first sign of infection.",
            "organic": "Spray copper sulphate solution (3 grams per litre water) every 5 days. Remove and burn all infected leaves immediately.",
        },
        "bacterial spot": {
            "summary": f"Bacterial Spot is caused by Xanthomonas bacteria and creates water-soaked spots on {crop} leaves and fruits. It spreads through rain splash, contaminated tools, and infected seeds.",
            "chemical": "Spray copper hydroxide (Kocide) at 3 grams per litre every 7 days. Do not spray in hot afternoon sun.",
            "organic": "Apply neem leaf extract solution or copper soap spray every week. Remove infected plant debris from the field.",
        },
        "powdery mildew": {
            "summary": f"Powdery Mildew is a fungal disease that forms a white powdery coating on {crop} leaves, stems, and flowers. It thrives in dry, warm conditions with high humidity.",
            "chemical": "Apply Sulphur 80% WP at 2-3 grams per litre of water. Spray every 10 days. Or use Hexaconazole 5% EC at 1ml per litre.",
            "organic": "Mix 1 tablespoon baking soda + few drops liquid soap in 1 litre water. Spray on affected areas every 5-7 days.",
        },
        "leaf scorch": {
            "summary": f"Leaf Scorch causes brown, dry edges on {crop} leaves, often due to fungal infection, heat stress, or drought. It can weaken the plant and reduce fruit quality significantly.",
            "chemical": "Apply Carbendazim 50% WP at 1 gram per litre if fungal. Spray twice at 10-day intervals.",
            "organic": "Spray neem oil (5ml per litre) to reduce fungal spread. Ensure adequate and regular watering at the plant base.",
        },
        "mosaic virus": {
            "summary": f"Mosaic Virus is a viral disease spread by aphids and whiteflies that causes mottled yellow-green patterns on {crop} leaves. There is no chemical cure — only prevention and control.",
            "chemical": "Spray Imidacloprid 17.8% SL at 0.5ml per litre to control vector insects (aphids/whiteflies) that spread the virus.",
            "organic": "Spray diluted neem oil (10ml per litre) to repel virus-carrying insects. Remove and destroy all infected plants to stop spread.",
        },
    }

    # Match disease to specific tips
    specific = None
    for key, tips in disease_tips.items():
        if key in disease_lower:
            specific = tips
            break

    # Generic fallback if no specific match
    if not specific:
        specific = {
            "summary": f"{disease} is a plant infection detected on your {crop} crop. It can spread rapidly under humid or warm conditions and cause significant yield loss if not treated early.",
            "chemical": "Apply Mancozeb 75% WP at 2.5 grams per litre of water. Spray every 7 days for 3 consecutive weeks. Spray in the morning or evening.",
            "organic": "Mix 10ml neem oil with 1 litre of water and a few drops of liquid soap. Spray on all leaf surfaces every 5-7 days.",
        }

    if mode == "short":
        return (
            f"Your {crop} has {disease} — remove infected leaves and avoid overhead watering immediately. "
            f"{specific['chemical'].split('.')[0]}. "
            f"Next season, use certified disease-resistant seeds to prevent recurrence."
        )

    return f"""DISEASE_SUMMARY:
{specific['summary']}

SEVERITY:
Moderate

VISIBLE_SYMPTOMS:
- Discolored, spotted, or unusual areas on leaves or stems
- Wilting or yellowing of affected plant parts
- Unusual growth patterns, lesions, or coating on leaves

IMMEDIATE_ACTION_48H:
- Remove and destroy all visibly infected leaves and plant parts immediately
- Avoid watering from above — water only at the base of the plant
- Isolate severely affected plants from healthy ones to prevent spread

CHEMICAL_TREATMENT:
{specific['chemical']}

ORGANIC_ALTERNATIVE:
{specific['organic']}

PREVENTION_NEXT_SEASON:
- Use certified disease-free or disease-resistant seeds for next planting
- Maintain proper spacing between plants for air circulation and sunlight

ECONOMIC_IMPACT:
Potential loss: ₹3,000 - ₹8,000 if not treated within 48 hours

CALL_OFFICER_IF:
If more than 30% of your plants show symptoms, or if the disease spreads to the stem or fruits within 3 days of treatment.

FARMER_TIP:
Act quickly — early treatment saves your crop. Even treating half the field today is better than waiting for tomorrow. You can do this!"""


def _parse_advisory_section(advisory: str, key: str) -> str:
    """Extract a section value from structured advisory (supports multiline values)."""
    sections = _extract_advisory_sections(advisory)
    return sections.get(key, "")


def _extract_advisory_sections(text: str) -> dict[str, str]:
    """Parse advisory text into sections using known keys (markdown-tolerant)."""
    keys = [
        "DISEASE_SUMMARY", "SEVERITY", "VISIBLE_SYMPTOMS",
        "IMMEDIATE_ACTION_48H", "CHEMICAL_TREATMENT", "ORGANIC_ALTERNATIVE",
        "PREVENTION_NEXT_SEASON", "ECONOMIC_IMPACT", "CALL_OFFICER_IF", "FARMER_TIP",
    ]
    sections: dict[str, str] = {}
    current_key: Optional[str] = None
    buffer: list[str] = []

    def flush() -> None:
        if current_key:
            sections[current_key] = "\n".join(buffer).strip()

    for raw in (text or "").splitlines():
        stripped = raw.strip().strip("*")
        matched_key = next((k for k in keys if stripped.upper().startswith(k + ":")), None)
        if matched_key:
            flush()
            current_key = matched_key
            buffer = []
            inline = stripped[len(matched_key) + 1 :].strip().strip("[]")
            if inline:
                buffer.append(inline)
            continue
        if current_key:
            clean = raw.strip().strip("*")
            if clean:
                buffer.append(clean)

    flush()
    return sections


def _normalize_detailed_advisory(advisory: str, disease: str, crop: str,
                                 severity: str = "Moderate", land_acres: float = 2.0) -> str:
    """Ensure detailed advisory always contains all required structured sections."""
    primary = _extract_advisory_sections(advisory)
    fallback = _extract_advisory_sections(_fallback_advisory(disease, crop, "detailed"))

    required = [
        "DISEASE_SUMMARY", "SEVERITY", "VISIBLE_SYMPTOMS", "IMMEDIATE_ACTION_48H",
        "CHEMICAL_TREATMENT", "ORGANIC_ALTERNATIVE", "PREVENTION_NEXT_SEASON",
        "ECONOMIC_IMPACT", "CALL_OFFICER_IF", "FARMER_TIP",
    ]

    merged: dict[str, str] = {}
    for key in required:
        value = (primary.get(key, "") or "").strip()
        if not value:
            value = (fallback.get(key, "") or "").strip()
        merged[key] = value

    if not merged["SEVERITY"]:
        merged["SEVERITY"] = severity

    if not merged["ECONOMIC_IMPACT"]:
        merged["ECONOMIC_IMPACT"] = (
            f"Potential loss: ₹3,000 - ₹8,000 if not treated within 48 hours for {land_acres} acres"
        )

    bullet_keys = {"VISIBLE_SYMPTOMS", "IMMEDIATE_ACTION_48H", "PREVENTION_NEXT_SEASON"}
    lines: list[str] = []

    for key in required:
        lines.append(f"{key}:")
        value = merged[key]
        if key in bullet_keys:
            items = [
                v.strip().lstrip("- ").strip()
                for v in value.splitlines()
                if v.strip()
            ]
            for item in items:
                lines.append(f"- {item}")
            if not items:
                lines.append("- Data not available")
        else:
            lines.append(value or "Data not available")
        lines.append("")

    return "\n".join(lines).strip()


def extract_severity(advisory: str) -> str:
    for line in advisory.splitlines():
        if line.strip().upper().startswith("SEVERITY:"):
            val = line.split(":", 1)[1].strip().lower()
            if "severe" in val or "high" in val or "critical" in val:
                return "high"
            if "moderate" in val or "medium" in val:
                return "medium"
            return "low"
    text = advisory.lower()
    if any(w in text for w in ["severe", "critical", "dangerous", "advanced"]):
        return "high"
    if any(w in text for w in ["moderate", "medium", "significant"]):
        return "medium"
    return "low"

# ─── Schemas ──────────────────────────────────────────────────────────────────
class AdvisoryRequest(BaseModel):
    disease_name: str
    confidence:   float
    crop:         str
    region:       str = "India"
    season:       str = "Kharif"
    language:     str = "en"
    mode:         str = "detailed"

class ChatRequest(BaseModel):
    question:       str
    language:       str = "en"
    state:          str = "India"
    weather_summary: str = ""
    active_crops:   str = ""

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    config_state = get_config_status()
    return {
        "status": "ok",
        "mock_mode": USE_MOCK_MODEL,
        "model_path": MODEL_PATH,
        "config_source": "database",
        "ai_sdk": "google-genai (new)" if _GENAI_NEW_SDK else "unavailable",
        "primary_model": PRIMARY_MODEL,
        "daily_ai_calls": _daily_count,
        "daily_limit": RATE_LIMIT_RPD,
        "gemini_key_pool": len(GEMINI_KEY_POOL),
        "gemini_keys_available": sum(1 for k in GEMINI_KEY_POOL if _key_available(k)),
        "services": {
            "gemini":      bool(GEMINI_KEY_POOL) and _GENAI_NEW_SDK,
            "weather":     config_state["WEATHER_API_KEY"],
            "agmarknet":   config_state["AGMARKNET_KEY"],
            "elevenlabs":  bool(ELEVENLABS_API_KEY),
            "schemes":     bool(GOV_SCHEMES_URL) or True,
        }
    }

@app.get("/api/config/status")
def config_status():
    return {
        "config_source": "database",
        "keys": get_config_status(),
    }

@app.post("/api/cache/clear")
def clear_cache():
    """Wipe the in-memory weather + mandi cache instantly."""
    global _cache
    cleared = len(_cache)
    _cache.clear()
    log.info(f"Cache cleared — {cleared} entries removed")
    return {"cleared_entries": cleared, "status": "ok"}

@app.post("/api/db/purge-stale")
def purge_stale_db():
    """Delete obsolete config keys (e.g. WHISPER_*) from the SQLite DB."""
    removed = purge_stale_keys()
    log.info(f"DB purge — removed stale keys: {removed}")
    return {"removed_keys": removed, "status": "ok"}

@app.get("/classes")
def get_classes():
    return {
        "count": len(CLASS_NAMES),
        "classes": [
            {"id": i, "raw": c, "crop": parse_class_name(c)[0],
             "disease": parse_class_name(c)[1], "healthy": parse_class_name(c)[2]}
            for i, c in enumerate(CLASS_NAMES)
        ],
    }

@app.post("/analyze")
async def analyze_image(
    file:     UploadFile = File(...),
    region:   str = Form("India"),
    season:   str = Form("Kharif"),
    language: str = Form("en"),
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image (JPEG or PNG).")

    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(413, "Image too large. Max 10 MB.")

    is_crop, crop_error = validate_crop_image(image_bytes)
    if not is_crop:
        raise HTTPException(400, crop_error)

    try:
        raw_class, confidence, crop, disease, is_healthy = predict_disease(image_bytes)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        log.exception("Prediction failed")
        raise HTTPException(500, f"Prediction error: {e}")

    advisory_short  = get_gemini_advisory(disease, confidence, crop, region, season, language, "short")

    # Pre-estimate severity from ML confidence for use in the Gemini prompt
    estimated_severity = _estimate_severity_from_confidence(confidence)
    advisory_detail = get_gemini_advisory(disease, confidence, crop, region, season, language, "detailed",
                                          severity=estimated_severity)

    # Ensure advisory is NEVER empty
    if not advisory_short or not advisory_short.strip():
        advisory_short = _fallback_advisory(disease, crop, "short")
    if not advisory_detail or not advisory_detail.strip():
        advisory_detail = _fallback_advisory(disease, crop, "detailed")

    severity = "low" if is_healthy else extract_severity(advisory_detail)

    # Parse ECONOMIC_IMPACT section from advisory
    economic_impact = _parse_advisory_section(advisory_detail, "ECONOMIC_IMPACT")

    # Build weather advisory using region/season/disease context
    disease_lower = disease.lower()
    if any(w in disease_lower for w in ["blight", "mildew", "rust", "rot", "mold", "spot"]):
        weather_advisory = (
            f"Fungal diseases like {disease} spread rapidly in humid weather (>80% humidity) and rainy conditions. "
            f"Avoid spraying chemicals before expected rain. Check the Weather tab for {region} conditions before scheduling treatment."
        )
    elif any(w in disease_lower for w in ["virus", "mosaic", "bacterial"]):
        weather_advisory = (
            f"Bacterial/viral spread is worsened by wet, windy weather. "
            f"Check the Weather tab for current {region} conditions to time your spraying window correctly."
        )
    else:
        weather_advisory = (
            f"Monitor {region} weather conditions before applying treatment. "
            f"Spray in early morning or evening — avoid hot afternoon sun. Check the Weather tab for a 5-day forecast."
        )

    # Build mandi advisory for the crop
    mandi_advisory = (
        f"Treating {disease} on {crop} promptly helps preserve crop market value. "
        f"Check live {crop} prices in the Market tab to time your sale. "
        f"Healthy {crop} fetches a significant premium over diseased produce."
    )

    log.info(f"Analysis complete — disease: {disease}, crop: {crop}, severity: {severity}, ai_used: {bool(GEMINI_API_KEY)}")

    result_payload = {
        "raw_class":           raw_class,
        "disease":             disease,
        "crop":                crop,
        "confidence":          round(confidence * 100, 1),
        "is_healthy":          is_healthy,
        "severity":            severity,
        "advisory_short":      advisory_short,
        "advisory_detail":     advisory_detail,
        "economic_impact":     economic_impact,
        "red_alert_triggered": severity == "high",
        "weather_advisory":    weather_advisory,
        "mandi_advisory":      mandi_advisory,
    }

    # Persist image + result to scan history (non-blocking — don't fail the request)
    try:
        scan_id = _save_scan(image_bytes, result_payload)
        result_payload["scan_id"] = scan_id
    except Exception as hist_err:
        log.warning(f"Could not save scan to history: {hist_err}")

    return result_payload

@app.post("/advisory")
def get_advisory(req: AdvisoryRequest):
    advisory = get_gemini_advisory(
        req.disease_name, req.confidence, req.crop,
        req.region, req.season, req.language, req.mode,
    )

    # GUARANTEE non-empty advisory
    if not advisory or not advisory.strip():
        advisory = _fallback_advisory(req.disease_name, req.crop, req.mode)

    if req.mode == "detailed":
        advisory = _normalize_detailed_advisory(
            advisory,
            disease=req.disease_name,
            crop=req.crop,
            severity="Moderate",
            land_acres=2.0,
        )

    return {
        "disease":  req.disease_name,
        "crop":     req.crop,
        "language": req.language,
        "mode":     req.mode,
        "advisory": advisory,
        "severity": extract_severity(advisory),
    }

# ─── Secure Gemini Chat Proxy ─────────────────────────────────────────────────

# ─── Scan History endpoints ───────────────────────────────────────────────────
@app.get("/api/scan-history")
def list_scan_history(limit: int = Query(default=30, le=100)):
    """Return the most recent scan records (no image bytes — use /image for that)."""
    return {"scans": _get_history(limit)}

@app.get("/api/scan-history/{scan_id}/image")
def get_scan_image(scan_id: str):
    """Serve the original image for a past scan."""
    # Validate ID is a UUID to prevent path traversal
    try:
        uuid.UUID(scan_id)
    except ValueError:
        raise HTTPException(400, "Invalid scan ID format.")
    img_path = _UPLOADS_DIR / f"{scan_id}.jpg"
    if not img_path.exists():
        raise HTTPException(404, "Image not found.")
    return FileResponse(str(img_path), media_type="image/jpeg")

@app.delete("/api/scan-history/{scan_id}")
def delete_scan(scan_id: str):
    """Delete a single scan record and its image."""
    try:
        uuid.UUID(scan_id)
    except ValueError:
        raise HTTPException(400, "Invalid scan ID format.")
    img_path = _UPLOADS_DIR / f"{scan_id}.jpg"
    if img_path.exists():
        img_path.unlink()
    conn = _history_conn()
    try:
        conn.execute("DELETE FROM scan_history WHERE id = ?", (scan_id,))
        conn.commit()
    finally:
        conn.close()
    return {"deleted": scan_id}

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    """
    Secure chat endpoint — Gemini API key never leaves the server.
    Frontend sends questions, backend adds context and calls Gemini.
    If AI is unavailable (quota/error), returns a graceful friendly message.
    """
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty.")

    if not GEMINI_KEY_POOL or not _GENAI_NEW_SDK:
        return {
            "response": "⚠️ AI is temporarily unavailable. Please try again in a few seconds, or consult your local KVK for immediate assistance.",
            "timestamp": __import__("datetime").datetime.now().isoformat(),
            "fallback": True,
        }

    prompt = build_chat_prompt(
        req.question, req.state, req.weather_summary, req.active_crops, req.language
    )

    try:
        result = await _call_gemini_async(prompt, max_tokens=800)

        if result is None:
            # Quota / rate limit — graceful response
            log.warning("Chat: AI quota reached — returning graceful fallback message")
            return {
                "response": "⚠️ AI is busy right now. Please try again in a few seconds. For urgent crop advice, contact your nearest KVK (Krishi Vigyan Kendra).",
                "timestamp": __import__("datetime").datetime.now().isoformat(),
                "fallback": True,
            }

        return {
            "response":  result,
            "timestamp": __import__("datetime").datetime.now().isoformat(),
            "fallback": False,
        }

    except asyncio.TimeoutError:
        log.error("Chat: Gemini timed out")
        return {
            "response": "⚠️ AI took too long to respond. Please try again.",
            "timestamp": __import__("datetime").datetime.now().isoformat(),
            "fallback": True,
        }
    except Exception as e:
        log.error(f"Chat endpoint error: {type(e).__name__}: {e}")
        return {
            "response": "⚠️ AI is temporarily unavailable. Please try again in a few seconds.",
            "timestamp": __import__("datetime").datetime.now().isoformat(),
            "fallback": True,
        }


# ─── Whisper transcription proxy ────────────────────────────────────────────
@app.post("/api/transcribe")
async def transcribe_endpoint(
    file: UploadFile = File(...),
    language: str = Form("en"),
):
    if not ELEVENLABS_API_KEY:
        raise HTTPException(503, "ElevenLabs STT not configured. Set ELEVENLABS_API_KEY in backend .env.")

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(400, "Audio file is empty.")

    # ElevenLabs language code (ISO 639-1)
    lang_code = language.split("-")[0] if language else "en"

    content_type = file.content_type or "audio/webm"
    files = {"file": (file.filename or "audio.webm", audio_bytes, content_type)}
    data = {
        "model_id": ELEVENLABS_STT_MODEL,
        "language_code": lang_code,
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                ELEVENLABS_STT_URL,
                headers={"xi-api-key": ELEVENLABS_API_KEY},
                files=files,
                data=data,
            )
    except httpx.TimeoutException:
        raise HTTPException(504, "ElevenLabs STT API timed out.")

    if resp.status_code == 401:
        detail = resp.json().get("detail", {})
        if isinstance(detail, dict) and "missing_permissions" in detail.get("status", ""):
            raise HTTPException(403, "ElevenLabs API key is missing the 'speech_to_text' permission. "
                                     "Generate a new key at elevenlabs.io/app/settings/api-keys with that scope enabled.")
        raise HTTPException(401, f"ElevenLabs authentication failed: {resp.text[:200]}")

    if resp.status_code != 200:
        detail = resp.text[:400]
        raise HTTPException(resp.status_code, f"ElevenLabs STT failed: {detail}")

    payload = resp.json()
    # ElevenLabs returns {"text": "...", "words": [...], "language_code": "..."}
    transcript = payload.get("text") or ""
    if not transcript.strip():
        raise HTTPException(502, "ElevenLabs STT returned an empty transcription.")

    return {"text": transcript.strip()}


# ─── Schemes proxy ───────────────────────────────────────────────────────────
@app.get("/api/schemes")
async def schemes_endpoint():
    if GOV_SCHEMES_URL:
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.get(GOV_SCHEMES_URL)
            if resp.status_code == 200:
                payload = resp.json()
                records = payload.get("records") or payload.get("data") or payload
                if isinstance(records, list) and records and isinstance(records[0], dict):
                    return {
                        "source": "gov.in",
                        "data": records,
                        "lastUpdated": __import__("datetime").datetime.now().isoformat(),
                    }
        except Exception as e:
            log.warning(f"Gov schemes fetch failed: {e}")

    return {
        "source": "local-fallback",
        "data": DEFAULT_SCHEMES,
        "lastUpdated": __import__("datetime").datetime.now().isoformat(),
    }


# ─── Weather Proxy (lat/lon or city) ──────────────────────────────────────────
@app.get("/api/weather")
async def weather_endpoint(
    lat:  Optional[float] = Query(None),
    lon:  Optional[float] = Query(None),
    city: Optional[str]   = Query(None),
):
    """
    Proxy OpenWeatherMap — keeps WEATHER_API_KEY server-side.
    Accepts lat/lon (preferred) or city name.
    5-minute cache to avoid rate limits.
    """
    if not WEATHER_API_KEY:
        raise HTTPException(503, "Weather service not configured. Set WEATHER_API_KEY in backend .env.")

    if lat is not None and lon is not None:
        cache_key  = f"weather_ll_{lat:.2f}_{lon:.2f}"
        query_part = f"lat={lat}&lon={lon}"
    elif city:
        cache_key  = f"weather_city_{city.lower().strip()}"
        query_part = f"q={city}"
    else:
        raise HTTPException(400, "Provide lat+lon or city parameter.")

    cached = cache_get(cache_key, ttl=300)
    if cached:
        log.info(f"Weather cache hit: {cache_key}")
        return cached

    base = "https://api.openweathermap.org/data/2.5"
    async with httpx.AsyncClient(timeout=8.0) as client:
        try:
            cur_resp  = await client.get(f"{base}/weather?{query_part}&appid={WEATHER_API_KEY}&units=metric")
            fore_resp = await client.get(f"{base}/forecast?{query_part}&appid={WEATHER_API_KEY}&units=metric")
        except httpx.TimeoutException:
            raise HTTPException(504, "Weather API timed out.")

    if cur_resp.status_code != 200:
        err = cur_resp.json().get("message", "Weather data unavailable")
        raise HTTPException(cur_resp.status_code, err)

    cur  = cur_resp.json()
    fore = fore_resp.json() if fore_resp.status_code == 200 else {"list": []}

    # Hourly (next 6 slots = 18h)
    hourly = []
    for item in fore["list"][:6]:
        d = __import__("datetime").datetime.fromtimestamp(item["dt"])
        hourly.append({
            "time":      d.strftime("%H:%M"),
            "temp":      round(item["main"]["temp"]),
            "condition": item["weather"][0]["main"],
            "rain":      round((item.get("pop") or 0) * 100),
            "icon":      item["weather"][0]["icon"],
        })

    # 5-day daily (skip today)
    seen, daily = set(), []
    today = __import__("datetime").date.today().isoformat()
    for item in fore["list"]:
        d = __import__("datetime").datetime.fromtimestamp(item["dt"])
        dk = d.date().isoformat()
        if dk == today or dk in seen or len(daily) >= 5:
            continue
        seen.add(dk)
        day_items = [x for x in fore["list"]
                     if __import__("datetime").datetime.fromtimestamp(x["dt"]).date().isoformat() == dk]
        temps = [x["main"]["temp"] for x in day_items]
        mid   = day_items[len(day_items)//2]
        daily.append({
            "day":       d.strftime("%A") if len(daily) > 0 else "Tomorrow",
            "condition": mid["weather"][0]["main"],
            "high":      round(max(temps)),
            "low":       round(min(temps)),
            "icon":      mid["weather"][0]["icon"],
        })

    result = {
        "location":      cur["name"],
        "country":       cur.get("sys", {}).get("country", "IN"),
        "temperature":   round(cur["main"]["temp"]),
        "feels_like":    round(cur["main"]["feels_like"]),
        "condition":     cur["weather"][0]["main"],
        "description":   cur["weather"][0]["description"],
        "humidity":      cur["main"]["humidity"],
        "windSpeed":     round(cur["wind"]["speed"] * 3.6),
        "pressure":      cur["main"]["pressure"],
        "visibility":    round(cur.get("visibility", 0) / 1000, 1),
        "icon":          cur["weather"][0]["icon"],
        "lat":           cur["coord"]["lat"],
        "lon":           cur["coord"]["lon"],
        "hourlyForecast":  hourly,
        "weeklyForecast":  daily,
        "lastUpdated":   __import__("datetime").datetime.now().isoformat(),
    }

    cache_set(cache_key, result)
    log.info(f"Weather fetched for {result['location']}")
    return result


# ─── Mandi (Agmarknet) Proxy ──────────────────────────────────────────────────
CITY_COORDS: dict[str, tuple[float, float]] = {
    "Thiruvananthapuram": (8.5241, 76.9366),
    "Kochi": (9.9312, 76.2673),
    "Kozhikode": (11.2588, 75.7804),
    "Mumbai": (19.0760, 72.8777),
    "Pune": (18.5204, 73.8567),
    "Delhi": (28.6139, 77.2090),
    "Bengaluru": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707),
    "Hyderabad": (17.3850, 78.4867),
    "Kolkata": (22.5726, 88.3639),
    "Ahmedabad": (23.0225, 72.5714),
    "Jaipur": (26.9124, 75.7873),
    "Lucknow": (26.8467, 80.9462),
    "Patna": (25.5941, 85.1376),
    "Bhopal": (23.2599, 77.4126),
    "Chandigarh": (30.7333, 76.7794),
    "Guwahati": (26.1445, 91.7362),
    "Ranchi": (23.3441, 85.3096),
    "Bhubaneswar": (20.2961, 85.8245),
    "Raipur": (21.2514, 81.6296),
    "Amritsar": (31.6340, 74.8723),
    "Ludhiana": (30.9010, 75.8573),
    "Nagpur": (21.1458, 79.0882),
    "Visakhapatnam": (17.6868, 83.2185),
    "Coimbatore": (11.0168, 76.9558),
    "Madurai": (9.9252, 78.1198),
    "Surat": (21.1702, 72.8311),
    "Vadodara": (22.3072, 73.1812),
    "Shimla": (31.1048, 77.1734),
    "Dehradun": (30.3165, 78.0322),
}

@app.get("/api/mandi")
async def mandi_endpoint(
    state:     str = Query("Maharashtra"),
    commodity: str = Query("Tomato"),
    limit:     int = Query(20, ge=1, le=100),
):
    """
    Proxy Agmarknet (data.gov.in) — keeps API key server-side.
    Falls back to state-specific mock data if key not set or API fails.
    Returns clean format with trend calculation.
    """
    cache_key = f"mandi_{state.lower()}_{commodity.lower()}"
    cached = cache_get(cache_key, ttl=300)
    if cached:
        log.info(f"Mandi cache hit: {cache_key}")
        return cached

    AGMARKNET_BASE = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"

    if AGMARKNET_KEY:
        # urllib.request preserves literal [ ] in the query string;
        # httpx percent-encodes them (%5B/%5D) which data.gov.in rejects.
        url = (
            f"{AGMARKNET_BASE}?api-key={AGMARKNET_KEY}&format=json&limit={limit}"
            f"&filters[state.keyword]={state}&filters[commodity]={commodity}"
        )
        def _fetch_agmarknet():
            req = urllib.request.Request(url, headers={"User-Agent": "KrishiMitra/1.0"})
            with urllib.request.urlopen(req, timeout=25) as resp:
                return json.loads(resp.read())

        try:
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(None, _fetch_agmarknet)

            if data.get("records"):
                def calc_trend(records):
                    if len(records) < 2:
                        return "stable"
                    try:
                        p1 = float(records[0].get("modal_price", 0))
                        p2 = float(records[1].get("modal_price", 0))
                        if p1 > p2 * 1.02:
                            return "up"
                        if p1 < p2 * 0.98:
                            return "down"
                    except Exception:
                        pass
                    return "stable"

                records = data["records"]
                trend   = calc_trend(records)
                cleaned = []
                for r in records:
                    cleaned.append({
                        "mandi":      r.get("market", "—"),
                        "district":   r.get("district", state),
                        "commodity":  r.get("commodity", commodity),
                        "variety":    r.get("variety", "Common"),
                        "minPrice":   r.get("min_price"),
                        "maxPrice":   r.get("max_price"),
                        "modalPrice": r.get("modal_price"),
                        "price":      r.get("modal_price"),
                        "trend":      trend,
                        "date":       r.get("arrival_date"),
                        "state":      r.get("state", state),
                    })

                result = {
                    "data":        cleaned,
                    "source":      "Agmarknet / data.gov.in",
                    "lastUpdated": __import__("datetime").datetime.now().isoformat(),
                    "count":       len(cleaned),
                }
                cache_set(cache_key, result)
                return result

        except Exception as e:
            log.error(f"Agmarknet fetch error [{type(e).__name__}]: {e}")

    # ── Fallback mock ──
    log.warning(f"Using mock mandi data for {state}/{commodity}")
    state_prices = {        "Maharashtra": {"Tomato": [800,1200], "Onion": [500,900], "Wheat": [2100,2400], "Potato": [1100,1500]},
        "Punjab":      {"Wheat": [2100,2500], "Rice": [1800,2200], "Maize": [1700,2000], "Potato": [1000,1400]},
        "Uttar Pradesh": {"Potato": [1200,1600], "Wheat": [2000,2300], "Sugarcane": [350,420], "Onion": [600,1000]},
        "Karnataka":   {"Tomato": [900,1400], "Onion": [600,1000], "Groundnut": [5000,6500], "Coconut": [1400,1900]},
        "Kerala":      {"Coconut": [1500,2000], "Banana": [2000,3500], "Rice": [2800,3500], "Tomato": [1200,1800]},
        "West Bengal": {"Rice": [2200,2800], "Potato": [900,1300], "Banana": [1800,2800], "Tomato": [700,1100]},
        "Tamil Nadu":  {"Rice": [2500,3200], "Tomato": [800,1400], "Onion": [500,900], "Banana": [1500,2500]},
        "Gujarat":     {"Groundnut": [4800,6000], "Cotton": [5500,6800], "Wheat": [2000,2400], "Tomato": [700,1200]},
        "Rajasthan":   {"Wheat": [2000,2400], "Onion": [400,800], "Tomato": [600,1100], "Maize": [1600,1900]},
        "Madhya Pradesh": {"Soyabean": [3800,4600], "Wheat": [1900,2300], "Maize": [1600,2000], "Tomato": [600,1100]},
    }
    prices = state_prices.get(state, {}).get(commodity, [1000, 2000])
    mid    = (prices[0] + prices[1]) // 2

    mock_result = {
        "data": [{
            "mandi":      f"{state} Main Mandi",
            "district":   state,
            "commodity":  commodity,
            "variety":    "Local",
            "minPrice":   prices[0],
            "maxPrice":   prices[1],
            "modalPrice": mid,
            "price":      mid,
            "trend":      "stable",
            "date":       __import__("datetime").date.today().isoformat(),
            "state":      state,
        }],
        "source":      "Estimated" if AGMARKNET_KEY else "Estimated (Agmarknet key not configured)",
        "lastUpdated": __import__("datetime").datetime.now().isoformat(),
        "count":       1,
    }
    cache_set(cache_key, mock_result)
    return mock_result


# ─── Fast2SMS — Send SMS proxy ────────────────────────────────────────────────
# Fast2SMS free tier (route "v") only delivers to numbers registered on the
# Fast2SMS test panel. Switch route to "q" after completing DLT registration
# with TRAI and obtaining a DLT Template ID for production use.

class SMSRequest(BaseModel):
    numbers:  list[str]      # E.g. ["9876543210", "9123456789"]
    message:  str
    language: str = "Hindi"  # Informational; Fast2SMS uses Unicode automatically

@app.post("/api/send-sms")
async def send_sms_endpoint(req: SMSRequest):
    """
    Send bulk SMS via Fast2SMS.
    API key is read from FAST2SMS_API_KEY env var — never exposed to frontend.

    NOTE: Free tier (route="v") only works for numbers whitelisted in the
    Fast2SMS sender dashboard. For production, switch to route="q" and supply
    a DLT-approved template ID in the 'message_id' param.
    """
    api_key = os.getenv("FAST2SMS_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(503, "SMS service not configured. Set FAST2SMS_API_KEY in backend .env.")

    if not req.numbers:
        raise HTTPException(400, "At least one recipient number is required.")

    if not req.message.strip():
        raise HTTPException(400, "Message cannot be empty.")

    if len(req.message) > 160:
        raise HTTPException(400, "Message exceeds 160 characters.")

    # Sanitise: keep only 10-digit numbers
    clean_numbers = [n.strip() for n in req.numbers if n.strip().isdigit() and len(n.strip()) == 10]
    if not clean_numbers:
        raise HTTPException(400, "No valid 10-digit numbers provided.")

    numbers_str = ",".join(clean_numbers)

    params = {
        "variables_values": req.message,
        "route":            "v",          # "v" = promotional free tier; "q" = transactional (needs DLT)
        "numbers":          numbers_str,
        # Uncomment and set for DLT transactional route:
        # "message_id": "YOUR_DLT_TEMPLATE_ID",
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                "https://www.fast2sms.com/dev/bulkV2",
                headers={"authorization": api_key},
                params=params,
            )
    except httpx.TimeoutException:
        log.error("Fast2SMS API timed out")
        raise HTTPException(504, "SMS gateway timed out. Please retry.")
    except Exception as e:
        log.error(f"Fast2SMS request error: {e}")
        raise HTTPException(502, f"SMS gateway error: {str(e)}")

    try:
        payload = resp.json()
    except Exception:
        log.error(f"Fast2SMS non-JSON response: {resp.text[:300]}")
        raise HTTPException(502, "Unexpected response from SMS gateway.")

    if resp.status_code != 200 or not payload.get("return", False):
        error_msg = payload.get("message", [resp.text[:200]])
        if isinstance(error_msg, list):
            error_msg = " ".join(error_msg)
        log.warning(f"Fast2SMS error: {error_msg}")
        raise HTTPException(400, f"SMS send failed: {error_msg}")

    message_ids = payload.get("message_id") or payload.get("request_id") or []
    if isinstance(message_ids, str):
        message_ids = [message_ids]

    log.info(f"Fast2SMS: sent to {len(clean_numbers)} numbers, ids={message_ids}")
    return {
        "success":     True,
        "count":       len(clean_numbers),
        "message_ids": message_ids,
    }
