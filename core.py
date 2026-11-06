"""Core (non-UI) logic for the Smart Classroom Energy System.

Kept separate from app.py so it can be tested without Streamlit.

Pipeline
--------
1. An ML model predicts whether the classroom is OCCUPIED from environmental
   sensor readings (light, temperature, humidity, CO2, hour of day).
2. A transparent rule turns that into a lighting recommendation:
       lights ON  = occupied AND room is dark (light < dark threshold)
       lights OFF = otherwise
"""

from pathlib import Path

import joblib
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"
REPORT_DIR = PROJECT_DIR / "reports"

PREFERRED_MODEL_NAME = "smart_classroom_final_model.joblib"

# Default "room is dark" threshold in the SAME units as the dataset's light
# column. Calibrate it for your sensor/dataset (train_model.py stores the value
# it used inside the model file).
DEFAULT_DARK_THRESHOLD = 60.0


def find_model_path():
    """Return the model file to load, or None.

    Order: models/<preferred>, ./<preferred>, then any other *.joblib in
    models/ or the project root (newest first, 'backup' files ignored).
    """
    for folder in (MODEL_DIR, PROJECT_DIR):
        candidate = folder / PREFERRED_MODEL_NAME
        if candidate.is_file():
            return candidate

    others = []
    for folder in (MODEL_DIR, PROJECT_DIR):
        if folder.is_dir():
            others += [
                p for p in folder.glob("*.joblib")
                if "backup" not in p.name.lower()
            ]
    others.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return others[0] if others else None


def load_artifact(path):
    """Load and validate a saved model artifact (a dict)."""
    saved = joblib.load(path)
    if not isinstance(saved, dict) or "model" not in saved:
        raise ValueError(
            "Unexpected model file format: expected a dict with a 'model' key."
        )
    model = saved["model"]
    features = list(saved.get("features") or
                    getattr(model, "feature_names_in_", []))
    if not features:
        raise ValueError("The model file does not list its input features.")
    return {
        "model": model,
        "features": features,
        "threshold": float(saved.get("threshold", 0.5)),
        "model_name": saved.get("model_name", type(model).__name__),
        "target": saved.get("target", "motion_detected"),
        "dark_threshold": float(
            saved.get("dark_threshold", DEFAULT_DARK_THRESHOLD)),
    }


def predict_occupancy(artifact, values):
    """Return (probability_occupied, occupied_bool) for one reading.

    `values` maps feature name -> number. Raises KeyError with a clear message
    if the model needs a feature that was not supplied.
    """
    features = artifact["features"]
    missing = [f for f in features if f not in values]
    if missing:
        raise KeyError(f"Missing input value(s) for: {', '.join(missing)}")

    row = pd.DataFrame([[values[f] for f in features]], columns=features)
    model = artifact["model"]
    classes = list(model.classes_)
    positive = 1 if 1 in classes else classes[-1]
    probability = float(model.predict_proba(row)[0][classes.index(positive)])
    return probability, probability >= artifact["threshold"]


def lighting_decision(occupied, light_level, dark_threshold):
    """Return ('ON'|'OFF', explanation)."""
    if not occupied:
        return "OFF", "The model predicts the classroom is empty."
    if light_level >= dark_threshold:
        return "OFF", (
            "The classroom is predicted to be occupied, but the measured "
            "light is already above the darkness threshold."
        )
    return "ON", (
        "The classroom is predicted to be occupied and the measured light is "
        "below the darkness threshold."
    )


def energy_estimate(power_w, hours_per_day, days_per_month, rate_per_kwh):
    """Return (daily_kwh, monthly_kwh, monthly_cost)."""
    daily = power_w * hours_per_day / 1000
    monthly = daily * days_per_month
    return daily, monthly, monthly * rate_per_kwh
