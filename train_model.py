"""Train and evaluate the Smart Classroom occupancy model on a REAL dataset.

Usage (terminal or Jupyter):
    python train_model.py --data data/your_kaggle_file.csv
    python train_model.py --data data/your_kaggle_file.csv --target-col pir --dark-threshold 60

What it does
------------
* Loads the dataset and maps its columns to: light_level, temperature,
  humidity, co2, hour and the occupancy/motion target.
* Splits the data 80/20 (stratified) into train and test sets.
* Compares Logistic Regression and a Decision Tree with 5-fold cross-validation
  on the TRAINING data only, tunes the decision threshold, then evaluates once
  on the untouched test set.
* Creates the lighting label (ON/OFF) with a documented rule and evaluates it.
* Writes, to reports/: metrics.json, classification_report.txt,
  confusion_matrix.png, model_comparison.csv, feature_importance.png,
  dataset_summary.txt
* Saves the model to models/smart_classroom_final_model.joblib
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, confusion_matrix, f1_score,
    precision_recall_fscore_support, roc_auc_score,
)
from sklearn.model_selection import (  # noqa: E402
    StratifiedKFold, cross_val_predict, train_test_split,
)
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

RANDOM_STATE = 42
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "smart_classroom_final_model.joblib"
REPORT_DIR = PROJECT_DIR / "reports"

FEATURES = ["light_level", "temperature", "humidity", "co2", "hour"]
TARGET = "motion_detected"

# Accepted column names (after lower-casing and replacing spaces with "_").
ALIASES = {
    "light_level": ["light_level", "light", "lux", "light_level_lux",
                    "illuminance"],
    "temperature": ["temperature", "temp"],
    "humidity": ["humidity", "hum", "relative_humidity"],
    "co2": ["co2", "co2_ppm"],
    "hour": ["hour", "hour_of_day"],
    TARGET: ["motion_detected", "pir", "motion", "occupancy", "occupied"],
}
TIMESTAMP_COLUMNS = ["timestamp", "datetime", "date_time", "date", "time", "ts"]


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def load_dataset(path, target_col=None):
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    print(f"Loaded {path} -> {df.shape[0]:,} rows, {df.shape[1]} columns")
    print("Columns found:", list(df.columns))

    mapping = {}
    for canonical, names in ALIASES.items():
        if canonical == TARGET and target_col:
            names = [target_col.strip().lower().replace(" ", "_")]
        found = next((n for n in names if n in df.columns), None)
        if found:
            mapping[canonical] = found

    # Derive the hour from a timestamp column if there is no hour column.
    if "hour" not in mapping:
        ts_col = next((c for c in TIMESTAMP_COLUMNS if c in df.columns), None)
        if ts_col is not None:
            series = df[ts_col]
            if pd.api.types.is_numeric_dtype(series):
                parsed = pd.to_datetime(series, unit="s", errors="coerce")
            else:
                parsed = pd.to_datetime(series, errors="coerce")
            df["hour"] = parsed.dt.hour
            mapping["hour"] = "hour"
            print(f"Derived 'hour' from timestamp column '{ts_col}'.")

    missing = [c for c in FEATURES + [TARGET] if c not in mapping]
    if missing:
        sys.exit(
            "\nCould not find these columns: " + ", ".join(missing) +
            "\nColumns in your file: " + ", ".join(df.columns) +
            "\nRename them in the CSV, add names to ALIASES in this script, "
            "or pass --target-col for the occupancy/motion column."
        )

    out = pd.DataFrame({c: df[mapping[c]] for c in FEATURES + [TARGET]})
    out = out.apply(pd.to_numeric, errors="coerce")
    before = len(out)
    out = out.dropna()
    print(f"Dropped {before - len(out):,} rows with missing/non-numeric values.")
    out[TARGET] = (out[TARGET] > 0).astype(int)  # PIR/occupancy -> 0/1
    return out.reset_index(drop=True), mapping


# --------------------------------------------------------------------------
# Metrics helpers
# --------------------------------------------------------------------------
def block_metrics(y_true, y_pred, class_names, y_score=None):
    """Accuracy, confusion matrix and per-class precision/recall/F1."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0)
    majority = max(np.mean(y_true), 1 - np.mean(y_true))
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "majority_class_baseline_accuracy": float(majority),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                   zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "per_class": {
            class_names[i]: {
                "precision": float(p[i]), "recall": float(r[i]),
                "f1": float(f[i]), "support": int(s[i]),
            } for i in (0, 1)
        },
    }
    if y_score is not None and len(set(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    return out


def report_text(title, m):
    lines = [title, "=" * len(title),
             f"Accuracy: {m['accuracy']:.4f}  "
             f"(majority-class baseline: "
             f"{m['majority_class_baseline_accuracy']:.4f})",
             f"Macro F1: {m['macro_f1']:.4f}"]
    if "roc_auc" in m:
        lines.append(f"ROC-AUC:  {m['roc_auc']:.4f}")
    cm = m["confusion_matrix"]
    a, b = m["class_names"]
    lines += ["", f"Confusion matrix (rows = actual, columns = predicted)",
              f"{'':>12}{a:>12}{b:>12}",
              f"{a:>12}{cm[0][0]:>12}{cm[0][1]:>12}",
              f"{b:>12}{cm[1][0]:>12}{cm[1][1]:>12}", "",
              f"{'class':>12}{'precision':>12}{'recall':>10}{'f1':>10}"
              f"{'support':>10}"]
    for name, v in m["per_class"].items():
        lines.append(f"{name:>12}{v['precision']:>12.3f}{v['recall']:>10.3f}"
                     f"{v['f1']:>10.3f}{v['support']:>10d}")
    return "\n".join(lines) + "\n"


def plot_confusion(ax, m, title):
    cm = np.array(m["confusion_matrix"])
    ax.imshow(cm, cmap="Greens")
    names = m["class_names"]
    ax.set_xticks([0, 1], names)
    ax.set_yticks([0, 1], names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14, fontweight="bold")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", required=True, help="Path to the Kaggle CSV file")
    ap.add_argument("--target-col", default=None,
                    help="Name of the occupancy/motion column, if not detected")
    ap.add_argument("--dark-threshold", type=float, default=None,
                    help="Light value below which the room counts as 'dark' "
                         "(default: 25th percentile of the light column)")
    ap.add_argument("--dataset-name", default="(fill in: dataset name, author, URL)",
                    help="Citation text stored in the report")
    args = ap.parse_args()

    df, mapping = load_dataset(args.data, args.target_col)
    REPORT_DIR.mkdir(exist_ok=True)
    MODEL_PATH.parent.mkdir(exist_ok=True)

    dark_thr = (args.dark_threshold if args.dark_threshold is not None
                else float(df["light_level"].quantile(0.25)))

    counts = df[TARGET].value_counts().sort_index()
    summary = (
        f"Dataset: {args.dataset_name}\nFile: {args.data}\n"
        f"Rows used: {len(df):,}\nColumn mapping: {mapping}\n"
        f"Class counts (0 = empty, 1 = occupied): {counts.to_dict()}\n\n"
        f"{df.describe().round(3).to_string()}\n"
    )
    (REPORT_DIR / "dataset_summary.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary)

    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
    print(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")

    candidates = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                class_weight="balanced", max_iter=1000,
                random_state=RANDOM_STATE)),
        ]),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=5, class_weight="balanced",
            random_state=RANDOM_STATE),
    }

    # --- model selection: cross-validation on the TRAINING set only ---
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    comparison, oof = [], {}
    for name, est in candidates.items():
        proba = cross_val_predict(est, X_train, y_train, cv=cv,
                                  method="predict_proba")[:, 1]
        oof[name] = proba
        pred = (proba >= 0.5).astype(int)
        comparison.append({
            "model": name,
            "cv_accuracy": accuracy_score(y_train, pred),
            "cv_macro_f1": f1_score(y_train, pred, average="macro"),
            "cv_roc_auc": roc_auc_score(y_train, proba),
        })
    comp_df = pd.DataFrame(comparison).round(4)
    comp_df.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    print("\n5-fold cross-validation on training data:\n", comp_df.to_string(index=False))

    best_name = comp_df.sort_values("cv_macro_f1", ascending=False).iloc[0]["model"]
    print(f"\nSelected model: {best_name}")

    # --- tune decision threshold on out-of-fold predictions (no test data) ---
    grid = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y_train, (oof[best_name] >= t).astype(int),
                       average="macro") for t in grid]
    threshold = float(grid[int(np.argmax(scores))])
    print(f"Tuned decision threshold: {threshold:.3f}")

    # --- final fit on train, single evaluation on the untouched test set ---
    model = candidates[best_name]
    model.fit(X_train, y_train)
    proba_test = model.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= threshold).astype(int)

    occ = block_metrics(y_test.values, pred_test, ["Empty", "Occupied"],
                        proba_test)

    # Lighting label rule: ON = occupied AND dark.
    light_test = X_test["light_level"].values
    true_on = ((y_test.values == 1) & (light_test < dark_thr)).astype(int)
    pred_on = ((pred_test == 1) & (light_test < dark_thr)).astype(int)
    lights = block_metrics(true_on, pred_on, ["OFF", "ON"])

    text = (report_text("OCCUPANCY MODEL - TEST SET", occ) + "\n" +
            report_text("LIGHTING DECISION (ON/OFF) - TEST SET", lights))
    (REPORT_DIR / "classification_report.txt").write_text(text, encoding="utf-8")
    print("\n" + text)

    # --- figures ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    plot_confusion(axes[0], occ, "Occupancy model (test set)")
    plot_confusion(axes[1], lights, "Lighting decision ON/OFF (test set)")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    if best_name == "Logistic Regression":
        imp = pd.Series(model.named_steps["classifier"].coef_[0], index=FEATURES)
        label = "Standardised coefficient"
    else:
        imp = pd.Series(model.feature_importances_, index=FEATURES)
        label = "Importance"
    fig, ax = plt.subplots(figsize=(6, 3.5))
    imp.sort_values().plot(kind="barh", ax=ax, color="#45A36B")
    ax.set_xlabel(label)
    ax.set_title(f"Feature influence - {best_name}")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)

    # --- metrics.json for the dashboard ---
    metrics = {
        "dataset": {
            "citation": args.dataset_name,
            "file": Path(args.data).name,
            "rows_used": int(len(df)),
            "class_counts": {"empty": int(counts.get(0, 0)),
                             "occupied": int(counts.get(1, 0))},
            "column_mapping": mapping,
        },
        "split": {"train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
                  "test_size": 0.20, "stratified": True,
                  "random_state": RANDOM_STATE},
        "label_rule": (
            "Occupancy label = the dataset's own motion/PIR/occupancy column "
            "(>0 means occupied). Lighting label: ON when the room is occupied "
            f"AND light_level < {dark_thr:.2f}; otherwise OFF."),
        "dark_threshold": dark_thr,
        "model_name": best_name,
        "decision_threshold": threshold,
        "cross_validation": comp_df.to_dict(orient="records"),
        "occupancy_model": occ,
        "lighting_decision": lights,
    }
    (REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2),
                                             encoding="utf-8")

    joblib.dump({
        "model": model,
        "features": FEATURES,
        "threshold": threshold,
        "model_name": best_name,
        "target": TARGET,
        "dark_threshold": dark_thr,
        "dataset": args.dataset_name,
    }, MODEL_PATH)
    print(f"\nSaved model   -> {MODEL_PATH}")
    print(f"Saved reports -> {REPORT_DIR}")


if __name__ == "__main__":
    main()
