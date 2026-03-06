"""
audit_model_versions.py
───────────────────────
Run this script immediately after training your models and BEFORE deploying.
It prints a manifest of:
  1. The current Python / library environment
  2. The version metadata embedded in each pkl file
  3. Any version deltas between train-time and current environment
  4. A ready-to-paste requirements.txt block

Usage:
  python scripts/audit_model_versions.py

Commit the output alongside your model .pkl files:
  python scripts/audit_model_versions.py > model_version_manifest.txt
  git add models/ model_version_manifest.txt
  git commit -m "chore: update models and version manifest"
"""

import sys
import warnings
import joblib
import sklearn
import xgboost
import numpy
import pandas
import platform
from pathlib import Path
from datetime import datetime

# ── Suppress version warnings during audit ────────────────────────────────────
warnings.filterwarnings("ignore", category=sklearn.exceptions.InconsistentVersionWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*older version.*")


def section(title: str):
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}")


def check(label: str, ok: bool, detail: str = ""):
    icon = "✓" if ok else "⚠"
    print(f"  {icon}  {label:<30}  {detail}")


def parse_version(v: str) -> tuple:
    try:
        return tuple(int(x) for x in str(v).split(".")[:3])
    except Exception:
        return (0, 0, 0)


# ── 1. Environment ─────────────────────────────────────────────────────────────
section("RUNTIME ENVIRONMENT")
print(f"  Generated   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Platform    : {platform.platform()}")
print(f"  Python      : {sys.version.split()[0]}")
print(f"  scikit-learn: {sklearn.__version__}")
print(f"  xgboost     : {xgboost.__version__}")
print(f"  joblib      : {joblib.__version__}")
print(f"  numpy       : {numpy.__version__}")
print(f"  pandas      : {pandas.__version__}")


# ── 2. Model file inspection ───────────────────────────────────────────────────
section("MODEL FILE INSPECTION")

MODEL_PATHS = {
    "XGBoost":         Path("models/tuned/xgboost_tuned.pkl"),
    "Random Forest":   Path("models/tuned/random_forest_tuned.pkl"),
    "Feature Columns": Path("models/tuned/feature_columns.pkl"),
}

loaded = {}
for name, path in MODEL_PATHS.items():
    if not path.exists():
        check(name, False, f"FILE NOT FOUND: {path}")
        loaded[name] = None
        continue

    try:
        obj = joblib.load(path)
        loaded[name] = obj
        size_kb = path.stat().st_size / 1024

        if name == "XGBoost":
            # XGBoost does not embed version in pkl — report what we know
            n_estimators = getattr(obj, 'n_estimators', "?")
            check(name, True,
                  f"loaded OK  |  size={size_kb:.1f}KB  |  n_estimators={n_estimators}  "
                  f"|  xgb_version_at_runtime={xgboost.__version__}")

        elif name == "Random Forest":
            train_version = getattr(obj, '_sklearn_version', 'not embedded')
            env_version   = sklearn.__version__
            major_match   = parse_version(train_version)[0] == parse_version(env_version)[0]
            n_estimators  = getattr(obj, 'n_estimators', "?")
            n_features    = getattr(obj, 'n_features_in_', "?")
            status = "VERSION MATCH" if train_version == env_version else (
                     "MINOR DELTA (safe)" if major_match else "MAJOR MISMATCH (UNSAFE)")
            check(name, major_match,
                  f"size={size_kb:.1f}KB  |  n_estimators={n_estimators}  "
                  f"|  n_features={n_features}  "
                  f"|  trained_with={train_version}  running={env_version}  "
                  f"|  {status}")

        elif name == "Feature Columns":
            n_cols = len(obj) if hasattr(obj, '__len__') else "?"
            check(name, True, f"loaded OK  |  n_features={n_cols}")

    except Exception as e:
        check(name, False, f"LOAD FAILED: {e}")
        loaded[name] = None


# ── 3. Smoke test ──────────────────────────────────────────────────────────────
section("SMOKE TEST (dummy prediction)")

xgb_obj = loaded.get("XGBoost")
rf_obj  = loaded.get("Random Forest")
fc_obj  = loaded.get("Feature Columns")

if xgb_obj and rf_obj and fc_obj:
    try:
        dummy = pandas.DataFrame([[0] * len(fc_obj)], columns=fc_obj)
        xp = xgb_obj.predict_proba(dummy)
        rp = rf_obj.predict_proba(dummy)
        ens = 0.6 * xp + 0.4 * rp
        check("XGBoost predict_proba",  True, f"output shape={xp.shape}")
        check("RF predict_proba",       True, f"output shape={rp.shape}")
        check("Ensemble (60/40)",       True, f"output={ens.round(3)}")
    except Exception as e:
        check("Smoke test", False, f"FAILED: {e}")
else:
    print("  ⚠  Skipped — one or more model files failed to load.")


# ── 4. Requirements block ──────────────────────────────────────────────────────
section("PASTE INTO requirements.txt")
print(f"  scikit-learn=={sklearn.__version__}")
print(f"  xgboost=={xgboost.__version__}")
print(f"  joblib=={joblib.__version__}")
print(f"  numpy=={numpy.__version__}")
print(f"  pandas=={pandas.__version__}")


# ── 5. Recommended save format ─────────────────────────────────────────────────
section("RECOMMENDED SAVE FORMAT (run after training)")
print("""
  # In your train_models.py, save models like this:

  import joblib, sklearn, xgboost

  # sklearn models — joblib is the canonical way
  joblib.dump(rf_model,  'models/tuned/random_forest_tuned.pkl')
  joblib.dump(fc,        'models/tuned/feature_columns.pkl')

  # XGBoost — use native format for maximum version portability
  # This avoids the "older version" pickle warning entirely
  xgb_model.save_model('models/tuned/xgboost_tuned.json')
  # Then load with:  xgb_model.load_model('models/tuned/xgboost_tuned.json')

  # Embed your environment metadata so future audits can check it:
  import json, sys, datetime
  manifest = {
      'trained_at':  datetime.datetime.now().isoformat(),
      'sklearn':     sklearn.__version__,
      'xgboost':     xgboost.__version__,
      'python':      sys.version.split()[0],
  }
  with open('models/tuned/version_manifest.json', 'w') as f:
      json.dump(manifest, f, indent=2)
""")

section("AUDIT COMPLETE")
print()
