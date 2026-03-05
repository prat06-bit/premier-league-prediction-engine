"""
model_loader.py  —  KickIQ production-safe model loader
─────────────────────────────────────────────────────────
Drop this file next to app.py and replace the existing load_models()
call with the one at the bottom of this file.

Key guarantees
  • Prints all library versions on startup so Streamlit Cloud logs are
    self-documenting.
  • Validates that every loaded object has the expected type.
  • Soft-blocks on a version mismatch: warns loudly but does NOT
    crash the app so you can still triage in production.
  • Hard-blocks if the sklearn mismatch is MAJOR (e.g. 1.x vs 2.x).
  • All version constants live in one place (REQUIRED_VERSIONS) so
    updating them is a single-line change.
"""

from __future__ import annotations

import warnings
import logging
from typing import Any

import joblib
import numpy as np
import streamlit as st

# ── optional imports — we handle ImportError gracefully ──────────────────────
try:
    import sklearn
    _SKLEARN_VERSION = sklearn.__version__
except ImportError:
    _SKLEARN_VERSION = "NOT INSTALLED"

try:
    import xgboost as xgb
    _XGB_VERSION = xgb.__version__
except ImportError:
    _XGB_VERSION = "NOT INSTALLED"

try:
    import pandas as pd
    _PANDAS_VERSION = pd.__version__
except ImportError:
    _PANDAS_VERSION = "NOT INSTALLED"

# ─────────────────────────────────────────────────────────────────────────────
# ▸  EDIT THESE to match the versions the models were trained with.
#    Run diagnose_models.py once to discover the correct values.
# ─────────────────────────────────────────────────────────────────────────────
REQUIRED_VERSIONS: dict[str, str] = {
    "sklearn" : "1.6.1",   # ← replace with version from diagnose_models output
    "xgboost" : "2.1.4",   # ← replace with version from diagnose_models output
    "numpy"   : "2.0.2",   # ← replace with version from diagnose_models output
    "pandas"  : "2.2.3",   # ← replace with version from diagnose_models output
}

# Path constants — change once here rather than in every caller
MODEL_PATHS = {
    "xgb" : "models/tuned/xgboost_tuned.pkl",
    "rf"  : "models/tuned/random_forest_tuned.pkl",
    "fc"  : "models/tuned/feature_columns.pkl",
}

logger = logging.getLogger("kickiq.model_loader")


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _runtime_versions() -> dict[str, str]:
    return {
        "sklearn" : _SKLEARN_VERSION,
        "xgboost" : _XGB_VERSION,
        "numpy"   : np.__version__,
        "pandas"  : _PANDAS_VERSION,
        "joblib"  : joblib.__version__,
    }


def _major(version_str: str) -> int:
    """Return the major component of a version string like '1.6.1'."""
    try:
        return int(version_str.split(".")[0])
    except (ValueError, AttributeError):
        return -1


def _check_versions() -> list[str]:
    """
    Compare runtime versions against REQUIRED_VERSIONS.

    Returns a list of human-readable warning strings (empty = all good).
    Raises RuntimeError for MAJOR version mismatches (e.g. sklearn 1.x vs 2.x).
    """
    issues: list[str] = []
    runtime = _runtime_versions()

    for lib, required in REQUIRED_VERSIONS.items():
        current = runtime.get(lib, "UNKNOWN")
        if current == "UNKNOWN" or current == "NOT INSTALLED":
            issues.append(f"{lib}: required {required}, but not installed.")
            continue
        if current == required:
            continue

        msg = (
            f"{lib}: model trained with {required}, "
            f"runtime is {current}."
        )
        issues.append(msg)

        # Hard-block on a MAJOR mismatch — predictions WILL be wrong
        if _major(required) != _major(current):
            raise RuntimeError(
                f"[KickIQ] FATAL version mismatch — {msg}  "
                f"Predictions cannot be trusted.  "
                f"Fix: pin {lib}=={required} in requirements.txt and redeploy."
            )

    return issues


def _validate_model(name: str, obj: Any, expected_type: type) -> bool:
    """
    Return True if obj is the expected type and has a predict_proba method.
    Logs an error and returns False otherwise.
    """
    if not isinstance(obj, expected_type):
        logger.error(
            "[KickIQ] %s: expected %s, got %s",
            name, expected_type.__name__, type(obj).__name__,
        )
        return False

    if not callable(getattr(obj, "predict_proba", None)):
        logger.error("[KickIQ] %s: has no callable predict_proba", name)
        return False

    return True


def _load_with_version_capture(path: str) -> Any:
    """
    Load a joblib file while capturing (not silencing) version warnings
    so they surface in Streamlit Cloud logs.
    """
    captured: list[str] = []

    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        obj = joblib.load(path)

    for w in ws:
        msg = f"[{w.category.__name__}] {w.message}"
        captured.append(msg)
        logger.warning("[KickIQ] joblib.load('%s') → %s", path, msg)

    return obj, captured


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading KickIQ prediction engine…")
def load_models():
    """
    Production-safe model loader for KickIQ.

    Returns (xgb_model, rf_model, feature_columns, ok: bool)
    where ok=False means at least one model failed to load correctly.

    The function:
      1. Logs all library versions to stdout (visible in Streamlit Cloud logs).
      2. Checks for version mismatches against REQUIRED_VERSIONS.
      3. Loads each model with full warning capture.
      4. Type-validates each loaded object.
      5. Fails safely — returns (None, None, None, False) on any hard error.
    """
    import sklearn.ensemble   # noqa: F401  — needed for isinstance check
    import xgboost as _xgb    # noqa: F401

    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier

    # ── 1. print library inventory to logs ───────────────────────────────────
    runtime = _runtime_versions()
    version_block = "  |  ".join(f"{k} {v}" for k, v in runtime.items())
    logger.info("[KickIQ] Runtime: %s", version_block)
    print(f"\n[KickIQ] ── Library versions ──────────────────────────────────")
    for lib, ver in runtime.items():
        print(f"[KickIQ]   {lib:<12} {ver}")
    print(f"[KickIQ] ──────────────────────────────────────────────────────\n")

    # ── 2. version gate ───────────────────────────────────────────────────────
    try:
        issues = _check_versions()
    except RuntimeError as fatal:
        st.error(str(fatal))
        logger.critical(str(fatal))
        return None, None, None, False

    if issues:
        for issue in issues:
            msg = (
                f"[KickIQ] Version mismatch: {issue}  "
                f"Predictions may be unreliable.  "
                f"Fix: update requirements.txt to match REQUIRED_VERSIONS in model_loader.py."
            )
            logger.warning(msg)
            st.warning(msg)

    # ── 3. load models ────────────────────────────────────────────────────────
    all_warnings: list[str] = []
    try:
        xgb_model, w1 = _load_with_version_capture(MODEL_PATHS["xgb"])
        rf_model,  w2 = _load_with_version_capture(MODEL_PATHS["rf"])
        fc,        w3 = _load_with_version_capture(MODEL_PATHS["fc"])
        all_warnings = w1 + w2 + w3
    except FileNotFoundError as exc:
        msg = f"[KickIQ] Model file not found: {exc}.  Check that models/ directory is committed to the repo."
        st.error(msg)
        logger.error(msg)
        return None, None, None, False
    except Exception as exc:
        msg = f"[KickIQ] Unexpected error loading models: {exc}"
        st.error(msg)
        logger.exception(msg)
        return None, None, None, False

    # ── 4. structural validation ──────────────────────────────────────────────
    valid = (
        _validate_model("xgb_model", xgb_model, XGBClassifier)
        and _validate_model("rf_model",  rf_model,  RandomForestClassifier)
    )
    if not valid:
        st.error("[KickIQ] Model validation failed — one or more models have an unexpected type.")
        return None, None, None, False

    # ── 5. warn in UI if joblib emitted version warnings ─────────────────────
    if all_warnings:
        with st.expander("⚠ Model compatibility warnings (click to expand)", expanded=False):
            st.markdown(
                "These warnings were emitted during model loading.  "
                "They indicate a version mismatch between the training environment and this runtime.  "
                "**Pin the correct library versions in requirements.txt** to eliminate them."
            )
            for w in all_warnings:
                st.code(w, language="text")

    logger.info("[KickIQ] Models loaded successfully.")
    return xgb_model, rf_model, fc, True
