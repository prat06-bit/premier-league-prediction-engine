"""
diagnose_models.py
──────────────────
Run this ONCE in whatever Python environment the .pkl files were originally
created in (or in your current environment) to fingerprint the serialised
models and print the exact library versions required.

Usage:
    python diagnose_models.py
"""

import sys
import warnings
import importlib
import joblib
import pickle

# ── colour helpers (no external deps) ────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):    print(f"  {GREEN}✔  {RESET}{msg}")
def warn(msg):  print(f"  {YELLOW}⚠  {RESET}{msg}")
def err(msg):   print(f"  {RED}✘  {RESET}{msg}")
def info(msg):  print(f"  {CYAN}ℹ  {RESET}{msg}")

# ── 1. current runtime versions ───────────────────────────────────────────────
LIBS = ["sklearn", "xgboost", "numpy", "pandas", "joblib", "streamlit"]

def current_versions() -> dict:
    versions = {}
    print(f"\n{BOLD}── CURRENT RUNTIME VERSIONS ──────────────────────────────────{RESET}")
    for lib in LIBS:
        try:
            mod = importlib.import_module(lib)
            ver = getattr(mod, "__version__", "unknown")
            versions[lib] = ver
            ok(f"{lib:<14} {ver}")
        except ImportError:
            versions[lib] = None
            warn(f"{lib:<14} NOT INSTALLED")
    return versions

# ── 2. introspect a .pkl file for embedded version tags ───────────────────────
def inspect_pkl(path: str) -> dict:
    """
    joblib/pickle embed the class's module path and, in sklearn >= 1.0,
    an explicit __sklearn_version__ attribute on fitted estimators.
    XGBoost Booster objects carry a __dict__ with version metadata
    when saved via joblib.

    Returns a dict of discovered metadata.
    """
    meta = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # suppress version warnings for diagnosis
        try:
            obj = joblib.load(path)
        except Exception as exc:
            meta["load_error"] = str(exc)
            return meta

    meta["type"]       = type(obj).__name__
    meta["module"]     = type(obj).__module__

    # ── sklearn estimators ────────────────────────────────────────────────────
    sklearn_ver = getattr(obj, "__sklearn_version__", None)
    if sklearn_ver:
        meta["sklearn_version_in_model"] = sklearn_ver

    # sklearn pipelines / ensembles expose child estimators
    if hasattr(obj, "estimators_"):
        child = obj.estimators_[0] if hasattr(obj.estimators_[0], "__sklearn_version__") \
                else (obj.estimators_[0][0] if hasattr(obj.estimators_[0], "__iter__") else None)
        if child:
            child_ver = getattr(child, "__sklearn_version__", None)
            if child_ver:
                meta["sklearn_version_in_child_estimator"] = child_ver

    # single-tree models
    if hasattr(obj, "tree_"):
        meta["has_fitted_tree"] = True

    # ── xgboost Booster / XGBClassifier ──────────────────────────────────────
    if hasattr(obj, "get_booster"):
        booster = obj.get_booster()
        cfg = booster.save_config() if hasattr(booster, "save_config") else None
        if cfg:
            import json
            try:
                cfg_dict = json.loads(cfg)
                xgb_ver  = cfg_dict.get("learner", {}).get("attributes", {}).get("scikit_learn", "")
                if xgb_ver:
                    meta["xgb_config_sklearn_attr"] = xgb_ver
                # also grab the raw version field Booster embeds
                meta["xgb_config_preview"] = str(cfg_dict)[:200]
            except Exception:
                pass
    elif hasattr(obj, "save_config"):           # raw Booster
        import json
        try:
            cfg_dict = json.loads(obj.save_config())
            meta["xgb_learner_attributes"] = cfg_dict.get("learner", {}).get("attributes", {})
        except Exception:
            pass

    # ── numpy arrays (feature_columns.pkl) ────────────────────────────────────
    import numpy as np
    if isinstance(obj, (list, np.ndarray)):
        meta["contents_preview"] = str(obj)[:200]

    return meta


# ── 3. compare current vs detected ───────────────────────────────────────────
def compare(label: str, detected: str | None, current: str | None):
    if detected is None:
        warn(f"{label}: version tag NOT EMBEDDED in model file "
             f"(current runtime = {current})")
        return
    if detected == current:
        ok(f"{label}: model={detected}  runtime={current}  — MATCH ✔")
    else:
        err(f"{label}: model={detected}  runtime={current}  — MISMATCH ✘")
        info(f"  → add  {label}=={detected}  to requirements.txt")


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    import sklearn
    import xgboost

    MODEL_PATHS = {
        "xgb_model"      : "models/tuned/xgboost_tuned.pkl",
        "rf_model"       : "models/tuned/random_forest_tuned.pkl",
        "feature_columns": "models/tuned/feature_columns.pkl",
    }

    runtime = current_versions()

    print(f"\n{BOLD}── MODEL FILE INSPECTION ─────────────────────────────────────{RESET}")
    for label, path in MODEL_PATHS.items():
        print(f"\n  {CYAN}{label}{RESET}  ({path})")
        meta = inspect_pkl(path)
        if "load_error" in meta:
            err(f"  Could not load: {meta['load_error']}")
            continue
        for k, v in meta.items():
            info(f"  {k}: {v}")

    print(f"\n{BOLD}── VERSION CROSS-CHECK ───────────────────────────────────────{RESET}")
    # sklearn check — joblib embeds __sklearn_version__ since sklearn 1.0
    # If not present, we fall back to the warning text from stderr
    print("\n  NOTE: If '__sklearn_version__' was NOT printed above, sklearn did")
    print("  not embed it (sklearn < 1.0 or stripped build). In that case,")
    print("  check the InconsistentVersionWarning text — it states the version.")
    print()
    print(f"  Your current scikit-learn : {sklearn.__version__}")
    print(f"  Your current xgboost      : {xgboost.__version__}")
    print()
    print(f"{BOLD}── RECOMMENDED requirements.txt PINS ────────────────────────{RESET}")
    print()
    print("  Pin every library to the version shown in the MODEL column above.")
    print("  Minimum safe example (fill in your actual versions):")
    print()
    for lib, ver in runtime.items():
        if ver:
            print(f"    {lib}=={ver}")

    print(f"\n{BOLD}── DONE ─────────────────────────────────────────────────────{RESET}\n")


if __name__ == "__main__":
    main()
