"""
Inference Script — Mortality Risk from Blood Panel
====================================================
Predict 4-year all-cause mortality risk from a standard blood panel.

Usage:
  python3 src/inference.py \
    --age 42 --gender 1 \
    --albumin 4.3 --creatinine 0.9 --glucose 95 --crp 0.2 \
    --lymph_pct 31 --mcv 89 --rdw 13.1 --alp 72 --wbc 6.5

Gender: 1=Male, 2=Female

DISCLAIMER: For research and educational purposes only.
Not a medical device. Does not constitute clinical advice.
"""

import argparse
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
MODEL_PATH  = RESULTS_DIR / "xgb_model.pkl"

FEATURE_COLS = [
    "albumin_gdl", "creatinine_mgdl", "glucose_mgdl", "crp_mgdl",
    "lymph_pct", "mcv_fl", "rdw_pct", "alp_ul", "wbc_si",
    "age", "gender", "race"
]

NORMAL_RANGES = {
    "albumin_gdl":     (3.5,  5.0,  "g/dL",   "Albumin"),
    "creatinine_mgdl": (0.6,  1.2,  "mg/dL",  "Creatinine"),
    "glucose_mgdl":    (70,   100,  "mg/dL",  "Glucose"),
    "crp_mgdl":        (0,    0.3,  "mg/dL",  "CRP"),
    "lymph_pct":       (20,   40,   "%",       "Lymphocyte %"),
    "mcv_fl":          (80,   100,  "fL",      "MCV"),
    "rdw_pct":         (11.5, 14.5, "%",       "RDW"),
    "alp_ul":          (44,   147,  "U/L",     "ALP"),
    "wbc_si":          (4.5,  11.0, "K/µL",   "WBC"),
}

SHAP_RANKING = {
    "rdw_pct":         ("RDW %",           "↑ higher = more risk"),
    "mcv_fl":          ("MCV",             "↑ higher = more risk"),
    "lymph_pct":       ("Lymphocyte %",    "↓ lower = more risk"),
    "creatinine_mgdl": ("Creatinine",      "↑ higher = more risk"),
    "crp_mgdl":        ("CRP",             "↑ higher = more risk"),
    "wbc_si":          ("WBC",             "↑ elevated = more risk"),
    "alp_ul":          ("ALP",             "↑ elevated = more risk"),
    "albumin_gdl":     ("Albumin",         "↓ lower = more risk"),
    "glucose_mgdl":    ("Glucose",         "↑ higher = more risk"),
}


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}\n"
            f"Run 'python3 src/model.py' first to train and save the model."
        )
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def flag_abnormal(values: dict) -> list:
    """Return list of biomarkers outside normal range with direction."""
    flags = []
    for col, (lo, hi, unit, name) in NORMAL_RANGES.items():
        val = values.get(col)
        if val is None:
            continue
        if val < lo:
            flags.append(f"  ⬇  {name}: {val} {unit} (below normal {lo}–{hi})")
        elif val > hi:
            flags.append(f"  ⬆  {name}: {val} {unit} (above normal {lo}–{hi})")
    return flags


def interpret_risk(prob: float) -> str:
    if prob < 0.02:
        return "Low"
    elif prob < 0.05:
        return "Moderate"
    elif prob < 0.10:
        return "Elevated"
    else:
        return "High"


def predict(args):
    model_data = load_model()
    model   = model_data["model"]
    imputer = model_data["imputer"]

    values = {
        "albumin_gdl":     args.albumin,
        "creatinine_mgdl": args.creatinine,
        "glucose_mgdl":    args.glucose,
        "crp_mgdl":        args.crp,
        "lymph_pct":       args.lymph_pct,
        "mcv_fl":          args.mcv,
        "rdw_pct":         args.rdw,
        "alp_ul":          args.alp,
        "wbc_si":          args.wbc,
        "age":             args.age,
        "gender":          args.gender,
        "race":            args.race,
    }

    X = pd.DataFrame([values])[FEATURE_COLS]
    X_imp = pd.DataFrame(imputer.transform(X), columns=FEATURE_COLS)
    prob  = model.predict_proba(X_imp)[0, 1]
    risk  = interpret_risk(prob)

    print("\n" + "=" * 52)
    print("  MORTALITY RISK ASSESSMENT")
    print("  (Research Use Only — Not Clinical Advice)")
    print("=" * 52)
    print(f"\n  Input: Age {args.age}, {'Male' if args.gender==1 else 'Female'}")
    print(f"\n  4-Year Mortality Risk:  {prob*100:.1f}%")
    print(f"  Risk Category:          {risk}")
    print(f"\n  Population baseline:    ~3.4% (NHANES 2015-2018)")

    flags = flag_abnormal(values)
    if flags:
        print(f"\n  Biomarkers Outside Normal Range:")
        for f in flags:
            print(f)
    else:
        print(f"\n  All biomarkers within normal range.")

    print(f"\n  Key biomarkers by mortality importance (from model):")
    bio_vals = {k: v for k, v in values.items() if k in SHAP_RANKING}
    for i, (col, (name, direction)) in enumerate(SHAP_RANKING.items(), 1):
        val = bio_vals.get(col, "N/A")
        lo, hi, unit, _ = NORMAL_RANGES[col]
        status = ""
        if isinstance(val, (int, float)):
            if val < lo:   status = " ← LOW"
            elif val > hi: status = " ← HIGH"
        print(f"  {i:2d}. {name:18s} {val} {unit}{status}")

    print("\n" + "=" * 52)
    print("  DISCLAIMER: This tool is for research purposes")
    print("  only. Consult a physician for medical advice.")
    print("=" * 52 + "\n")

    return prob


def main():
    parser = argparse.ArgumentParser(
        description="Predict 4-year mortality risk from blood panel values."
    )
    parser.add_argument("--age",         type=float, required=True)
    parser.add_argument("--gender",      type=int,   default=1,    help="1=Male, 2=Female")
    parser.add_argument("--race",        type=int,   default=4,    help="NHANES race code (default=4, Non-Hispanic White)")
    parser.add_argument("--albumin",     type=float, default=None, help="Albumin (g/dL)")
    parser.add_argument("--creatinine",  type=float, default=None, help="Creatinine (mg/dL)")
    parser.add_argument("--glucose",     type=float, default=None, help="Glucose (mg/dL)")
    parser.add_argument("--crp",         type=float, default=None, help="CRP (mg/dL)")
    parser.add_argument("--lymph_pct",   type=float, default=None, help="Lymphocyte %")
    parser.add_argument("--mcv",         type=float, default=None, help="MCV (fL)")
    parser.add_argument("--rdw",         type=float, default=None, help="RDW (%)")
    parser.add_argument("--alp",         type=float, default=None, help="ALP (U/L)")
    parser.add_argument("--wbc",         type=float, default=None, help="WBC (1000/µL)")
    args = parser.parse_args()
    predict(args)


if __name__ == "__main__":
    main()
