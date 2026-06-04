"""
Biological Age Modeling — Phase 2
===================================
Direct mortality prediction from biomarkers using XGBoost.
Target = actual death outcome (mortstat). No Levine formula dependency.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
from sklearn.impute import SimpleImputer
import xgboost as xgb
import shap

PROC_DIR    = Path(__file__).parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

BIOMARKER_COLS = [
    "albumin_gdl", "creatinine_mgdl", "glucose_mgdl", "crp_mgdl",
    "lymph_pct", "mcv_fl", "rdw_pct", "alp_ul", "wbc_si"
]
DEMO_COLS    = ["age", "gender", "race"]
FEATURE_COLS = BIOMARKER_COLS + DEMO_COLS

LABELS = {
    "albumin_gdl":     "Albumin (g/dL)",
    "creatinine_mgdl": "Creatinine (mg/dL)",
    "glucose_mgdl":    "Glucose (mg/dL)",
    "crp_mgdl":        "CRP (mg/dL)",
    "lymph_pct":       "Lymphocyte %",
    "mcv_fl":          "MCV (fL)",
    "rdw_pct":         "RDW %",
    "alp_ul":          "ALP (U/L)",
    "wbc_si":          "WBC (1000/µL)",
    "age":             "Age (yrs)",
    "gender":          "Gender",
    "race":            "Race/Ethnicity",
}


def load_cohort():
    df = pd.read_parquet(PROC_DIR / "nhanes_merged.parquet")
    df = df[df["cycle"].isin(["2015-2016", "2017-2018"])].copy()
    df = df[df["mortstat"].isin([0, 1])].copy()
    df["n_bio"] = df[BIOMARKER_COLS].notna().sum(axis=1)
    df = df[df["n_bio"] >= 6].copy()
    print(f"Cohort: {len(df):,} participants")
    print(f"  Alive:    {(df['mortstat']==0).sum():,} ({100*(df['mortstat']==0).mean():.1f}%)")
    print(f"  Deceased: {(df['mortstat']==1).sum():,} ({100*(df['mortstat']==1).mean():.1f}%)")
    return df


def prep_xy(df):
    """Impute and return X array + y series."""
    X_raw = df[FEATURE_COLS].copy()
    y     = df["mortstat"].astype(int).copy()
    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X_raw), columns=FEATURE_COLS)
    return X_imp, y, imputer


def make_model(scale_pos_weight):
    return xgb.XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
    )


def cross_validate(X, y, n_splits=5):
    cv  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    spw = (y == 0).sum() / (y == 1).sum()

    aucs, aps = [], []
    oof_probs = np.zeros(len(y))

    print(f"\nCross-validation ({n_splits}-fold stratified):")
    for fold, (tr, val) in enumerate(cv.split(X, y)):
        X_tr, X_val = X.iloc[tr], X.iloc[val]
        y_tr, y_val = y.iloc[tr], y.iloc[val]

        m = make_model(spw)
        m.fit(X_tr, y_tr,
              eval_set=[(X_val, y_val)],
              verbose=False)

        probs = m.predict_proba(X_val)[:, 1]
        oof_probs[val] = probs

        auc = roc_auc_score(y_val, probs)
        ap  = average_precision_score(y_val, probs)
        aucs.append(auc)
        aps.append(ap)
        print(f"  Fold {fold+1}: AUC={auc:.3f}  AP={ap:.3f}")

    print(f"\n  Mean AUC : {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    print(f"  Mean AP  : {np.mean(aps):.3f} ± {np.std(aps):.3f}")
    return oof_probs, aucs, aps


def plot_roc_pr(y, oof_probs):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    RocCurveDisplay.from_predictions(y, oof_probs, ax=axes[0], name="XGBoost (OOF)")
    axes[0].plot([0,1],[0,1],"k--", alpha=0.4, label="Random")
    axes[0].set_title("ROC Curve — Mortality Prediction", fontsize=13)
    axes[0].legend()

    PrecisionRecallDisplay.from_predictions(y, oof_probs, ax=axes[1], name="XGBoost (OOF)")
    axes[1].axhline(y.mean(), color="k", linestyle="--", alpha=0.4,
                    label=f"Baseline ({y.mean():.3f})")
    axes[1].set_title("Precision-Recall Curve", fontsize=13)
    axes[1].legend()

    plt.tight_layout()
    path = RESULTS_DIR / "roc_pr_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


def plot_shap(model, X):
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # ── Bar: mean |SHAP| ─────────────────────────────────────────────────
    mean_shap = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=X.columns
    ).sort_values(ascending=True)
    mean_shap.index = [LABELS.get(c, c) for c in mean_shap.index]

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#e63946" if any(k in label for k in ["Age","RDW","CRP","Albumin"])
              else "#457b9d" for label in mean_shap.index]
    mean_shap.plot(kind="barh", ax=ax, color=colors, edgecolor="white")
    ax.set_xlabel("Mean |SHAP Value| (impact on mortality risk)", fontsize=11)
    ax.set_title("Biomarker Importance for Mortality Prediction\n(our model — no Levine formula)",
                 fontsize=12)
    plt.tight_layout()
    path = RESULTS_DIR / "shap_importance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")

    # ── Beeswarm ─────────────────────────────────────────────────────────
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X,
                      feature_names=[LABELS.get(c,c) for c in X.columns],
                      show=False, max_display=12)
    plt.title("SHAP Summary — Direction & Magnitude of Each Biomarker",
              fontsize=12, pad=12)
    plt.tight_layout()
    path = RESULTS_DIR / "shap_beeswarm.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")

    return mean_shap


def run():
    print("=" * 55)
    print("Phase 2: Mortality Prediction from Biomarkers")
    print("=" * 55)

    df = load_cohort()
    X, y, imputer = prep_xy(df)

    # Cross-validation
    oof_probs, aucs, aps = cross_validate(X, y)
    plot_roc_pr(y, oof_probs)

    # Final model on full data for SHAP
    print("\nTraining final model for SHAP...")
    spw = (y == 0).sum() / (y == 1).sum()
    final_model = make_model(spw)
    final_model.fit(X, y, verbose=False)

    print("Computing SHAP values...")
    mean_shap = plot_shap(final_model, X)

    # Save results
    importance_df = mean_shap.sort_values(ascending=False).reset_index()
    importance_df.columns = ["feature", "mean_abs_shap"]
    importance_df["rank"] = range(1, len(importance_df)+1)
    importance_df.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)

    pd.DataFrame({
        "metric": ["AUC_mean","AUC_std","AP_mean","AP_std"],
        "value":  [np.mean(aucs), np.std(aucs), np.mean(aps), np.std(aps)]
    }).to_csv(RESULTS_DIR / "cv_summary.csv", index=False)

    print("\n" + "=" * 55)
    print("BIOMARKER IMPORTANCE RANKING (SHAP)")
    print("=" * 55)
    print(mean_shap.sort_values(ascending=False).to_string())
    # Save model for inference
    save_model(final_model, imputer)
    print(f"\nAll results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    run()


def save_model(model, imputer):
    """Save model + imputer for inference script."""
    import pickle
    payload = {"model": model, "imputer": imputer}
    path = RESULTS_DIR / "xgb_model.pkl"
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    print(f"Saved → {path}")
