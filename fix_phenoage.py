"""
Run from project root: python3 fix_phenoage.py
Diagnoses the xb distribution and finds the correct PhenoAge formula calibration.
"""
import pandas as pd
import numpy as np

df = pd.read_parquet("data/processed/nhanes_merged.parquet")

# Drop rows missing any biomarker
bio_cols = ["albumin_gdl","creatinine_mgdl","glucose_mgdl","crp_mgdl",
            "lymph_pct","mcv_fl","rdw_pct","alp_ul","wbc_si"]
df = df.dropna(subset=bio_cols).copy()
print(f"Rows with complete biomarkers: {len(df):,}")
print(f"Mean age: {df['age'].mean():.1f}")

# ── Step 1: compute xb and inspect its distribution ──────────────────────────
log_crp = np.log(df["crp_mgdl"].clip(lower=0.001))

xb = (
    -19.9067
    + (-0.0336  * df["albumin_gdl"])
    + ( 0.0095  * df["creatinine_mgdl"] * 88.402)
    + ( 0.1953  * df["glucose_mgdl"]    * 0.05551)
    + ( 0.0954  * (log_crp + np.log(10)))
    + (-0.0120  * df["lymph_pct"])
    + (-0.0268  * df["mcv_fl"])
    + ( 0.3306  * df["rdw_pct"])
    + ( 0.00188 * df["alp_ul"])
    + ( 0.0554  * df["wbc_si"])
)

print(f"\nxb distribution:")
print(f"  mean={xb.mean():.4f}  std={xb.std():.4f}")
print(f"  min={xb.min():.4f}  max={xb.max():.4f}")
print(f"  p5={xb.quantile(0.05):.4f}  p95={xb.quantile(0.95):.4f}")

# ── Step 2: inspect each term's contribution ──────────────────────────────────
print(f"\nMean contribution per term:")
print(f"  intercept          : -19.9067")
print(f"  albumin            : {(-0.0336 * df['albumin_gdl']).mean():.4f}")
print(f"  creatinine (umol)  : {(0.0095 * df['creatinine_mgdl'] * 88.402).mean():.4f}")
print(f"  glucose (mmol)     : {(0.1953 * df['glucose_mgdl'] * 0.05551).mean():.4f}")
print(f"  ln_crp (mg/L)      : {(0.0954 * (log_crp + np.log(10))).mean():.4f}")
print(f"  lymph_pct          : {(-0.0120 * df['lymph_pct']).mean():.4f}")
print(f"  mcv                : {(-0.0268 * df['mcv_fl']).mean():.4f}")
print(f"  rdw                : {(0.3306 * df['rdw_pct']).mean():.4f}")
print(f"  alp                : {(0.00188 * df['alp_ul']).mean():.4f}")
print(f"  wbc                : {(0.0554 * df['wbc_si']).mean():.4f}")
print(f"  TOTAL xb mean      : {xb.mean():.4f}")

# ── Step 3: what xb would we NEED for the formula to give age~50? ────────────
# For PhenoAge = age (well-calibrated), we need to find what xb corresponds to
# phenoage = 50 using the Gompertz parameters
b, mu0, c2 = 0.0076927, 1.51714e-5, 0.00553

# Work backwards: what M gives phenoage=50?
# 50 = 141.50225 + (1/b)*ln((1/c2)*(-ln(1-M)))
# (50 - 141.50225)*b = ln((1/c2)*(-ln(1-M)))
# exp((50-141.50225)*b) = (1/c2)*(-ln(1-M))
# -ln(1-M) = c2 * exp((50-141.50225)*b)

for target_age in [40, 50, 60, 70]:
    val = c2 * np.exp((target_age - 141.50225) * b)
    M_target = 1 - np.exp(-val)
    # Now what exp(xb) gives this M?
    # M = 1 - exp(-mu0 * exp(xb) * (exp(b*120)-1)/b)
    # exp(xb) = -ln(1-M) * b / (mu0 * (exp(b*120)-1))
    factor = mu0 * (np.exp(b * 120) - 1) / b
    exp_xb_needed = -np.log(1 - M_target) / factor
    xb_needed = np.log(exp_xb_needed)
    print(f"\nFor PhenoAge={target_age}: need xb={xb_needed:.4f}, M={M_target:.8f}")

print(f"\nOur actual mean xb = {xb.mean():.4f}")
print(f"Required xb for mean age {df['age'].mean():.0f} ≈ {xb.mean():.4f}")
print(f"\nCorrection needed to intercept: target_xb - actual_xb")

# The right intercept = -19.9067 + (xb_needed_at_mean_age - xb.mean())
mean_age = df['age'].mean()
val = c2 * np.exp((mean_age - 141.50225) * b)
M_at_mean = 1 - np.exp(-val)
factor = mu0 * (np.exp(b * 120) - 1) / b
xb_needed_at_mean = np.log(-np.log(1 - M_at_mean) / factor)
correction = xb_needed_at_mean - xb.mean()
print(f"Corrected intercept = -19.9067 + {correction:.4f} = {-19.9067 + correction:.4f}")
