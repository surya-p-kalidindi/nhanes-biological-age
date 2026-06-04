"""Run from project root: python3 fix_phenoage2.py"""
import pandas as pd
import numpy as np

df = pd.read_parquet("data/processed/nhanes_merged.parquet")
bio_cols = ["albumin_gdl","creatinine_mgdl","glucose_mgdl","crp_mgdl",
            "lymph_pct","mcv_fl","rdw_pct","alp_ul","wbc_si"]

# Only rows with ALL 9 biomarkers present
complete = df.dropna(subset=bio_cols).copy()
print(f"Complete cases (all 9 biomarkers): {len(complete):,} of {len(df):,}")

log_crp = np.log(complete["crp_mgdl"].clip(lower=0.001))
xb = (
    -4.4978
    + (-0.0336  * complete["albumin_gdl"])
    + ( 0.0095  * complete["creatinine_mgdl"] * 88.402)
    + ( 0.1953  * complete["glucose_mgdl"]    * 0.05551)
    + ( 0.0954  * (log_crp + np.log(10)))
    + (-0.0120  * complete["lymph_pct"])
    + (-0.0268  * complete["mcv_fl"])
    + ( 0.3306  * complete["rdw_pct"])
    + ( 0.00188 * complete["alp_ul"])
    + ( 0.0554  * complete["wbc_si"])
)

b, mu0, c2 = 0.0076927, 1.51714e-5, 0.00553
M = 1 - np.exp(-mu0 * np.exp(xb) * (np.exp(b * 120) - 1) / b)
phenoage = 141.50225 + (1 / b) * np.log((1 / c2) * (-np.log(1 - M.clip(1e-10, 1-1e-10))))

complete["xb"] = xb
complete["M"]  = M
complete["phenoage_raw"] = phenoage

print(f"\nxb: mean={xb.mean():.3f}  std={xb.std():.3f}  min={xb.min():.3f}  max={xb.max():.3f}")
print(f"M:  mean={M.mean():.6f}  min={M.min():.8f}  max={M.max():.6f}")
print(f"    M==0 count: {(M<=0).sum()}  |  M>=1 count: {(M>=1).sum()}")
print(f"\nPhenoAge: mean={phenoage.mean():.1f}  std={phenoage.std():.1f}")
print(f"          min={phenoage.min():.1f}  max={phenoage.max():.1f}")
print(f"          in [0,120]: {phenoage.between(0,120).sum():,}")
print(f"          < 0:        {(phenoage<0).sum():,}")
print(f"          > 120:      {(phenoage>120).sum():,}")

# Look at outlier rows
outliers = complete[~phenoage.between(0, 120)].copy()
print(f"\nOutlier sample (phenoage outside 0-120):")
print(outliers[bio_cols + ["xb","M","phenoage_raw"]].describe().round(3))

# What makes them outliers? Check xb of outliers vs normal
normal = complete[phenoage.between(0, 120)]
print(f"\nxb for valid (0-120): mean={xb[phenoage.between(0,120)].mean():.3f}  std={xb[phenoage.between(0,120)].std():.3f}")
print(f"xb for outliers:       mean={xb[~phenoage.between(0,120)].mean():.3f}  std={xb[~phenoage.between(0,120)].std():.3f}")

# Check which biomarkers drive outliers
print("\nOutlier biomarker means vs normal:")
for col in bio_cols:
    v_mean = normal[col].mean()
    o_mean = outliers[col].mean()
    print(f"  {col:22s}  valid={v_mean:.2f}  outlier={o_mean:.2f}")
