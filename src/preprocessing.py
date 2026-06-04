"""
NHANES Preprocessing Pipeline
==============================
Merges demographics, biochemistry, CBC, CRP, and mortality data
across three cycles into one clean analysis-ready DataFrame.

Output columns (final):
  SEQN                  - participant ID
  cycle                 - source cycle label
  age                   - chronological age (years)
  gender                - 1=Male, 2=Female
  race                  - race/ethnicity code
  albumin_gdl           - Albumin (g/dL)
  creatinine_mgdl       - Creatinine (mg/dL)
  glucose_mgdl          - Glucose (mg/dL)
  crp_mgdl              - C-Reactive Protein (mg/dL)
  lymph_pct             - Lymphocyte % of WBC
  mcv_fl                - Mean Corpuscular Volume (fL)
  rdw_pct               - Red Cell Distribution Width (%)
  alp_ul                - Alkaline Phosphatase (U/L)
  wbc_si                - White Blood Cell count (1000/uL)
  mortstat              - 0=assumed alive, 1=deceased (mortality cycles only)
  permth_exm            - follow-up months from exam (mortality cycles only)
"""

import pandas as pd
import numpy as np
import pyreadstat
from pathlib import Path

RAW_DIR   = Path(__file__).parent.parent / "data" / "raw"
PROC_DIR  = Path(__file__).parent.parent / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

# ── Variable name maps ────────────────────────────────────────────────────────
# NHANES uses different variable names across files and cycles.
# Maps: raw NHANES column -> our standard column name

DEMO_COLS = {
    "SEQN":       "SEQN",
    "RIDAGEYR":   "age",
    "RIAGENDR":   "gender",
    "RIDRETH3":   "race",        # 2015+ uses RIDRETH3 (includes NH Asian)
    "RIDRETH1":   "race",        # fallback for older cycles
}

# Biochemistry panel (BIOPRO files)
BIO_COLS = {
    "SEQN":       "SEQN",
    "LBXSAL":     "albumin_gdl",      # Albumin g/dL
    "LBXSCR":     "creatinine_mgdl",  # Creatinine mg/dL
    "LBXSGL":     "glucose_mgdl",     # Glucose mg/dL
    "LBXSAPSI":   "alp_ul",           # Alkaline Phosphatase U/L
    # Some cycles use alternate names:
    "LBDSALSI":   "albumin_gdl",
    "LBDSCRLC":   "creatinine_mgdl",
    "LBDSGLU":    "glucose_mgdl",
}

# CBC panel
CBC_COLS = {
    "SEQN":       "SEQN",
    "LBXLYPCT":   "lymph_pct",   # Lymphocyte %
    "LBXMCVSI":   "mcv_fl",      # MCV fL
    "LBXRDW":     "rdw_pct",     # RDW %
    "LBXWBCSI":   "wbc_si",      # WBC 1000/uL
    # alternate names
    "LBDLYMNO":   "lymph_pct",
}

# High-sensitivity CRP
CRP_COLS = {
    "SEQN":       "SEQN",
    "LBXHSCRP":   "crp_mgdl",    # hs-CRP mg/dL
    "LBDHSCRP":   "crp_mgdl",    # alternate name
}

# Mortality linkage columns to keep
MORT_COLS = ["SEQN", "MORTSTAT", "PERMTH_EXM", "ELIGSTAT"]


# ── Cycle file manifest ───────────────────────────────────────────────────────
CYCLES = [
    {
        "label":       "2015-2016",
        "folder":      "2015_2016",
        "demo":        "DEMO_I.XPT",
        "bio":         "BIOPRO_I.XPT",
        "cbc":         "CBC_I.XPT",
        "crp":         "HSCRP_I.XPT",
        "mortality":   "NHANES_2015_2016_MORT_2019_PUBLIC.dat",
        "has_mort":    True,
    },
    {
        "label":       "2017-2018",
        "folder":      "2017_2018",
        "demo":        "DEMO_J.XPT",
        "bio":         "BIOPRO_J.XPT",
        "cbc":         "CBC_J.XPT",
        "crp":         "HSCRP_J.XPT",
        "mortality":   "NHANES_2017_2018_MORT_2019_PUBLIC.dat",
        "has_mort":    True,
    },
    {
        "label":       "2017-2020",
        "folder":      "2017_2020",
        "demo":        "P_DEMO.xpt",
        "bio":         "P_BIOPRO.xpt",
        "cbc":         "P_CBC.xpt",
        "crp":         "P_HSCRP.xpt",
        "mortality":   None,
        "has_mort":    False,
    },
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def read_xpt(path: Path) -> pd.DataFrame:
    df, _ = pyreadstat.read_xport(str(path))
    return df


def select_rename(df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    """Keep only columns in col_map that exist in df, rename them."""
    available = {k: v for k, v in col_map.items() if k in df.columns}
    return df[list(available.keys())].rename(columns=available)


def parse_mortality(path: Path) -> pd.DataFrame:
    colspecs = [
        (0,  6),   # SEQN
        (14, 15),  # ELIGSTAT
        (15, 16),  # MORTSTAT
        (42, 46),  # PERMTH_INT
        (46, 50),  # PERMTH_EXM
    ]
    names = ["SEQN", "ELIGSTAT", "MORTSTAT", "PERMTH_INT", "PERMTH_EXM"]
    df = pd.read_fwf(path, colspecs=colspecs, names=names, na_values=[".", " "])
    df["SEQN"] = df["SEQN"].astype(int)
    return df[["SEQN", "ELIGSTAT", "MORTSTAT", "PERMTH_EXM"]]


def load_one_cycle(cycle: dict) -> pd.DataFrame:
    """Load and merge all components for a single cycle."""
    folder = RAW_DIR / cycle["folder"]
    label  = cycle["label"]
    print(f"\n  Loading cycle {label}...")

    # Demographics
    demo_raw = read_xpt(folder / cycle["demo"])
    demo = select_rename(demo_raw, DEMO_COLS)
    # If both RIDRETH3 and RIDRETH1 present, RIDRETH3 takes priority (already handled)
    # If race column missing after rename, try fallback
    if "race" not in demo.columns and "RIDRETH1" in demo_raw.columns:
        demo["race"] = demo_raw["RIDRETH1"]
    print(f"    demo:  {demo.shape[0]:,} rows")

    # Biochemistry
    bio_raw = read_xpt(folder / cycle["bio"])
    bio = select_rename(bio_raw, BIO_COLS)
    # Deduplicate if both LBXSAL and LBDSALSI present
    bio = bio.loc[:, ~bio.columns.duplicated()]
    print(f"    bio:   {bio.shape[0]:,} rows, cols: {[c for c in bio.columns if c != 'SEQN']}")

    # CBC
    cbc_raw = read_xpt(folder / cycle["cbc"])
    cbc = select_rename(cbc_raw, CBC_COLS)
    cbc = cbc.loc[:, ~cbc.columns.duplicated()]  # drop duplicate cols if both LBXLYPCT and LBDLYMNO present
    print(f"    cbc:   {cbc.shape[0]:,} rows, cols: {[c for c in cbc.columns if c != 'SEQN']}")

    # CRP
    crp_raw = read_xpt(folder / cycle["crp"])
    crp = select_rename(crp_raw, CRP_COLS)
    print(f"    crp:   {crp.shape[0]:,} rows")

    # Merge on SEQN (left join from demo)
    df = demo.merge(bio,  on="SEQN", how="left")
    df = df.merge(cbc,   on="SEQN", how="left")
    df = df.merge(crp,   on="SEQN", how="left")

    # Mortality
    if cycle["has_mort"]:
        mort_path = RAW_DIR / "mortality" / cycle["mortality"]
        mort = parse_mortality(mort_path)
        df = df.merge(mort, on="SEQN", how="left")
        print(f"    mort:  {mort.shape[0]:,} rows")
    else:
        df["MORTSTAT"]  = np.nan
        df["PERMTH_EXM"] = np.nan
        df["ELIGSTAT"]  = np.nan

    df["cycle"] = label
    print(f"    merged: {df.shape[0]:,} rows, {df.shape[1]} cols")
    return df


# ── Main pipeline ─────────────────────────────────────────────────────────────

def preprocess(save: bool = True) -> pd.DataFrame:
    print("=" * 55)
    print("NHANES Preprocessing Pipeline")
    print("=" * 55)

    # Load and stack all cycles
    frames = [load_one_cycle(c) for c in CYCLES]
    df = pd.concat(frames, ignore_index=True)
    print(f"\nCombined: {df.shape[0]:,} rows, {df.shape[1]} cols")

    # ── Standardize column names to lowercase ─────────────────────────────
    df.columns = [c.lower() for c in df.columns]

    # ── Age filter: 20–85 (NHANES releases 80+ as 80 in some cycles) ──────
    n_before = len(df)
    df = df[df["age"].between(20, 85)].copy()
    print(f"Age filter 20–85: {n_before:,} → {len(df):,} rows")

    # ── Eligibility filter: keep only exam-eligible participants ──────────
    # ELIGSTAT=1 means eligible for mortality follow-up
    # For 2017-2020 (no mortality), eligstat is NaN — keep all
    n_before = len(df)
    df = df[(df["eligstat"] == 1) | (df["eligstat"].isna())].copy()
    print(f"Eligibility filter: {n_before:,} → {len(df):,} rows")

    # ── Define the 9 PhenoAge biomarker columns ───────────────────────────
    biomarker_cols = [
        "albumin_gdl", "creatinine_mgdl", "glucose_mgdl", "crp_mgdl",
        "lymph_pct", "mcv_fl", "rdw_pct", "alp_ul", "wbc_si"
    ]

    # Report missingness before imputation
    print("\nMissingness per biomarker (before imputation):")
    for col in biomarker_cols:
        if col in df.columns:
            n_miss = df[col].isna().sum()
            pct = 100 * n_miss / len(df)
            print(f"  {col:20s}: {n_miss:,} missing ({pct:.1f}%)")
        else:
            print(f"  {col:20s}: COLUMN NOT FOUND")

    # ── Drop rows missing ALL biomarkers (no lab draw) ────────────────────
    existing_bio = [c for c in biomarker_cols if c in df.columns]
    n_before = len(df)
    df = df.dropna(subset=existing_bio, how="all").copy()
    print(f"\nDrop rows with no lab data: {n_before:,} → {len(df):,} rows")

    # ── Physiologically implausible value removal ─────────────────────────
    # Values outside these ranges are measurement errors or data entry issues
    bounds = {
        "albumin_gdl":      (1.0,  6.0),
        "creatinine_mgdl":  (0.1, 15.0),
        "glucose_mgdl":     (40,  600),
        "crp_mgdl":         (0.0,  20.0),
        "lymph_pct":        (1.0,  80.0),
        "mcv_fl":           (50,   130),
        "rdw_pct":          (9.0,  30.0),
        "alp_ul":           (10,   500),
        "wbc_si":           (0.5,  50.0),
    }
    for col, (lo, hi) in bounds.items():
        if col in df.columns:
            n_out = ((df[col] < lo) | (df[col] > hi)).sum()
            if n_out > 0:
                print(f"  Clipping {n_out} out-of-range values in {col}")
            df[col] = df[col].clip(lo, hi)

    # ── Compute PhenoAge ──────────────────────────────────────────────────
    df = compute_phenoage(df)

    # ── Final column selection and ordering ───────────────────────────────
    keep = ["seqn", "cycle", "age", "gender", "race"] + existing_bio + \
           ["phenoage", "bio_age_accel", "mortstat", "permth_exm"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()

    print(f"\nFinal dataset: {df.shape[0]:,} rows, {df.shape[1]} cols")
    print(f"Cycles: {df['cycle'].value_counts().to_dict()}")
    print(f"Deceased (mortality cycles): "
          f"{int(df['mortstat'].sum())} / {int(df['mortstat'].notna().sum())}")

    if save:
        out = PROC_DIR / "nhanes_merged.parquet"
        df.to_parquet(out, index=False)
        print(f"\nSaved → {out}")

    return df


def compute_phenoage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute PhenoAge using the Levine 2018 formula.
    All biomarkers in US clinical units (as collected in NHANES).

    Formula verified against published coefficients from:
    Levine et al. (2018) Aging (Albany NY) Table S8.

    Units used:
      albumin     : g/dL
      creatinine  : mg/dL
      glucose     : mg/dL
      crp         : mg/dL  (natural log applied internally)
      lymph_pct   : %
      mcv         : fL
      rdw         : %
      alp         : U/L
      wbc         : 1000 cells/uL
    """
    d = df.copy()

    # CRP: natural log of mg/dL
    log_crp = np.log(d["crp_mgdl"].clip(lower=0.001))

    # Linear predictor xb
    xb = (
        -19.9067
        + (-0.0336  * d["albumin_gdl"])
        + ( 0.0095  * d["creatinine_mgdl"] * 88.402)   # convert mg/dL → umol/L
        + ( 0.1953  * d["glucose_mgdl"]    * 0.05551)  # convert mg/dL → mmol/L
        + ( 0.0954  * (log_crp + np.log(10)))           # convert mg/dL → mg/L then ln
        + (-0.0120  * d["lymph_pct"])
        + (-0.0268  * d["mcv_fl"])
        + ( 0.3306  * d["rdw_pct"])
        + ( 0.00188 * d["alp_ul"])
        + ( 0.0554  * d["wbc_si"])
    )

    # Gompertz mortality score (Levine 2018 parameters)
    b    = 0.0076927
    mu0  = 1.51714e-5
    c2   = 0.00553

    M = 1 - np.exp(-mu0 * np.exp(xb) * (np.exp(b * 120) - 1) / b)
    M  = M.clip(1e-10, 1 - 1e-10)

    phenoage = 141.50225 + (1 / b) * np.log((1 / c2) * (-np.log(1 - M)))

    d["phenoage"]      = phenoage.round(2)
    d["bio_age_accel"] = (phenoage - d["age"]).round(2)

    valid = d["phenoage"].between(0, 150)
    print(f"\nPhenoAge computed for {valid.sum():,} participants")
    print(f"  Mean PhenoAge:        {d.loc[valid,'phenoage'].mean():.1f} yrs")
    print(f"  Mean Chrono Age:      {d.loc[valid,'age'].mean():.1f} yrs")
    print(f"  Mean Accel (bio-chr): {d.loc[valid,'bio_age_accel'].mean():.1f} yrs")

    return d


if __name__ == "__main__":
    df = preprocess(save=True)
    print("\nSample (first 5 rows):")
    print(df[["seqn", "cycle", "age", "albumin_gdl", "crp_mgdl",
              "phenoage", "bio_age_accel", "mortstat"]].head().to_string())
