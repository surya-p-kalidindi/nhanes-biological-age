"""
Synthetic NHANES Data Generator
================================
Generates realistic synthetic data that mirrors NHANES lab distributions
for development, testing, and CI pipelines when real data is unavailable.

Statistical parameters sourced from:
  Levine et al. (2018) "An epigenetic biomarker of aging for lifespan and
  healthspan." Aging (Albany NY). DOI: 10.18632/aging.101414
  Table 1: Sample characteristics, NHANES III

All continuous biomarkers use log-normal or normal distributions
calibrated to published means ± SDs. Correlation structure approximated
from the paper's correlation table.

NOTE: This synthetic data is for pipeline development only.
Real analysis must use actual NHANES data (see data_loader.py).
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── Published biomarker statistics (Levine 2018, Table 1) ──────────────────
# (mean, std) in original units; age 30-85 general population
BIOMARKER_PARAMS = {
    # name              mean    std    dist     unit         lo    hi
    "Albumin":        (4.14,  0.36,  "normal", "g/dL",     2.5,  5.5),
    "Creatinine":     (0.95,  0.30,  "lognorm","mg/dL",    0.3,  8.0),
    "Glucose":        (98.0,  21.0,  "lognorm","mg/dL",    60,   350),
    "CRP":            (0.30,  0.55,  "lognorm","mg/dL",    0.001, 10.0),
    "LymphPct":       (32.0,  8.5,   "normal", "%",         5,    60),
    "MCV":            (90.0,  5.5,   "normal", "fL",        65,   115),
    "RDW":            (13.1,  1.1,   "normal", "%",         10,   22),
    "ALP":            (75.0,  33.0,  "lognorm","U/L",       20,   300),
    "WBC":            (6.8,   2.1,   "lognorm","1000/uL",   2.0,  20.0),
}

# Approximate correlation matrix for the 9 biomarkers (order matches above)
# Derived from published literature; diagonal = 1.0
# Correlations are modest - most biomarkers weakly correlated
CORR_MATRIX = np.array([
    # Alb   Cr    Glu   CRP   Lym   MCV   RDW   ALP   WBC
    [ 1.00, 0.05, 0.00,-0.15, 0.05, 0.10,-0.20, 0.10,-0.05],  # Albumin
    [ 0.05, 1.00, 0.08,-0.05, 0.00, 0.00, 0.05, 0.05, 0.00],  # Creatinine
    [ 0.00, 0.08, 1.00, 0.12,-0.05,-0.05, 0.05, 0.05, 0.05],  # Glucose
    [-0.15,-0.05, 0.12, 1.00,-0.15, 0.00, 0.15, 0.10, 0.30],  # CRP
    [ 0.05, 0.00,-0.05,-0.15, 1.00,-0.05,-0.10, 0.00,-0.05],  # LymphPct
    [ 0.10, 0.00,-0.05, 0.00,-0.05, 1.00, 0.20,-0.05,-0.10],  # MCV
    [-0.20, 0.05, 0.05, 0.15,-0.10, 0.20, 1.00, 0.05, 0.10],  # RDW
    [ 0.10, 0.05, 0.05, 0.10, 0.00,-0.05, 0.05, 1.00, 0.05],  # ALP
    [-0.05, 0.00, 0.05, 0.30,-0.05,-0.10, 0.10, 0.05, 1.00],  # WBC
])


def _generate_correlated_normals(n: int, corr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Generate correlated standard normal samples via Cholesky decomposition."""
    L = np.linalg.cholesky(corr)
    z = rng.standard_normal((n, corr.shape[0]))
    return z @ L.T


def _apply_marginal(z: np.ndarray, mean: float, std: float,
                    dist: str, lo: float, hi: float) -> np.ndarray:
    """Transform standard normals to target marginal distribution, then clip."""
    if dist == "normal":
        vals = mean + std * z
    elif dist == "lognorm":
        # For lognormal: back-calculate sigma_log from (mean, std) of original
        # Using method of moments: sigma^2 = log(1 + (std/mean)^2)
        sigma_sq = np.log(1 + (std / mean) ** 2)
        mu_log = np.log(mean) - 0.5 * sigma_sq
        sigma_log = np.sqrt(sigma_sq)
        # Convert standard normals to lognormal via quantile transform
        p = stats.norm.cdf(z)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        vals = stats.lognorm.ppf(p, s=sigma_log, scale=np.exp(mu_log))
    else:
        raise ValueError(f"Unknown dist: {dist}")
    return np.clip(vals, lo, hi)


def _age_adjustment(biomarkers: pd.DataFrame, ages: np.ndarray) -> pd.DataFrame:
    """
    Apply age-related shifts to biomarkers.
    In real data, biomarkers drift with age. These approximate known trends:
      - Creatinine slightly increases with age (kidney function decline)
      - Albumin decreases slightly with age
      - CRP increases with age (chronic inflammation)
      - RDW increases with age
      - ALP mildly increases with age
    """
    df = biomarkers.copy()
    age_z = (ages - 55) / 20  # center around ~55, scale

    df["Albumin"]    += -0.05 * age_z
    df["Creatinine"] +=  0.04 * age_z
    df["CRP"]        *=  np.exp(0.15 * age_z)
    df["RDW"]        +=  0.15 * age_z
    df["ALP"]        +=  3.0 * age_z
    df["Glucose"]    +=  3.0 * age_z

    # Re-clip after adjustments
    for col, (mean, std, dist, unit, lo, hi) in BIOMARKER_PARAMS.items():
        df[col] = df[col].clip(lo, hi)

    return df


def compute_phenoage(df: pd.DataFrame) -> pd.Series:
    """
    Compute PhenoAge from the 9 biomarkers using the published formula.

    Levine et al. (2018): Two-step calculation
    Step 1: Compute 'mortality score' (linear combination of biomarkers)
    Step 2: Map mortality score to biological age via an inverse Gompertz function

    Published coefficients (Table 2, Levine 2018):
    """
    # Coefficients from Levine et al. 2018
    # These are the published xb coefficients
    coefficients = {
        "Albumin":    -0.0336,
        "Creatinine":  0.0095,
        "Glucose":     0.1953,
        "CRP":         0.0954,   # log(CRP) in the original
        "LymphPct":   -0.0120,
        "MCV":        -0.0268,
        "RDW":         0.3306,
        "ALP":         0.00188,
        "WBC":         0.0554,
    }
    intercept = -19.9067

    # CRP is entered as natural log
    log_crp = np.log(df["CRP"].clip(lower=0.001))

    xb = (
        intercept
        + coefficients["Albumin"]    * df["Albumin"]
        + coefficients["Creatinine"] * df["Creatinine"]
        + coefficients["Glucose"]    * df["Glucose"]
        + coefficients["CRP"]        * log_crp
        + coefficients["LymphPct"]   * df["LymphPct"]
        + coefficients["MCV"]        * df["MCV"]
        + coefficients["RDW"]        * df["RDW"]
        + coefficients["ALP"]        * df["ALP"]
        + coefficients["WBC"]        * df["WBC"]
    )

    # Step 2: Gompertz inverse — map mortality risk to biological age
    # gamma = 0.0076927 (Gompertz shape), published in Levine 2018
    gamma = 0.0076927
    mortality_score = 1 - np.exp(-np.exp(xb) * (np.exp(120 * gamma) - 1) / gamma)
    mortality_score = mortality_score.clip(1e-6, 1 - 1e-6)

    # Biological age from mortality score via Gompertz inverse
    phenoage = np.log(-np.log(1 - mortality_score) * gamma /
                      (np.exp(0 * gamma) * (np.exp(gamma) - 1))) / gamma

    return phenoage.clip(20, 120)


def generate_synthetic_nhanes(
    n: int = 5000,
    seed: int = 42,
    include_mortality: bool = True,
) -> pd.DataFrame:
    """
    Generate a synthetic NHANES-like DataFrame with realistic biomarker
    distributions, demographic variables, and (optionally) simulated
    mortality outcomes.

    Parameters
    ----------
    n : int
        Number of participants (default 5000, ~NHANES cycle size)
    seed : int
        Random seed for reproducibility
    include_mortality : bool
        Whether to simulate mortality outcomes

    Returns
    -------
    pd.DataFrame with columns:
        SEQN, AgeInYearsAtScreening, Gender, Race, [9 biomarkers],
        PhenoAge, BioAgeAcceleration, [optional: MortalityRisk, Deceased]
    """
    rng = np.random.default_rng(seed)

    # ── Demographics ───────────────────────────────────────────────────────
    ages = rng.integers(20, 86, size=n).astype(float)
    gender = rng.choice([1, 2], size=n, p=[0.49, 0.51])  # 1=Male, 2=Female
    race = rng.choice([1, 2, 3, 4, 5], size=n,
                      p=[0.09, 0.11, 0.12, 0.63, 0.05])
    # 1=Mexican American, 2=Other Hispanic, 3=Non-Hispanic Black,
    # 4=Non-Hispanic White, 5=Other

    # ── Correlated biomarkers ──────────────────────────────────────────────
    z = _generate_correlated_normals(n, CORR_MATRIX, rng)

    bio_cols = list(BIOMARKER_PARAMS.keys())
    biomarkers = {}
    for i, (col, params) in enumerate(BIOMARKER_PARAMS.items()):
        mean, std, dist, unit, lo, hi = params
        biomarkers[col] = _apply_marginal(z[:, i], mean, std, dist, lo, hi)

    df_bio = pd.DataFrame(biomarkers)
    df_bio = _age_adjustment(df_bio, ages)

    # ── PhenoAge calculation ───────────────────────────────────────────────
    phenoage = compute_phenoage(df_bio)
    bio_age_accel = phenoage - ages  # positive = biologically older

    # ── Assemble main DataFrame ────────────────────────────────────────────
    df = pd.DataFrame({
        "SEQN":                  np.arange(1, n + 1),
        "AgeInYearsAtScreening": ages,
        "Gender":                gender,
        "Race":                  race,
    })
    df = pd.concat([df, df_bio], axis=1)
    df["PhenoAge"] = phenoage.values
    df["BioAgeAcceleration"] = bio_age_accel.values

    # ── Simulated mortality (optional) ────────────────────────────────────
    if include_mortality:
        # Mortality probability increases with PhenoAge and age
        # Rough 10-year mortality risk based on age + phenoage
        base_mort_risk = 1 / (1 + np.exp(-(phenoage.values - 75) / 8))
        # Add some noise
        noise = rng.normal(0, 0.02, size=n)
        mort_prob = np.clip(base_mort_risk + noise, 0, 1)

        # Binary outcome: simulated death within 10-year follow-up
        deceased = rng.binomial(1, mort_prob).astype(bool)

        # Follow-up time in months (censored at 120 months = 10 years)
        follow_up = rng.uniform(12, 120, size=n)
        # Those who died have shorter follow-up on average
        follow_up[deceased] = rng.uniform(6, 120, size=deceased.sum())

        df["MortalityRisk"] = mort_prob
        df["Deceased"] = deceased.astype(int)
        df["FollowUpMonths"] = np.round(follow_up, 1)

    # ── Missing data (realistic ~5-15% missingness per lab) ───────────────
    missing_rates = {
        "Albumin":    0.05,
        "Creatinine": 0.04,
        "Glucose":    0.07,
        "CRP":        0.08,
        "LymphPct":   0.06,
        "MCV":        0.05,
        "RDW":        0.05,
        "ALP":        0.06,
        "WBC":        0.05,
    }
    for col, rate in missing_rates.items():
        mask = rng.random(n) < rate
        df.loc[mask, col] = np.nan

    return df


def save_synthetic_data(n: int = 5000, seed: int = 42) -> Path:
    """Generate and save synthetic dataset to processed directory."""
    print(f"Generating synthetic NHANES-like dataset (n={n})...")
    df = generate_synthetic_nhanes(n=n, seed=seed)
    out_path = PROCESSED_DIR / "synthetic_nhanes.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved to {out_path}")
    print(f"Shape: {df.shape}")
    print(f"\nSample statistics:")
    bio_cols = list(BIOMARKER_PARAMS.keys())
    print(df[["AgeInYearsAtScreening"] + bio_cols + ["PhenoAge", "BioAgeAcceleration"]]
          .describe().round(2).to_string())
    return out_path


if __name__ == "__main__":
    save_synthetic_data()
