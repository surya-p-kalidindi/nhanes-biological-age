# NHANES Biological Age Estimator

Predicting **biological age** from clinical biomarkers using NHANES public data and the PhenoAge framework (Levine et al. 2018).

## Project Goal

Identify which blood biomarkers most strongly predict biological aging, and build a model that estimates how "old" a person is biologically — independent of their calendar age.

## Scientific Foundation

Built on **PhenoAge** (Levine et al., *Aging*, 2018) — a validated biological age clock derived from 9 clinical biomarkers available in routine blood panels:

| Biomarker | Unit | Direction with aging |
|---|---|---|
| Albumin | g/dL | ↓ |
| Creatinine | mg/dL | ↑ |
| Glucose | mg/dL | ↑ |
| C-Reactive Protein (CRP) | mg/dL | ↑ |
| Lymphocyte % | % | ↓ |
| Mean Corpuscular Volume (MCV) | fL | ↑ |
| Red Cell Distribution Width (RDW) | % | ↑ |
| Alkaline Phosphatase (ALP) | U/L | ↑ |
| White Blood Cell Count (WBC) | 1000/µL | ↑ |

## Project Structure

```
nhanes-biological-age/
├── data/
│   ├── raw/          # NHANES XPT files (downloaded by data_loader.py)
│   └── processed/    # Cleaned, merged datasets
├── notebooks/        # EDA and modeling notebooks (coming)
├── src/
│   ├── data_loader.py    # Downloads NHANES data from CDC
│   └── synthetic_data.py # Synthetic data generator for dev/CI
├── results/          # Model outputs, plots, SHAP values
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/nhanes-biological-age
cd nhanes-biological-age
pip install -r requirements.txt
```

## Getting the Data

NHANES data is fully public — no data use agreement required.

```bash
python3 src/data_loader.py
```

This downloads ~15 MB of XPT files from CDC for cycles 2015–2016, 2017–2018, and 2017–2020 into `data/raw/`. Mortality linkage files are also downloaded automatically.

> **Note:** The download script must be run from your local machine. CDC blocks automated downloads from some cloud environments.

### Development Without Real Data

A synthetic dataset (n=5,000) with realistic NHANES-matched distributions is available for pipeline development:

```python
from src.synthetic_data import generate_synthetic_nhanes
df = generate_synthetic_nhanes(n=5000, seed=42)
```

## Roadmap

- [x] Project structure
- [x] NHANES data downloader (2015–2020 cycles)
- [x] Synthetic data generator (calibrated to Levine 2018 Table 1)
- [ ] Preprocessing pipeline (merge, impute, unit harmonization)
- [ ] PhenoAge formula replication + validation
- [ ] XGBoost/LightGBM biological age model
- [ ] SHAP feature importance analysis
- [ ] Survival analysis (mortality ~ biological age acceleration)
- [ ] Inference script + model card

## References

Levine, M.E. et al. (2018). An epigenetic biomarker of aging for lifespan and healthspan. *Aging (Albany NY)*, 10(4), 573–591. https://doi.org/10.18632/aging.101414

## Data Source

National Health and Nutrition Examination Survey (NHANES), CDC/NCHS.
Public domain. No DUA required. https://www.cdc.gov/nchs/nhanes/
