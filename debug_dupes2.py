"""Run from project root: python3 debug_dupes2.py"""
import pyreadstat, pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")

DEMO_COLS = {"SEQN":"SEQN","RIDAGEYR":"age","RIAGENDR":"gender","RIDRETH3":"race","RIDRETH1":"race"}
BIO_COLS  = {"SEQN":"SEQN","LBXSAL":"albumin_gdl","LBXSCR":"creatinine_mgdl","LBXSGL":"glucose_mgdl","LBXSAPSI":"alp_ul","LBDSALSI":"albumin_gdl","LBDSCRLC":"creatinine_mgdl","LBDSGLU":"glucose_mgdl"}
CBC_COLS  = {"SEQN":"SEQN","LBXLYPCT":"lymph_pct","LBXMCVSI":"mcv_fl","LBXRDW":"rdw_pct","LBXWBCSI":"wbc_si","LBDLYMNO":"lymph_pct"}
CRP_COLS  = {"SEQN":"SEQN","LBXHSCRP":"crp_mgdl","LBDHSCRP":"crp_mgdl"}

def select_rename(df, col_map):
    available = {k:v for k,v in col_map.items() if k in df.columns}
    result = df[list(available.keys())].rename(columns=available)
    result = result.loc[:, ~result.columns.duplicated()]
    return result

CYCLES = [
    ("2015-2016","2015_2016","DEMO_I.XPT","BIOPRO_I.XPT","CBC_I.XPT","HSCRP_I.XPT"),
    ("2017-2018","2017_2018","DEMO_J.XPT","BIOPRO_J.XPT","CBC_J.XPT","HSCRP_J.XPT"),
    ("2017-2020","2017_2020","P_DEMO.xpt","P_BIOPRO.xpt","P_CBC.xpt","P_HSCRP.xpt"),
]

frames = []
for label, folder, d, b, c, r in CYCLES:
    base = RAW_DIR / folder
    demo, _ = pyreadstat.read_xport(str(base/d))
    bio,  _ = pyreadstat.read_xport(str(base/b))
    cbc,  _ = pyreadstat.read_xport(str(base/c))
    crp,  _ = pyreadstat.read_xport(str(base/r))

    demo = select_rename(demo, DEMO_COLS)
    bio  = select_rename(bio,  BIO_COLS)
    cbc  = select_rename(cbc,  CBC_COLS)
    crp  = select_rename(crp,  CRP_COLS)

    merged = demo.merge(bio, on="SEQN", how="left")
    merged = merged.merge(cbc, on="SEQN", how="left")
    merged = merged.merge(crp, on="SEQN", how="left")
    merged["cycle"] = label

    dupes = [c for c in merged.columns if list(merged.columns).count(c) > 1]
    print(f"{label}: {merged.shape} | cols: {list(merged.columns)}")
    print(f"  dupes: {dupes if dupes else 'none'}")
    frames.append(merged)

print("\nAttempting concat...")
try:
    df = pd.concat(frames, ignore_index=True)
    print(f"Success: {df.shape}")
except Exception as e:
    print(f"FAILED: {e}")
    for i, f in enumerate(frames):
        print(f"  Frame {i} index unique: {f.index.is_unique}, cols unique: {len(f.columns)==len(set(f.columns))}")
