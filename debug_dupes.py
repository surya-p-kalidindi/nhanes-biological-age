"""Run from project root: python3 debug_dupes.py"""
import pyreadstat
from pathlib import Path

RAW_DIR = Path("data/raw")

files = [
    ("2015_2016", "DEMO_I.XPT"),
    ("2015_2016", "BIOPRO_I.XPT"),
    ("2015_2016", "CBC_I.XPT"),
    ("2015_2016", "HSCRP_I.XPT"),
    ("2017_2018", "DEMO_J.XPT"),
    ("2017_2018", "BIOPRO_J.XPT"),
    ("2017_2018", "CBC_J.XPT"),
    ("2017_2018", "HSCRP_J.XPT"),
    ("2017_2020", "P_DEMO.xpt"),
    ("2017_2020", "P_BIOPRO.xpt"),
    ("2017_2020", "P_CBC.xpt"),
    ("2017_2020", "P_HSCRP.xpt"),
]

for folder, fname in files:
    path = RAW_DIR / folder / fname
    df, _ = pyreadstat.read_xport(str(path))
    dupes = [c for c in df.columns if list(df.columns).count(c) > 1]
    print(f"{folder}/{fname:20s}: {df.shape[1]} cols | dupes: {dupes if dupes else 'none'}")
