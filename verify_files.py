"""
Run this from your project root:
  python3 verify_files.py
"""
from pathlib import Path
import pyreadstat

RAW_DIR = Path("data/raw")

def is_valid_xpt(path):
    if not path.exists() or path.stat().st_size < 50_000:
        return False
    try:
        with open(path, 'rb') as f:
            header = f.read(80)
        return b'HEADER RECORD' in header or b'LIBRARY' in header
    except:
        return False

print("=== XPT Files ===")
all_ok = True
expected = {
    "2015_2016": ["DEMO_I.XPT", "BIOPRO_I.XPT", "CBC_I.XPT", "HSCRP_I.XPT"],
    "2017_2018": ["DEMO_J.XPT", "BIOPRO_J.XPT", "CBC_J.XPT", "HSCRP_J.XPT"],
    "2017_2020": ["P_DEMO.xpt", "P_BIOPRO.xpt", "P_CBC.xpt", "P_HSCRP.xpt"],
}
for folder, files in expected.items():
    for fname in files:
        path = RAW_DIR / folder / fname
        if is_valid_xpt(path):
            size_kb = path.stat().st_size // 1024
            print(f"  [OK]      {folder}/{fname} ({size_kb:,} KB)")
        else:
            exists = path.exists()
            size = path.stat().st_size if exists else 0
            print(f"  [MISSING] {folder}/{fname} — exists={exists}, size={size}")
            all_ok = False

print("\n=== Mortality Files ===")
for fname in ["NHANES_2015_2016_MORT_2019_PUBLIC.dat", "NHANES_2017_2018_MORT_2019_PUBLIC.dat"]:
    path = RAW_DIR / "mortality" / fname
    if path.exists() and path.stat().st_size > 100_000:
        print(f"  [OK]      {fname} ({path.stat().st_size // 1024:,} KB)")
    else:
        print(f"  [MISSING] {fname}")
        all_ok = False

print()
if all_ok:
    print("All files present and valid — ready to load!")
    print("\n=== Quick load test ===")
    for path_str, label in [
        ("2015_2016/DEMO_I.XPT",    "DEMO_I"),
        ("2015_2016/BIOPRO_I.XPT",  "BIOPRO_I"),
        ("2015_2016/CBC_I.XPT",     "CBC_I"),
        ("2017_2020/P_DEMO.xpt",    "P_DEMO"),
        ("2017_2020/P_BIOPRO.xpt",  "P_BIOPRO"),
    ]:
        df, _ = pyreadstat.read_xport(str(RAW_DIR / path_str))
        print(f"  {label:12s}: {df.shape[0]:,} rows, {df.shape[1]} cols")
else:
    print("Some files missing — fix above then re-run.")
