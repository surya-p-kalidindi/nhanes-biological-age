"""
NHANES Data Loader
==================
Downloads and caches NHANES XPT files for biological age estimation.

Cycles covered: 2015-2016, 2017-2018, 2017-2020 (pre-pandemic)
Biomarkers targeted (PhenoAge panel):
  - Albumin (LBDSALSI -> LBXSAL)
  - Creatinine (LBXSCR)
  - Glucose (LBXSGL)
  - C-Reactive Protein (LBXHSCRP)
  - Lymphocyte % (LBXLYPCT)
  - Mean Corpuscular Volume (LBXMCVSI)
  - Red Cell Distribution Width (LBXRDW)
  - Alkaline Phosphatase (LBXSAPSI)
  - White Blood Cell count (LBXWBCSI)
  - Chronological Age (RIDAGEYR)

Mortality linkage via NCHS Public Use Linked Mortality Files.
"""

import os
import requests
import pandas as pd
import pyreadstat
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

NHANES_BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes"

# ── NHANES file manifest ─────────────────────────────────────────────────────
# Format: (cycle_label, component, filename)
NHANES_FILES = [
    # ── 2015-2016 ─────────────────────────────────────────────────────────
    ("2015-2016", "demographics", "DEMO_I.XPT"),
    ("2015-2016", "biochemistry", "BIOPRO_I.XPT"),
    ("2015-2016", "CBC",          "CBC_I.XPT"),
    ("2015-2016", "CRP",          "HSCRP_I.XPT"),

    # ── 2017-2018 ─────────────────────────────────────────────────────────
    ("2017-2018", "demographics", "DEMO_J.XPT"),
    ("2017-2018", "biochemistry", "BIOPRO_J.XPT"),
    ("2017-2018", "CBC",          "CBC_J.XPT"),
    ("2017-2018", "CRP",          "HSCRP_J.XPT"),

    # ── 2017-2020 (pre-pandemic, cycle P) ─────────────────────────────────
    ("2017-2020", "demographics", "DEMO_P.XPT"),
    ("2017-2020", "biochemistry", "BIOPRO_P.XPT"),
    ("2017-2020", "CBC",          "CBC_P.XPT"),
    ("2017-2020", "CRP",          "HSCRP_P.XPT"),
]

# URL patterns to try for each file (CDC has changed URL structure over time)
def get_urls(cycle: str, filename: str) -> list[str]:
    """Return candidate URLs in priority order."""
    year_start = cycle.split("-")[0]
    return [
        # Current CDC URL structure (2024+)
        f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year_start}/{filename}",
        # Older structure
        f"https://wwwn.cdc.gov/Nchs/Nhanes/{cycle}/{filename}",
        # Alternative path
        f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/{cycle}/{filename}",
    ]

MORTALITY_FILES = [
    ("2015-2016", "NHANES_2015_2016_MORT_2019_PUBLIC.dat",
     "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/NHANES_2015_2016_MORT_2019_PUBLIC.dat"),
    ("2017-2018", "NHANES_2017_2018_MORT_2019_PUBLIC.dat",
     "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/NHANES_2017_2018_MORT_2019_PUBLIC.dat"),
]

MIN_VALID_SIZE_BYTES = 50_000  # Real XPT files are at least 50KB


def is_valid_xpt(path: Path) -> bool:
    """Check if file looks like a real XPT (SAS transport) file."""
    if not path.exists():
        return False
    if path.stat().st_size < MIN_VALID_SIZE_BYTES:
        return False
    # XPT files start with 'HEADER RECORD'
    try:
        with open(path, 'rb') as f:
            header = f.read(80)
        return b'HEADER RECORD' in header or b'LIBRARY' in header
    except Exception:
        return False


def download_file(urls: list[str], dest: Path, label: str = "") -> bool:
    """
    Try each URL in sequence until one succeeds and returns a valid file.
    Returns True on success.
    """
    if is_valid_xpt(dest):
        print(f"  [cached]  {dest.name} ({dest.stat().st_size // 1024} KB)")
        return True

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Accept": "application/octet-stream, */*",
        "Referer": "https://wwwn.cdc.gov/nchs/nhanes/",
    }

    for url in urls:
        print(f"  [trying]  {url} ...", end=" ", flush=True)
        try:
            r = requests.get(url, headers=headers, timeout=120, stream=True)
            if r.status_code != 200:
                print(f"HTTP {r.status_code}")
                continue

            # Stream to temp file first
            tmp = dest.with_suffix(".tmp")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)

            size_kb = tmp.stat().st_size // 1024

            # Validate it's actually an XPT file
            if is_valid_xpt(tmp):
                tmp.rename(dest)
                print(f"done ({size_kb} KB)")
                return True
            else:
                print(f"invalid file ({size_kb} KB) — CDC may have returned HTML")
                tmp.unlink(missing_ok=True)

        except Exception as e:
            print(f"error: {e}")

    print(f"  [FAILED]  All URLs failed for {label}")
    return False


def download_mortality(url: str, dest: Path, label: str = "") -> bool:
    """Download mortality .dat file."""
    if dest.exists() and dest.stat().st_size > 100_000:
        print(f"  [cached]  {dest.name} ({dest.stat().st_size // 1024} KB)")
        return True

    print(f"  [downloading] {label} ...", end=" ", flush=True)
    try:
        r = requests.get(url, timeout=120, stream=True)
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
        print(f"done ({dest.stat().st_size // 1024} KB)")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False


def load_xpt(path: Path) -> pd.DataFrame:
    """Read a SAS XPT file into a DataFrame."""
    df, _ = pyreadstat.read_xport(str(path))
    return df


def parse_mortality(path: Path) -> pd.DataFrame:
    """Parse NCHS fixed-width linked mortality file."""
    colspecs = [
        (0,  6),   # SEQN
        (14, 15),  # ELIGSTAT
        (15, 16),  # MORTSTAT
        (16, 19),  # UCOD_LEADING
        (19, 20),  # DIABETES
        (20, 21),  # HYPERTEN
        (42, 46),  # PERMTH_INT
        (46, 50),  # PERMTH_EXM
    ]
    names = ["SEQN", "ELIGSTAT", "MORTSTAT", "UCOD_LEADING",
             "DIABETES", "HYPERTEN", "PERMTH_INT", "PERMTH_EXM"]
    df = pd.read_fwf(path, colspecs=colspecs, names=names, na_values=[".", " "])
    df["SEQN"] = df["SEQN"].astype(int)
    return df


def download_all() -> None:
    """Download all NHANES + mortality files."""
    print("\n=== Downloading NHANES Lab + Demographics files ===")
    success, failed = 0, []
    for cycle, component, filename in NHANES_FILES:
        cycle_dir = RAW_DIR / cycle.replace("-", "_")
        cycle_dir.mkdir(exist_ok=True)
        dest = cycle_dir / filename
        urls = get_urls(cycle, filename)
        ok = download_file(urls, dest, label=f"{cycle} {component}")
        if ok:
            success += 1
        else:
            failed.append(f"{cycle} {component} ({filename})")

    print("\n=== Downloading Mortality Linkage files ===")
    mort_dir = RAW_DIR / "mortality"
    mort_dir.mkdir(exist_ok=True)
    for cycle, filename, url in MORTALITY_FILES:
        download_mortality(url, mort_dir / filename, label=f"{cycle} mortality")

    print(f"\n=== Summary: {success}/{len(NHANES_FILES)} lab files downloaded ===")
    if failed:
        print("Failed:")
        for f in failed:
            print(f"  - {f}")
        print("\nIf files failed, try downloading manually from:")
        print("  https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/")


def load_cycle(cycle: str) -> dict[str, pd.DataFrame]:
    """Load all components for a given cycle."""
    cycle_dir = RAW_DIR / cycle.replace("-", "_")
    frames = {}
    component_map = {k: [] for k in ["demographics", "biochemistry", "CBC", "CRP"]}
    for c, comp, filename in NHANES_FILES:
        if c == cycle:
            component_map[comp].append(filename)

    for component, filenames in component_map.items():
        if filenames:
            path = cycle_dir / filenames[0]
            if is_valid_xpt(path):
                frames[component] = load_xpt(path)
            else:
                print(f"  [missing/invalid] {path.name} — run download_all() first")
    return frames


def load_mortality(cycle: str) -> pd.DataFrame | None:
    """Load mortality linkage for a cycle."""
    mort_dir = RAW_DIR / "mortality"
    for c, filename, _ in MORTALITY_FILES:
        if c == cycle:
            path = mort_dir / filename
            if path.exists() and path.stat().st_size > 100_000:
                return parse_mortality(path)
            else:
                print(f"  [missing] {path.name}")
    return None


if __name__ == "__main__":
    download_all()
    print("\n=== Verifying loads ===")
    for cycle in ["2015-2016", "2017-2018"]:
        frames = load_cycle(cycle)
        mort = load_mortality(cycle)
        print(f"\nCycle {cycle}:")
        for k, df in frames.items():
            print(f"  {k:15s}: {df.shape[0]:,} rows, {df.shape[1]} cols")
        if mort is not None:
            print(f"  {'mortality':15s}: {mort.shape[0]:,} rows")
