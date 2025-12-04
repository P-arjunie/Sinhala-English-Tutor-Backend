import os
from pathlib import Path
import zipfile
import pandas as pd

# Configure Kaggle credentials: place kaggle.json in %USERPROFILE%\.kaggle\kaggle.json or set envs
# KAGGLE_USERNAME and KAGGLE_KEY

DATA_DIR = Path(__file__).parent
REPO_ROOT = DATA_DIR.parent
LOCAL_KAGGLE_JSON = REPO_ROOT / "kaggle.json"

# If a local kaggle.json exists in the repo root, direct Kaggle API to use it
if LOCAL_KAGGLE_JSON.exists():
    os.environ.setdefault("KAGGLE_CONFIG_DIR", str(REPO_ROOT))
OUT_CSV = DATA_DIR / "vocab.csv"

# Example dataset slug; replace with the exact one used in the notebook if different.
# This is a placeholder; user should confirm the dataset.
KAGGLE_DATASET = os.getenv("KAGGLE_DATASET", "programmerrdai/sinhala-english-singlish-translation-dataset")


def download_and_prepare():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as e:
        raise RuntimeError("kaggle package not installed. Run: pip install kaggle") from e

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as auth_err:
        raise RuntimeError(
            "Kaggle authentication failed. Ensure one of the following:\n"
            "- %USERPROFILE%\\.kaggle\\kaggle.json exists, or\n"
            "- Environment variables KAGGLE_USERNAME and KAGGLE_KEY are set, or\n"
            "- Place kaggle.json in the repo root and we will use it (not recommended to commit)."
        ) from auth_err

    dest = DATA_DIR / "kaggle_download"
    dest.mkdir(exist_ok=True)

    # Download dataset files
    api.dataset_download_files(KAGGLE_DATASET, path=str(dest), unzip=False)
    zips = list(dest.glob("*.zip"))
    if not zips:
        raise RuntimeError("No zip file downloaded from Kaggle dataset.")

    # Unzip first zip
    zpath = zips[0]
    with zipfile.ZipFile(zpath, 'r') as zf:
        zf.extractall(dest)

    # Heuristic: find a CSV and map columns to expected format
    csvs = list(dest.glob("*.csv"))
    if not csvs:
        raise RuntimeError("No CSV found after unzip.")

    df = pd.read_csv(csvs[0])

    # Attempt column mapping
    # Column mapping for multiple dataset schemas.
    # For 'Sinhala-English-Singlish Translation Dataset', columns are likely: Sinhala, English, Singlish
    mapping_candidates = [
        ("Sinhala", "sinhala"), ("si", "sinhala"), ("si_word", "sinhala"), ("sinhala", "sinhala"),
        ("English", "english"), ("en", "english"), ("en_word", "english"), ("english", "english"),
        ("Singlish", "transliteration"), ("Transliteration", "transliteration"), ("roman", "transliteration"), ("transliteration", "transliteration"),
        ("POS", "pos"), ("pos", "pos"),
        ("Example_SI", "example_si"), ("example_si", "example_si"),
        ("Example_EN", "example_en"), ("example_en", "example_en"),
    ]
    colmap = {}
    for src, tgt in mapping_candidates:
        if src in df.columns:
            colmap[src] = tgt
    df = df.rename(columns=colmap)

    # Normalize to expected columns
    expected = ["sinhala", "english", "transliteration", "pos", "example_si", "example_en"]
    for col in expected:
        if col not in df.columns:
            df[col] = ""
    df = df[expected].fillna("")
    # Coerce types to string to avoid NaN/None in API
    for col in expected:
        df[col] = df[col].astype(str)

    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    return OUT_CSV

if __name__ == "__main__":
    out = download_and_prepare()
    print(f"Saved: {out}")
