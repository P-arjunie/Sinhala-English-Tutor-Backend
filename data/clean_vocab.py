import argparse
import os
import re
import sys
import pandas as pd

SINHALA_BLOCK = (0x0D80, 0x0DFF)

def has_sinhala(text: str) -> bool:
    if not isinstance(text, str):
        return False
    for ch in text:
        c = ord(ch)
        if SINHALA_BLOCK[0] <= c <= SINHALA_BLOCK[1]:
            return True
    return False

def word_count(text: str) -> int:
    if not isinstance(text, str) or not text.strip():
        return 0
    return len(re.findall(r"\b\w+\b", text, flags=re.UNICODE))

def normalize_spaces(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("\u200d", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def strip_placeholders(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\[(?:unk|unknown|unkown|UNK|Unknown|Unkown)\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def contains_placeholder(text: str) -> bool:
    if not isinstance(text, str):
        return False
    return bool(re.search(r"\[(?:unk|unknown|unkown)\]", text, flags=re.IGNORECASE))

def clean_df(df: pd.DataFrame,
             max_words: int = 20,
             drop_if_translit_placeholder: bool = True) -> pd.DataFrame:
    cols = ["sinhala", "english", "transliteration", "pos", "example_si", "example_en"]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].fillna("").astype(str)

    df["sinhala"] = df["sinhala"].map(normalize_spaces)
    df["english"] = df["english"].map(normalize_spaces)
    df["transliteration"] = df["transliteration"].map(normalize_spaces).map(strip_placeholders)
    df["pos"] = df["pos"].map(normalize_spaces)
    df["example_si"] = df["example_si"].map(normalize_spaces)
    df["example_en"] = df["example_en"].map(normalize_spaces)

    initial = len(df)
    reasons = {}

    mask_valid_si = df["sinhala"].apply(has_sinhala)
    reasons["no_sinhala_script"] = int((~mask_valid_si).sum())
    df = df[mask_valid_si]

    si_wc = df["sinhala"].apply(word_count)
    en_wc = df["english"].apply(word_count)
    mask_len = (si_wc <= max_words) & (en_wc <= max_words)
    reasons["too_long"] = int((~mask_len).sum())
    df = df[mask_len]

    mask_nonempty = df["sinhala"].str.len() > 0
    mask_nonempty &= df["english"].str.len() > 0
    reasons["empty_fields"] = int((~mask_nonempty).sum())
    df = df[mask_nonempty]

    if drop_if_translit_placeholder:
        mask_placeholder = df["transliteration"].apply(contains_placeholder)
        reasons["translit_placeholder"] = int(mask_placeholder.sum())
        df = df[~mask_placeholder]

    before_dedup = len(df)
    df = df.drop_duplicates(subset=["sinhala", "english"]).reset_index(drop=True)
    reasons["deduplicated"] = int(before_dedup - len(df))

    df = df.reset_index(drop=True)

    removed_total = initial - len(df)
    stats = {
        "initial": initial,
        "final": len(df),
        "removed_total": removed_total,
    }
    stats.update(reasons)

    return df, stats

def main():
    parser = argparse.ArgumentParser(description="Clean vocab dataset and export cleaned CSV.")
    parser.add_argument("--input", default=os.path.join("data", "vocab.csv"), help="Input CSV path")
    parser.add_argument("--output", default=os.path.join("data", "vocab_clean.csv"), help="Output CSV path")
    parser.add_argument("--max-words", type=int, default=20, help="Max words per field for sinhala/english")
    parser.add_argument("--keep-translit-placeholders", action="store_true", help="Do not drop rows with transliteration placeholders")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.input)
    cleaned, stats = clean_df(df, max_words=args.max_words, drop_if_translit_placeholder=not args.keep_translit_placeholders)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    cleaned.to_csv(args.output, index=False)

    print("Cleaning summary:")
    for k, v in stats.items():
        print(f"- {k}: {v}")
    print(f"Saved cleaned data to: {args.output}")

if __name__ == "__main__":
    main()
