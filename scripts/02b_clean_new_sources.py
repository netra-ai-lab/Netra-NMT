import re
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Khmer Unicode block: U+1780–U+17FF
KHMER_RE = re.compile(r"[ក-៿]")
REPETITION_RE = re.compile(r"^(\S+\s+)\1{4,}$")  # same word repeated 5+ times


def khmer_ratio(text: str) -> float:
    if not text:
        return 0.0
    khmer_chars = sum(1 for c in text if "ក" <= c <= "៿")
    return khmer_chars / len(text)


def is_valid_common(en: str, km: str) -> bool:
    en = en.strip()
    km = km.strip()

    # Min length
    if len(en) < 5 or len(km) < 5:
        return False

    # Max length
    if len(en) > 1000 or len(km) > 2000:
        return False

    # Must contain at least one word character
    if not re.search(r"\w", en) or not re.search(r"\w", km):
        return False

    # Khmer field must be ≥30% Khmer Unicode characters
    if khmer_ratio(km) < 0.30:
        return False

    # English field must have <5% Khmer Unicode characters
    if khmer_ratio(en) >= 0.05:
        return False

    # Length ratio: km_chars / en_chars must be in [0.2, 10.0]
    ratio = len(km) / len(en)
    if ratio < 0.2 or ratio > 10.0:
        return False

    return True


def is_valid_paracrawl(en: str, km: str) -> bool:
    # Truncation marker
    if en.rstrip().endswith("...") or km.rstrip().endswith("..."):
        return False

    # Repetition (e.g. "ha ha ha ha ha")
    if REPETITION_RE.match(en.strip()) or REPETITION_RE.match(km.strip()):
        return False

    # Min English word count
    if len(en.strip().split()) < 2:
        return False

    return True


def clean(path: Path, source_label: str, extra_fn=None) -> pd.DataFrame:
    df = pd.read_parquet(path)
    before = len(df)
    print(f"\n[{source_label}] Loaded {before:,} rows from {path.name}")

    df["en"] = df["en"].astype(str)
    df["km"] = df["km"].astype(str)

    mask = df.apply(lambda r: is_valid_common(r["en"], r["km"]), axis=1)
    if extra_fn is not None:
        mask &= df.apply(lambda r: extra_fn(r["en"], r["km"]), axis=1)

    df = df[mask].reset_index(drop=True)
    after = len(df)
    print(f"[{source_label}] After cleaning: {after:,} rows  (removed {before - after:,}, kept {after/before:.1%})")

    # Spot-check 5 random rows
    print(f"[{source_label}] Sample rows:")
    for _, row in df.sample(min(5, len(df)), random_state=0).iterrows():
        print(f"  EN: {row['en'][:80]}")
        print(f"  KM: {row['km'][:80]}")
        print()

    return df


# --- ParaCrawl ---
paracrawl_in = ROOT / "data" / "raw" / "paracrawl_v2" / "data.parquet"
paracrawl_out = ROOT / "data" / "raw" / "paracrawl_v2" / "data_clean.parquet"
df_pc = clean(paracrawl_in, "paracrawl_v2", extra_fn=is_valid_paracrawl)
df_pc.to_parquet(paracrawl_out, index=False)
print(f"Saved to {paracrawl_out}")

# --- SeyhaLite ---
seyha_in = ROOT / "data" / "raw" / "seyha_en_kh_all" / "data.parquet"
seyha_out = ROOT / "data" / "raw" / "seyha_en_kh_all" / "data_clean.parquet"
df_se = clean(seyha_in, "seyha_en_kh_all")
df_se.to_parquet(seyha_out, index=False)
print(f"Saved to {seyha_out}")

print("\nDone.")
print(f"ParaCrawl clean: {len(df_pc):,}")
print(f"SeyhaLite clean: {len(df_se):,}")
print(f"Combined new pairs: {len(df_pc) + len(df_se):,}")
