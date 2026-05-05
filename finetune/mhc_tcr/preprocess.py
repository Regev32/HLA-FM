"""
Preprocess the mhc_tcr_bond dataset into tcrBERT-ready format.

Reads:  data/finetune/mhc_tcr/mhc_tcr_bond.csv
        data/pretrain/allele_to_aa.json
Writes: data/finetune/mhc_tcr/mhc_tcr_bond_tcrbert.csv

Usage (from project root):
    python -m finetune.mhc_tcr.preprocess
"""

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent.parent

SRC_CSV     = ROOT / "data/finetune/mhc_tcr/mhc_tcr_bond.csv"
ALLELE_JSON = ROOT / "data/pretrain/allele_to_aa.json"
OUT_CSV     = ROOT / "data/finetune/mhc_tcr/mhc_tcr_bond_tcrbert.csv"


def clean_allele(s: str) -> str:
    s = re.sub(r"\s+HLA", "", str(s)).strip()
    s = re.sub(r"\*\s+", "*", s)
    return s


def run():
    df = pd.read_csv(SRC_CSV, usecols=["cdr3b", "vb", "jb", "hla_seq"])
    print(f"Loaded: {len(df):,} rows")

    df = df.dropna(subset=["cdr3b", "vb", "jb", "hla_seq"])
    print(f"After dropping incomplete rows: {len(df):,}")

    with open(ALLELE_JSON) as f:
        allele_to_aa = json.load(f)

    df["hla_seq"] = df["hla_seq"].apply(clean_allele)
    df = df[df["hla_seq"].isin(allele_to_aa)]
    print(f"After dropping unmappable HLA alleles: {len(df):,}")

    df["hla_seq"] = df["hla_seq"].map(allele_to_aa)

    df["tcrbert_input"] = (
        df["vb"] + " "
        + df["cdr3b"].apply(lambda s: " ".join(s))
        + " " + df["jb"]
    )

    out = df[["tcrbert_input", "hla_seq"]]
    out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved → {OUT_CSV}")
    print("\nSample rows:")
    print(out.head(3).to_string())


if __name__ == "__main__":
    run()
