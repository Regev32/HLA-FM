import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
from encoders.utils import _tokenize
from encoders.peptide_encoder import PeptideEncoder
from finetune.binding.base import BindingTask

with open(ROOT / "config/train_config.json") as f:
    CFG = json.load(f)


class PeptideMHCTask(BindingTask):
    name            = "peptide_mhc"
    query_dim       = CFG["s2_peptide_dim"]
    batch_size      = CFG["s2_peptides_per_batch"]
    test_split_path = CFG["s2_test_split_path"]

    def load_data(self) -> tuple:
        df = pd.read_csv(CFG["s2_csv_path"]).dropna(subset=["Epitope", "MHC_AA"])
        df = df.drop_duplicates(subset=["Epitope", "MHC_AA"])

        peptide_list = sorted(df["Epitope"].unique())
        pep_to_idx   = {p: i for i, p in enumerate(peptide_list)}
        df = df.copy()
        df["pep_idx"] = df["Epitope"].map(pep_to_idx)

        all_mhc_seqs  = sorted(df["MHC_AA"].unique())
        mhc_to_local  = {mhc: i for i, mhc in enumerate(all_mhc_seqs)}
        df["mhc_local"] = df["MHC_AA"].map(mhc_to_local)
        mhc_tokens      = _tokenize(all_mhc_seqs)

        n = len(peptide_list)
        rng  = np.random.default_rng(CFG["seed"])
        perm = rng.permutation(n)

        n_test = int(n * CFG["s2_test_fraction"])
        n_val  = int(n * CFG["s2_val_fraction"])
        test_set  = set(perm[:n_test].tolist())
        val_set   = set(perm[n_test : n_test + n_val].tolist())
        train_set = set(perm[n_test + n_val :].tolist())

        def build_pos(idx_set):
            sub = df[df["pep_idx"].isin(idx_set)]
            return sub.groupby("mhc_local")["pep_idx"].apply(set).to_dict()

        return (
            mhc_tokens, peptide_list,
            np.array(sorted(train_set)),
            np.array(sorted(val_set)),
            np.array(sorted(test_set)),
            build_pos(train_set),
            build_pos(val_set),
            build_pos(test_set),
        )

    def encode_queries(self, peptide_list: list[str]) -> torch.Tensor:
        cache = CFG["s2_pep_embed_cache"]
        if os.path.exists(cache):
            cached = torch.load(cache, map_location="cpu")
            if cached.get("peptide_list") == peptide_list:
                print(f"Loading cached peptide embeddings from {cache} ...")
                return cached["embs"]
            print("Cache peptide list mismatch — regenerating ...")
            os.remove(cache)

        print(f"Encoding {len(peptide_list):,} peptides with ESM2 (one-time) ...")
        encoder = PeptideEncoder()
        encoder.load()
        embs = encoder.encode(peptide_list, batch_size=64)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        torch.save({"embs": embs, "peptide_list": peptide_list}, cache)
        print(f"Saved peptide embeddings to {cache}")
        return embs

    @property
    def embed_cache_path(self) -> str:
        return CFG["s2_pep_embed_cache"]

    @property
    def query_key(self) -> str:
        return "peptide_list"
