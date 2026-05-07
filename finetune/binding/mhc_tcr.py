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
from encoders.tcr_encoder import TCREncoder
from finetune.binding.base import BindingTask

with open(ROOT / "config/train_config.json") as f:
    CFG = json.load(f)


class MHCTCRTask(BindingTask):
    name            = "mhc_tcr"
    query_dim       = CFG["s2_tcr_dim"]
    batch_size      = CFG["s2_tcr_per_batch"]
    test_split_path = CFG["s2_tcr_test_split_path"]
    eval_k_values   = [5, 10, 20]

    def load_data(self) -> tuple:
        df = pd.read_csv(CFG["s2_tcr_csv_path"]).dropna(subset=["tcrbert_input", "hla_seq"])
        df = df.drop_duplicates(subset=["tcrbert_input", "hla_seq"])

        tcr_list   = sorted(df["tcrbert_input"].unique())
        tcr_to_idx = {t: i for i, t in enumerate(tcr_list)}
        df = df.copy()
        df["tcr_idx"] = df["tcrbert_input"].map(tcr_to_idx)

        all_mhc_seqs  = sorted(df["hla_seq"].unique())
        mhc_to_local  = {mhc: i for i, mhc in enumerate(all_mhc_seqs)}
        df["mhc_local"] = df["hla_seq"].map(mhc_to_local)
        mhc_tokens      = _tokenize(all_mhc_seqs)

        n = len(tcr_list)
        rng  = np.random.default_rng(CFG["seed"])
        perm = rng.permutation(n)

        n_test = int(n * CFG["s2_test_fraction"])
        n_val  = int(n * CFG["s2_val_fraction"])
        test_set  = set(perm[:n_test].tolist())
        val_set   = set(perm[n_test : n_test + n_val].tolist())
        train_set = set(perm[n_test + n_val :].tolist())

        def build_pos(idx_set):
            sub = df[df["tcr_idx"].isin(idx_set)]
            return sub.groupby("mhc_local")["tcr_idx"].apply(set).to_dict()

        return (
            mhc_tokens, tcr_list,
            np.array(sorted(train_set)),
            np.array(sorted(val_set)),
            np.array(sorted(test_set)),
            build_pos(train_set),
            build_pos(val_set),
            build_pos(test_set),
        )

    def encode_queries(self, tcr_list: list[str]) -> torch.Tensor:
        cache = CFG["s2_tcr_embed_cache"]
        if os.path.exists(cache):
            cached = torch.load(cache, map_location="cpu")
            if cached.get("tcr_list") == tcr_list:
                print(f"Loading cached TCR embeddings from {cache} ...")
                return cached["embs"]
            print("Cache TCR list mismatch — regenerating ...")
            os.remove(cache)

        print(f"Encoding {len(tcr_list):,} TCRs with tcrBERT (one-time) ...")
        encoder = TCREncoder()
        encoder.load()
        embs = encoder.encode(tcr_list, batch_size=64)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        torch.save({"embs": embs, "tcr_list": tcr_list}, cache)
        print(f"Saved TCR embeddings to {cache}")
        return embs

    @property
    def embed_cache_path(self) -> str:
        return CFG["s2_tcr_embed_cache"]

    @property
    def query_key(self) -> str:
        return "tcr_list"
