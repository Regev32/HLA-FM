"""
Evaluate the pretrained MAE encoder on masked token prediction.

Loads the pretrained model and the saved train/val/test split, then reports
the percentage of incorrectly predicted masked tokens on the test set.

Usage (from project root):
    python evaluate_pretrain.py
"""

import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from encoders.attn_encoder import MHCAttentionEncoder
from encoders.utils import PAD_IDX, _tokenize

with open("config/train_config.json") as f:
    CFG = json.load(f)

MASK_IDX = 22
NUM_AAS  = 21


def load_sequences() -> torch.Tensor:
    with open(CFG["s1_allele_aa_path"]) as f:
        aa_map = json.load(f)
    seqs = list(dict.fromkeys(aa_map.values()))
    return _tokenize(seqs)


def mask_batch(tokens: torch.Tensor, ratio: float):
    is_padding = tokens == PAD_IDX
    rand       = torch.rand_like(tokens, dtype=torch.float)
    mask       = (rand < ratio) & ~is_padding
    masked     = tokens.clone()
    masked[mask] = MASK_IDX
    return masked, mask, tokens


def main():
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    ckpt_path = os.path.join(CFG["s1_pretrained_mae_dir"], "mae_pretrained.pt")
    ckpt      = torch.load(ckpt_path, map_location=device)
    arch      = ckpt["arch"]

    encoder = MHCAttentionEncoder(
        vocab_size = 23,
        embed_dim  = arch["embed_dim"],
        nhead      = arch["nhead"],
        num_layers = arch["num_layers"],
        output_dim = arch["output_dim"],
        dropout    = arch["dropout"],
    ).to(device)
    encoder.load_state_dict(ckpt["mhc_encoder"])
    encoder.eval()

    decoder = nn.Linear(arch["embed_dim"], NUM_AAS).to(device)
    decoder.load_state_dict(ckpt["mae_decoder"])
    decoder.eval()

    print(f"Loaded pretrained encoder from {ckpt_path}  (val_loss={ckpt['val_loss']:.4f})")

    # ── Load split ────────────────────────────────────────────────────────────
    split     = torch.load(CFG["s1_split_path"], weights_only=True)
    test_idx  = split["test"]
    print(f"Test set: {len(test_idx):,} sequences")

    tokens     = load_sequences()
    test_tokens = tokens[test_idx]

    # ── Evaluate ──────────────────────────────────────────────────────────────
    torch.manual_seed(CFG["seed"])

    batch_size = CFG["s1_batch_size"]
    mask_ratio = CFG["s1_mask_ratio"]

    total_masked  = 0
    total_errors  = 0

    with torch.no_grad():
        for start in range(0, len(test_tokens), batch_size):
            batch  = test_tokens[start : start + batch_size].to(device)
            masked, mask, targets = mask_batch(batch, mask_ratio)

            if not mask.any():
                continue

            token_embs = encoder(masked, return_sequence=True)  # (B, L, D)
            logits     = decoder(token_embs)                    # (B, L, 21)
            preds      = logits[mask].argmax(dim=-1)
            gt         = targets[mask]

            total_masked += mask.sum().item()
            total_errors += (preds != gt).sum().item()

    error_pct = 100.0 * total_errors / max(total_masked, 1)
    print(f"\nMasked tokens : {total_masked:,}")
    print(f"Errors        : {total_errors:,}")
    print(f"Error rate    : {error_pct:.2f}%")
    print(f"Accuracy      : {100 - error_pct:.2f}%")


if __name__ == "__main__":
    main()
