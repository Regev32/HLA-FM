"""
MAE pretraining for the MHC transformer encoder.

Trains on all ~25k allele sequences from allele_to_aa.json using masked
amino acid prediction. Run this FIRST, before train_mhc.py.

Usage (from project root):
    python pretrain_mae.py
"""

import copy
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from encoders.attn_encoder import MHCAttentionEncoder
from encoders.utils import PAD_IDX, _tokenize

# ── Config ────────────────────────────────────────────────────────────────────

with open("config/train_config.json") as f:
    CFG = json.load(f)

MASK_IDX = 22   # appended to the 22-token vocab (indices 0–21)
NUM_AAS  = 21   # prediction targets: indices 0–20 (20 standard AAs + X)

# ── Data ──────────────────────────────────────────────────────────────────────

def load_sequences() -> torch.Tensor:
    with open(CFG["s1_allele_aa_path"]) as f:
        aa_map = json.load(f)
    seqs  = list(aa_map.values())
    n_raw = len(seqs)
    seqs  = list(dict.fromkeys(seqs))   # deduplicate, preserve order
    print(f"Loaded {n_raw:,} allele sequences → {len(seqs):,} unique from {CFG['s1_allele_aa_path']}")
    tokens = _tokenize(seqs)
    print(f"Tokenized: shape={tokens.shape}  (max_len={tokens.shape[1]})")
    return tokens

# ── Masking ───────────────────────────────────────────────────────────────────

def mask_batch(tokens: torch.Tensor, ratio: float):
    """Randomly mask non-padding positions at a fixed ratio.

    Returns:
        masked   – (B, L) tokens with some positions replaced by MASK_IDX
        mask     – (B, L) bool, True at masked positions
        tokens   – original (B, L) tokens (prediction targets)
    """
    is_padding = tokens == PAD_IDX
    rand       = torch.rand_like(tokens, dtype=torch.float)
    mask       = (rand < ratio) & ~is_padding
    masked     = tokens.clone()
    masked[mask] = MASK_IDX
    return masked, mask, tokens

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    np.random.seed(CFG["seed"])
    torch.manual_seed(CFG["seed"])
    torch.cuda.manual_seed_all(CFG["seed"])

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps"  if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    tokens = load_sequences()
    n      = len(tokens)

    rng      = np.random.default_rng(CFG["seed"])
    perm     = rng.permutation(n)
    val_size = max(1, int(n * CFG["s1_val_fraction"]))
    val_idx  = perm[:val_size]
    train_idx = perm[val_size:]
    print(f"Train: {len(train_idx):,}  Val: {len(val_idx):,}")

    arch       = {"embed_dim": CFG["s1_embed_dim"], "nhead": CFG["s1_nhead"], "num_layers": CFG["s1_num_layers"], "output_dim": CFG["s1_output_dim"], "dropout": CFG["s1_dropout"]}
    output_dir = CFG["s1_pretrained_mae_dir"]
    os.makedirs(output_dir, exist_ok=True)

    encoder = MHCAttentionEncoder(
        vocab_size = 23,           # 0–21 normal + 22 MASK
        embed_dim  = arch["embed_dim"],
        nhead      = arch["nhead"],
        num_layers = arch["num_layers"],
        output_dim = arch["output_dim"],
        dropout    = arch["dropout"],
    ).to(device)

    # Per-token decoder: embed_dim → 21 AA classes
    decoder = nn.Linear(encoder.embed_dim, NUM_AAS).to(device)

    n_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"Encoder parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr           = CFG["s1_lr"],
        weight_decay = 0.01,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode     = "min",
        patience = CFG["s1_lr_patience"],
        factor   = CFG["s1_lr_factor"],
        min_lr   = CFG["s1_min_lr"],
    )

    train_tokens = tokens[train_idx]
    val_tokens   = tokens[val_idx]

    batch_size   = CFG["s1_batch_size"]
    mask_ratio   = CFG["s1_mask_ratio"]
    n_epochs     = CFG["s1_epochs"]
    patience_max = CFG["s1_early_stop_patience"]

    best_val     = float("inf")
    patience     = 0
    best_weights = {}
    train_losses, val_losses = [], []

    for epoch in range(n_epochs):
        # ── train ──────────────────────────────────────────────────────────────
        encoder.train()
        decoder.train()

        idx       = torch.randperm(len(train_tokens))
        t_loss    = 0.0
        n_batches = 0

        for start in range(0, len(train_tokens), batch_size):
            batch  = train_tokens[idx[start : start + batch_size]].to(device)
            masked, mask, targets = mask_batch(batch, mask_ratio)

            if not mask.any():
                continue

            token_embs = encoder(masked, return_sequence=True)  # (B, seq_len, D)
            logits     = decoder(token_embs)                    # (B, seq_len, 21)
            loss       = F.cross_entropy(logits[mask], targets[mask])

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(decoder.parameters()), 1.0
            )
            optimizer.step()

            t_loss    += loss.item()
            n_batches += 1

        train_loss = t_loss / max(n_batches, 1)

        # ── val ────────────────────────────────────────────────────────────────
        encoder.eval()
        decoder.eval()
        v_loss    = 0.0
        v_batches = 0

        with torch.no_grad():
            for start in range(0, len(val_tokens), batch_size):
                batch  = val_tokens[start : start + batch_size].to(device)
                masked, mask, targets = mask_batch(batch, mask_ratio)

                if not mask.any():
                    continue

                token_embs = encoder(masked, return_sequence=True)
                logits     = decoder(token_embs)
                v_loss    += F.cross_entropy(logits[mask], targets[mask]).item()
                v_batches += 1

        val_loss = v_loss / max(v_batches, 1)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"epoch {epoch+1:3d}/{n_epochs}  train={train_loss:.4f}  val={val_loss:.4f}  lr={current_lr:.2e}")

        if val_loss < best_val:
            best_val, patience = val_loss, 0
            best_weights = {
                "mhc_encoder": copy.deepcopy(encoder.state_dict()),
                "arch":        arch,
                "val_loss":    best_val,
            }
        else:
            patience += 1
            if patience >= patience_max and current_lr <= CFG["s1_min_lr"]:
                print(f"Early stop at epoch {epoch+1} — LR at floor ({current_lr:.2e}), no improvement for {patience} epochs")
                break

    torch.save(best_weights, os.path.join(output_dir, "mae_pretrained.pt"))
    print(f"\nSaved pretrained encoder to {output_dir}/mae_pretrained.pt  (val={best_val:.4f})")

    fig, ax = plt.subplots()
    ax.plot(range(1, len(train_losses) + 1), train_losses, label="train")
    ax.plot(range(1, len(val_losses)   + 1), val_losses,   label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (CE)")
    ax.set_title("MAE pretraining")
    ax.legend()
    fig.savefig(os.path.join(output_dir, "pretrain_loss.png"), dpi=150)
    plt.close(fig)
    print(f"Saved loss plot to {output_dir}/pretrain_loss.png")


if __name__ == "__main__":
    main()
