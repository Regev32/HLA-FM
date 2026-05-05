"""
MAE pretraining for the MHC transformer encoder.

When compare_architectures=true in config, trains both the classic and modern
encoder side-by-side and saves a loss comparison plot.

Usage (from project root):
    python main.py pretrain
"""

import copy
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from encoders.attn_encoder import MHCAttentionEncoder
from encoders.modern_attn_encoder import ModernMHCEncoder, find_matching_hidden_dim
from encoders.utils import PAD_IDX, _tokenize

# ── Config ────────────────────────────────────────────────────────────────────

with open(ROOT / "config/train_config.json") as f:
    CFG = json.load(f)

MASK_IDX = 22
NUM_AAS  = 21

# ── Data ──────────────────────────────────────────────────────────────────────

def load_sequences() -> torch.Tensor:
    with open(CFG["s1_allele_aa_path"]) as f:
        aa_map = json.load(f)
    seqs  = list(aa_map.values())
    n_raw = len(seqs)
    seqs  = list(dict.fromkeys(seqs))
    print(f"Loaded {n_raw:,} allele sequences → {len(seqs):,} unique")
    tokens = _tokenize(seqs)
    print(f"Tokenized: shape={tokens.shape}")
    return tokens

# ── Masking ───────────────────────────────────────────────────────────────────

def mask_batch(tokens: torch.Tensor, ratio: float):
    is_padding = tokens == PAD_IDX
    rand       = torch.rand_like(tokens, dtype=torch.float)
    mask       = (rand < ratio) & ~is_padding
    masked     = tokens.clone()
    masked[mask] = MASK_IDX
    return masked, mask, tokens

# ── Per-architecture training ─────────────────────────────────────────────────

def _train(encoder: nn.Module, tokens: torch.Tensor,
           train_idx, val_idx, output_dir: str, device: str) -> tuple[list, list]:
    """Train one encoder variant; returns (train_losses, val_losses)."""
    os.makedirs(output_dir, exist_ok=True)

    decoder = nn.Linear(encoder.embed_dim, NUM_AAS).to(device)
    encoder = encoder.to(device)

    n_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"  Encoder parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=CFG["s1_lr"], weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=CFG["s1_lr_patience"],
        factor=CFG["s1_lr_factor"], min_lr=CFG["s1_min_lr"],
    )

    train_tokens = tokens[train_idx]
    val_tokens   = tokens[val_idx]
    batch_size   = CFG["s1_batch_size"]
    mask_ratio   = CFG["s1_mask_ratio"]
    n_epochs     = CFG["s1_epochs"]
    patience_max = CFG["s1_early_stop_patience"]

    best_val, patience = float("inf"), 0
    best_weights       = {}
    train_losses, val_losses = [], []

    for epoch in range(n_epochs):
        # ── train ──────────────────────────────────────────────────────────────
        encoder.train(); decoder.train()
        idx = torch.randperm(len(train_tokens))
        t_loss, n_batches = 0.0, 0

        for start in range(0, len(train_tokens), batch_size):
            batch  = train_tokens[idx[start : start + batch_size]].to(device)
            masked, mask, targets = mask_batch(batch, mask_ratio)
            if not mask.any():
                continue

            token_embs = encoder(masked, return_sequence=True)
            logits     = decoder(token_embs)
            loss       = F.cross_entropy(logits[mask], targets[mask])

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(decoder.parameters()), 1.0
            )
            optimizer.step()
            t_loss += loss.item(); n_batches += 1

        train_loss = t_loss / max(n_batches, 1)

        # ── val ────────────────────────────────────────────────────────────────
        encoder.eval(); decoder.eval()
        v_loss, v_batches = 0.0, 0

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
        print(f"  epoch {epoch+1:3d}/{n_epochs}  train={train_loss:.4f}  val={val_loss:.4f}  lr={current_lr:.2e}")

        if val_loss < best_val:
            best_val, patience = val_loss, 0
            arch = {k: CFG[k] for k in
                    ["s1_embed_dim","s1_nhead","s1_num_layers","s1_output_dim","s1_dropout"]}
            arch = {k.replace("s1_",""):v for k,v in arch.items()}
            best_weights = {
                "mhc_encoder": copy.deepcopy(encoder.state_dict()),
                "mae_decoder": copy.deepcopy(decoder.state_dict()),
                "arch":        arch,
                "val_loss":    best_val,
            }
        else:
            patience += 1
            if patience >= patience_max and current_lr <= CFG["s1_min_lr"]:
                print(f"  Early stop at epoch {epoch+1}")
                break

    torch.save(best_weights, os.path.join(output_dir, "mae_pretrained.pt"))
    print(f"  Saved → {output_dir}/mae_pretrained.pt  (val={best_val:.4f})")
    return train_losses, val_losses


def _save_loss_plot(losses_dict: dict, output_dir: str, filename: str):
    """losses_dict: {label: (train_losses, val_losses)}"""
    fig, axes = plt.subplots(1, len(losses_dict), figsize=(6 * len(losses_dict), 5), sharey=True)
    if len(losses_dict) == 1:
        axes = [axes]
    for ax, (label, (train_l, val_l)) in zip(axes, losses_dict.items()):
        epochs = range(1, len(train_l) + 1)
        ax.plot(epochs, train_l, label="train")
        ax.plot(epochs, val_l,   label="val")
        clipped = train_l[9:] + val_l[9:]
        if clipped:
            ax.set_ylim(min(clipped), max(clipped))
        ax.set_xlim(left=10)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (CE)")
        ax.set_title(label); ax.legend()
    fig.tight_layout()
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved loss plot → {path}")

# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    np.random.seed(CFG["seed"])
    torch.manual_seed(CFG["seed"])
    torch.cuda.manual_seed_all(CFG["seed"])

    device = ("cuda" if torch.cuda.is_available()
               else "mps" if torch.backends.mps.is_available()
               else "cpu")
    print(f"Device: {device}")

    tokens = load_sequences()
    n      = len(tokens)

    rng       = np.random.default_rng(CFG["seed"])
    perm      = rng.permutation(n)
    test_size = max(1, int(n * CFG["s1_test_fraction"]))
    val_size  = max(1, int(n * CFG["s1_val_fraction"]))
    test_idx  = perm[:test_size]
    val_idx   = perm[test_size : test_size + val_size]
    train_idx = perm[test_size + val_size:]
    print(f"Train: {len(train_idx):,}  Val: {len(val_idx):,}  Test: {len(test_idx):,}")

    os.makedirs(os.path.dirname(CFG["s1_split_path"]), exist_ok=True)
    torch.save({"train": train_idx.tolist(), "val": val_idx.tolist(),
                "test":  test_idx.tolist()}, CFG["s1_split_path"])

    arch_kwargs = dict(
        vocab_size  = 23,
        embed_dim   = CFG["s1_embed_dim"],
        nhead       = CFG["s1_nhead"],
        num_layers  = CFG["s1_num_layers"],
        output_dim  = CFG["s1_output_dim"],
        dropout     = CFG["s1_dropout"],
    )
    all_losses = {}

    # ── Classic ──────────────────────────────────────────────────────────────
    print(f"\n{'='*55}\n  Classic encoder\n{'='*55}")
    classic_enc = MHCAttentionEncoder(**arch_kwargs)
    all_losses["classic"] = _train(
        classic_enc, tokens, train_idx, val_idx,
        CFG["s1_pretrained_mae_dir"], device,
    )

    # ── Modern ───────────────────────────────────────────────────────────────
    if CFG.get("compare_architectures", False):
        print(f"\n{'='*55}\n  Modern encoder  "
              f"(MoE={'on' if CFG['s1_modern_use_moe'] else 'off'})\n{'='*55}")
        classic_n_params = sum(p.numel() for p in classic_enc.parameters())
        modern_base_kwargs = dict(
            **arch_kwargs,
            use_moe     = CFG["s1_modern_use_moe"],
            num_experts = CFG["s1_modern_num_experts"],
            top_k       = CFG["s1_modern_top_k"],
        )
        ffn_h = find_matching_hidden_dim(classic_n_params, modern_base_kwargs)
        modern_enc = ModernMHCEncoder(**modern_base_kwargs, ffn_hidden_dim=ffn_h)
        all_losses["modern"] = _train(
            modern_enc, tokens, train_idx, val_idx,
            CFG["s1_modern_mae_dir"], device,
        )

    # ── Comparison plot ───────────────────────────────────────────────────────
    _save_loss_plot(all_losses, CFG["pretrain_results_dir"], "pretrain_comparison.png")


if __name__ == "__main__":
    run()
