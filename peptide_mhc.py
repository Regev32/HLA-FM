"""
Train MHC encoder with multi-positive InfoNCE against a frozen ESM2
peptide encoder.

Peptides are split into train / val / test before training.
The test split is saved to disk for evaluate_peptide_mhc.py.

Three models are trained:
  attn          — MAE pretrained backbone, last k layers unfrozen
  attn_random — random init backbone, fully trained
  attn_frozen    — MAE pretrained backbone frozen, only projection heads trained

Usage (from project root):
    python peptide_mhc.py
"""

import copy
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from encoders.attn_encoder import MHCAttentionEncoder
from encoders.utils import _tokenize, build_mlp
from encoders.peptide_encoder import PeptideEncoder

# ── Load config ────────────────────────────────────────────────────────────────

with open("config/train_config.json") as f:
    CFG = json.load(f)

PARAMS = {
    "output_dim": CFG["s2_output_dim"],
    "lr":         CFG["s2_lr"],
    "dropout":    CFG["s2_dropout"],
}

# ── Data ──────────────────────────────────────────────────────────────────────

def load_data() -> tuple:
    df = pd.read_csv(CFG["s2_csv_path"]).dropna(subset=["Epitope", "MHC_AA"])
    df = df.drop_duplicates(subset=["Epitope", "MHC_AA"])

    peptide_list = sorted(df["Epitope"].unique())
    pep_to_idx   = {p: i for i, p in enumerate(peptide_list)}
    df = df.copy()
    df["pep_idx"] = df["Epitope"].map(pep_to_idx)

    # All MHC alleles (shared across splits)
    all_mhc_seqs = sorted(df["MHC_AA"].unique())
    mhc_to_local = {mhc: i for i, mhc in enumerate(all_mhc_seqs)}
    df["mhc_local"] = df["MHC_AA"].map(mhc_to_local)
    mhc_tokens = _tokenize(all_mhc_seqs)

    # Split peptides into train / val / test
    n_pep = len(peptide_list)
    rng   = np.random.default_rng(CFG["seed"])
    perm  = rng.permutation(n_pep)

    n_test = int(n_pep * CFG["s2_test_fraction"])
    n_val  = int(n_pep * CFG["s2_val_fraction"])

    test_pep_set  = set(perm[:n_test].tolist())
    val_pep_set   = set(perm[n_test : n_test + n_val].tolist())
    train_pep_set = set(perm[n_test + n_val :].tolist())

    def build_pos_dict(pep_set):
        sub = df[df["pep_idx"].isin(pep_set)]
        return sub.groupby("mhc_local")["pep_idx"].apply(set).to_dict()

    train_pos = build_pos_dict(train_pep_set)
    val_pos   = build_pos_dict(val_pep_set)
    test_pos  = build_pos_dict(test_pep_set)

    train_pep_idx = np.array(sorted(train_pep_set))
    val_pep_idx   = np.array(sorted(val_pep_set))
    test_pep_idx  = np.array(sorted(test_pep_set))

    return (mhc_tokens, peptide_list,
            train_pep_idx, val_pep_idx, test_pep_idx,
            train_pos, val_pos, test_pos)


def encode_all_peptides(peptide_list: list[str]) -> torch.Tensor:
    cache = CFG["s2_pep_embed_cache"]
    if os.path.exists(cache):
        cached = torch.load(cache, map_location="cpu")
        cached_list = cached.get("peptide_list")
        if cached_list != peptide_list:
            print(f"Cache peptide list mismatch ({len(cached_list):,} vs {len(peptide_list):,}) — regenerating ...")
            os.remove(cache)
        else:
            print(f"Loading cached peptide embeddings from {cache} ...")
            return cached["embs"]

    print(f"Encoding {len(peptide_list):,} peptides with ESM2 (one-time) ...")
    encoder = PeptideEncoder()
    encoder.load()
    embs = encoder.encode(peptide_list, batch_size=64)
    torch.save({"embs": embs, "peptide_list": peptide_list}, cache)
    print(f"Saved peptide embeddings to {cache}")
    return embs

# ── Loss ──────────────────────────────────────────────────────────────────────

def multi_positive_infonce(mhc_emb, pep_emb, pos_mask):
    logits = (mhc_emb @ pep_emb.T) / CFG["s2_temp"]

    def _anchor_loss(lg, mask):
        log_denom = torch.logsumexp(lg, dim=1, keepdim=True)
        log_probs = lg - log_denom
        n_pos     = mask.float().sum(dim=1).clamp(min=1)
        loss_per  = -(log_probs * mask.float()).sum(dim=1) / n_pos
        return loss_per[mask.any(dim=1)].mean()

    return (_anchor_loss(logits, pos_mask) + _anchor_loss(logits.T, pos_mask.T)) / 2

# ── Batch helpers ─────────────────────────────────────────────────────────────

def make_pos_mask(mhc_pos_dict, batch_pep_indices, n_mhc, device):
    local_map     = {int(g): l for l, g in enumerate(batch_pep_indices)}
    batch_pep_set = set(local_map)
    mask = torch.zeros(n_mhc, len(batch_pep_indices), dtype=torch.bool)
    for mhc_idx, pos_peps in mhc_pos_dict.items():
        for pep_idx in pos_peps & batch_pep_set:
            mask[mhc_idx, local_map[pep_idx]] = True
    return mask.to(device)

# ── Train / eval loop ─────────────────────────────────────────────────────────

def run_epoch(mhc_encoder, pep_proj,
              mhc_tokens, pep_embs, mhc_pos_dict,
              pep_indices, device, optimizer=None):
    training = optimizer is not None
    mhc_encoder.train(training)
    pep_proj.train(training)

    ctx            = torch.enable_grad() if training else torch.no_grad()
    mhc_tokens_dev = mhc_tokens.to(device)
    n_mhc          = mhc_tokens.shape[0]
    batch_size     = CFG["s2_peptides_per_batch"]

    if training:
        pep_indices = pep_indices.copy()
        np.random.shuffle(pep_indices)

    total_loss, n_batches = 0.0, 0

    with ctx:
        for start in range(0, len(pep_indices), batch_size):
            batch_idx = pep_indices[start : start + batch_size]

            mhc_emb = F.normalize(mhc_encoder(mhc_tokens_dev), dim=-1)
            pep_emb = F.normalize(pep_proj(pep_embs[batch_idx].to(device)), dim=-1)

            pos_mask = make_pos_mask(mhc_pos_dict, batch_idx, n_mhc, device)
            if not pos_mask.any():
                continue

            loss = multi_positive_infonce(mhc_emb, pep_emb, pos_mask)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

    return total_loss / max(n_batches, 1)

# ── Encoder builder ───────────────────────────────────────────────────────────

def build_encoder(params: dict, device: str,
                  pretrained: bool = True,
                  unfreeze_layers: int = None) -> MHCAttentionEncoder:
    """
    Args:
        unfreeze_layers: None → all unfrozen (random init / full fine-tune).
                         0    → only proj head trainable (frozen backbone).
                         k>0  → proj + last k transformer layers unfrozen.
    """
    mae_ckpt_path = os.path.join(CFG["s1_pretrained_mae_dir"], "mae_pretrained.pt")

    if pretrained and os.path.exists(mae_ckpt_path):
        ckpt = torch.load(mae_ckpt_path, map_location=device)
        arch = ckpt["arch"]
        encoder = MHCAttentionEncoder(
            vocab_size  = 23,
            embed_dim   = arch["embed_dim"],
            nhead       = arch["nhead"],
            num_layers  = arch["num_layers"],
            dropout     = params.get("dropout", arch["dropout"]),
            output_dim  = params["output_dim"],
            proj_layers = CFG["s2_proj_layers"],
        ).to(device)
        backbone_state = {k: v for k, v in ckpt["mhc_encoder"].items()
                          if not k.startswith("proj.")}
        encoder.load_state_dict(backbone_state, strict=False)
        print(f"  Loaded MAE pretrained backbone")
    else:
        if pretrained:
            print(f"  No MAE checkpoint found at {mae_ckpt_path} — falling back to random init")
        else:
            print(f"  Random init backbone")
        encoder = MHCAttentionEncoder(
            vocab_size  = 23,
            embed_dim   = CFG["s1_embed_dim"],
            nhead       = CFG["s1_nhead"],
            num_layers  = CFG["s1_num_layers"],
            dropout     = params.get("dropout", 0.1),
            output_dim  = params["output_dim"],
            proj_layers = CFG["s2_proj_layers"],
        ).to(device)

    if unfreeze_layers is not None:
        # Freeze everything first
        for p in encoder.parameters():
            p.requires_grad_(False)
        # Always unfreeze the projection head
        for p in encoder.proj.parameters():
            p.requires_grad_(True)
        # Unfreeze last k transformer layers
        if unfreeze_layers > 0:
            total = len(encoder.transformer.layers)
            for layer in encoder.transformer.layers[max(0, total - unfreeze_layers):]:
                for p in layer.parameters():
                    p.requires_grad_(True)
        n_train = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in encoder.parameters())
        print(f"  Unfroze proj + last {unfreeze_layers} transformer layers "
              f"({n_train:,}/{n_total:,} params trainable)")

    return encoder

# ── Training run ─────────────────────────────────────────────────────────────

def retrain_best(params, pretrained, unfreeze_layers,
                 mhc_tokens, pep_embs, train_pep_idx, val_pep_idx,
                 train_pos, val_pos, device, results_dir):
    mhc_encoder = build_encoder(params, device, pretrained=pretrained,
                                 unfreeze_layers=unfreeze_layers)
    pep_proj = build_mlp(CFG["s2_peptide_dim"], params["output_dim"],
                         CFG["s2_proj_layers"], CFG["s2_dropout"]).to(device)
    optimizer   = torch.optim.Adam(
        [p for p in mhc_encoder.parameters() if p.requires_grad]
        + list(pep_proj.parameters()),
        lr=params["lr"],
    )

    best_val     = float("inf")
    patience     = 0
    decays_left  = CFG["s2_lr_decays"]
    best_weights = {}
    train_losses, val_losses = [], []

    for epoch in range(CFG["s2_n_epochs_final"]):
        train_loss = run_epoch(mhc_encoder, pep_proj,
                               mhc_tokens, pep_embs, train_pos,
                               train_pep_idx, device, optimizer=optimizer)
        val_loss   = run_epoch(mhc_encoder, pep_proj,
                               mhc_tokens, pep_embs, val_pos,
                               val_pep_idx, device, optimizer=None)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        lr = optimizer.param_groups[0]["lr"]
        print(f"  epoch {epoch+1:2d}/{CFG['s2_n_epochs_final']}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  lr={lr:.1e}")

        if val_loss < best_val:
            best_val, patience = val_loss, 0
            best_weights = {
                "mhc_encoder": copy.deepcopy(mhc_encoder.state_dict()),
                "pep_proj":    copy.deepcopy(pep_proj.state_dict()),
                "val_loss":    best_val,
                "output_dim":  params["output_dim"],
                "proj_layers": CFG["s2_proj_layers"],
            }
        else:
            patience += 1
            if patience >= CFG["s2_early_stop_patience"]:
                if decays_left > 0:
                    new_lr = lr * CFG["s2_lr_factor"]
                    for pg in optimizer.param_groups:
                        pg["lr"] = new_lr
                    decays_left -= 1
                    patience = 0
                    print(f"  plateau — decayed lr to {new_lr:.1e} "
                          f"({decays_left} decays remaining)")
                else:
                    print(f"  early stop at epoch {epoch+1}")
                    break

    torch.save(best_weights, os.path.join(results_dir, "best_model.pt"))
    print(f"Saved model to {results_dir}/best_model.pt  (val={best_val:.4f})")

    return train_losses, val_losses

# ── Per-model training ────────────────────────────────────────────────────────

def train_model(results_name, pretrained, unfreeze_layers,
                mhc_tokens, pep_embs, train_pep_idx, val_pep_idx,
                train_pos, val_pos, device):
    results_dir = os.path.join(CFG["results_dir"], results_name)
    os.makedirs(results_dir, exist_ok=True)

    train_losses, val_losses = retrain_best(
        PARAMS, pretrained, unfreeze_layers,
        mhc_tokens, pep_embs, train_pep_idx, val_pep_idx,
        train_pos, val_pos, device, results_dir,
    )

    _save_loss_plot(train_losses, val_losses, results_name, results_dir)

    return train_losses, val_losses

# ── Plotting ──────────────────────────────────────────────────────────────────

def _save_loss_plot(train_losses, val_losses, title, results_dir):
    epochs = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots()
    ax.plot(epochs, train_losses, label="train")
    ax.plot(epochs, val_losses,   label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    fig.savefig(os.path.join(results_dir, "loss_curve.png"), dpi=150)
    plt.close(fig)


def _save_combined_loss_plot(all_losses, out_path):
    order = ["attn", "attn_random", "attn_frozen"]
    present = [name for name in order if name in all_losses]
    n = len(present)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, present):
        train_l, val_l = all_losses[name]
        epochs = range(1, len(train_l) + 1)
        ax.plot(epochs, train_l, label="train")
        ax.plot(epochs, val_l,   label="val")
        ax.set_xlabel("Epoch")
        ax.set_title(name)
        ax.legend()

    axes[0].set_ylabel("Loss")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined loss plot to {out_path}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    np.random.seed(CFG["seed"])
    torch.manual_seed(CFG["seed"])
    torch.cuda.manual_seed_all(CFG["seed"])

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    mhc_tokens, peptide_list, train_pep_idx, val_pep_idx, test_pep_idx, \
        train_pos, val_pos, test_pos = load_data()

    print(f"MHCs: {mhc_tokens.shape[0]}  "
          f"Peptides: {len(peptide_list):,}  "
          f"(train={len(train_pep_idx)}, val={len(val_pep_idx)}, test={len(test_pep_idx)})")

    pep_embs = encode_all_peptides(peptide_list)

    # Save test data — remap peptide indices to local (0..N_test-1)
    global_to_local = {int(g): l for l, g in enumerate(test_pep_idx)}
    test_pos_local = {}
    for mhc_i, pep_set in test_pos.items():
        test_pos_local[mhc_i] = {global_to_local[p] for p in pep_set}

    torch.save({"mhc_tokens": mhc_tokens,
                "test_pep_idx": test_pep_idx.tolist(),
                "pos_dict": test_pos_local,
                "peptide_list": peptide_list},
               CFG["s2_test_split_path"])
    print(f"Saved test split to {CFG['s2_test_split_path']}")

    shared = dict(mhc_tokens=mhc_tokens, pep_embs=pep_embs,
                  train_pep_idx=train_pep_idx, val_pep_idx=val_pep_idx,
                  train_pos=train_pos, val_pos=val_pos, device=device)

    all_losses = {}

    # Model 1: MAE pretrained backbone, last k layers unfrozen
    print(f"\n{'='*55}\n  Model: attn  (pretrained)\n{'='*55}")
    all_losses["attn"] = train_model("attn", pretrained=True,
                unfreeze_layers=CFG["s2_unfreeze_layers"], **shared)

    # Model 2: Random init backbone, fully trained (ablation)
    print(f"\n{'='*55}\n  Model: attn_random  (random init)\n{'='*55}")
    all_losses["attn_random"] = train_model("attn_random", pretrained=False,
                unfreeze_layers=None, **shared)

    # Model 3: Frozen MAE backbone, only projection heads trained (probe)
    print(f"\n{'='*55}\n  Model: attn_frozen  (frozen backbone)\n{'='*55}")
    all_losses["attn_frozen"] = train_model("attn_frozen", pretrained=True,
                unfreeze_layers=0, **shared)

    _save_combined_loss_plot(all_losses,
                             os.path.join(CFG["results_dir"], "models_loss.png"))


if __name__ == "__main__":
    main()
