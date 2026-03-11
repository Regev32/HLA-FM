"""
Train MHC encoder with multi-positive InfoNCE against a frozen ESM2
peptide encoder.

MHC alleles are split into train / val / test before training.
The test split is saved to disk for evaluate.py.

Three models are trained:
  attn          — MAE pretrained backbone, fine-tuned contrastively
  attn_baseline — random init backbone, fine-tuned contrastively
  attn_probe    — MAE pretrained backbone frozen, only projection heads trained

Usage (from project root):
    python train_mhc.py
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
from encoders.utils import _tokenize
from encoders.peptide_encoder import PeptideEncoder

# ── Load config ────────────────────────────────────────────────────────────────

with open("config/train_config.json") as f:
    CFG = json.load(f)

PARAMS = {
    "output_dim": CFG["s2_output_dim"],
    "lr":         CFG["s2_lr"],
    "dropout":    CFG["s2_dropout"],
    "init_temp":  CFG["s2_init_temp"],
}

# ── Data ──────────────────────────────────────────────────────────────────────

def load_data() -> tuple:
    df = pd.read_csv(CFG["s2_csv_path"]).dropna(subset=["Epitope", "MHC_AA"])
    df = df.drop_duplicates(subset=["Epitope", "MHC_AA"])

    peptide_list = sorted(df["Epitope"].unique())
    pep_to_idx   = {p: i for i, p in enumerate(peptide_list)}
    df = df.copy()
    df["pep_idx"] = df["Epitope"].map(pep_to_idx)

    # Split MHC alleles into train / val / test
    all_mhcs = np.array(sorted(df["MHC_AA"].unique()))
    rng      = np.random.default_rng(CFG["seed"])
    perm     = rng.permutation(len(all_mhcs))

    n_test = int(len(all_mhcs) * CFG["s2_test_fraction"])
    n_val  = int(len(all_mhcs) * CFG["s2_val_fraction"])

    test_mhcs  = set(all_mhcs[perm[:n_test]])
    val_mhcs   = set(all_mhcs[perm[n_test : n_test + n_val]])
    train_mhcs = set(all_mhcs[perm[n_test + n_val :]])

    def build_split(mhc_set):
        mhc_list     = sorted(mhc_set)
        mhc_to_local = {mhc: i for i, mhc in enumerate(mhc_list)}
        sub          = df[df["MHC_AA"].isin(mhc_set)].copy()
        sub["local"] = sub["MHC_AA"].map(mhc_to_local)
        pos_dict     = sub.groupby("local")["pep_idx"].apply(set).to_dict()
        tokens       = _tokenize(mhc_list)
        return tokens, pos_dict

    train_tokens, train_pos = build_split(train_mhcs)
    val_tokens,   val_pos   = build_split(val_mhcs)
    test_tokens,  test_pos  = build_split(test_mhcs)

    return train_tokens, val_tokens, test_tokens, peptide_list, train_pos, val_pos, test_pos


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

def multi_positive_infonce(mhc_emb, pep_emb, pos_mask, log_temp):
    temp   = log_temp.exp().clamp(min=1e-4)
    logits = (mhc_emb @ pep_emb.T) / temp

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

def run_epoch(mhc_encoder, pep_proj, log_temp,
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

            loss = multi_positive_infonce(mhc_emb, pep_emb, pos_mask, log_temp)

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
                  freeze_backbone: bool = False) -> MHCAttentionEncoder:
    mae_ckpt_path = os.path.join(CFG["s1_pretrained_mae_dir"], "mae_pretrained.pt")

    if pretrained and os.path.exists(mae_ckpt_path):
        ckpt = torch.load(mae_ckpt_path, map_location=device)
        arch = ckpt["arch"]
        encoder = MHCAttentionEncoder(
            vocab_size = 23,
            embed_dim  = arch["embed_dim"],
            nhead      = arch["nhead"],
            num_layers = arch["num_layers"],
            dropout    = params.get("dropout", arch["dropout"]),
            output_dim = params["output_dim"],
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
            vocab_size = 23,
            embed_dim  = CFG["s1_embed_dim"],
            nhead      = CFG["s1_nhead"],
            num_layers = CFG["s1_num_layers"],
            dropout    = params.get("dropout", 0.1),
            output_dim = params["output_dim"],
        ).to(device)

    if freeze_backbone:
        for name, p in encoder.named_parameters():
            if not name.startswith("proj."):
                p.requires_grad_(False)
        print(f"  Backbone frozen — training projection head only")

    return encoder

# ── Training run ─────────────────────────────────────────────────────────────

def retrain_best(params, pretrained, freeze_backbone,
                 train_mhc_tokens, val_mhc_tokens, pep_embs,
                 train_pos, val_pos, pep_idx, device, results_dir):
    print(f"\nFinal training  pretrained={pretrained}  freeze_backbone={freeze_backbone}  params={params}")

    mhc_encoder = build_encoder(params, device, pretrained=pretrained,
                                 freeze_backbone=freeze_backbone)
    pep_proj    = nn.Linear(CFG["s2_peptide_dim"], params["output_dim"]).to(device)
    log_temp    = nn.Parameter(torch.tensor(float(np.log(params["init_temp"]))).to(device))
    optimizer   = torch.optim.Adam(
        [p for p in mhc_encoder.parameters() if p.requires_grad]
        + list(pep_proj.parameters()) + [log_temp],
        lr=params["lr"],
    )

    best_val     = float("inf")
    patience     = 0
    best_weights = {}
    train_losses, val_losses = [], []

    for epoch in range(CFG["s2_n_epochs_final"]):
        train_loss = run_epoch(mhc_encoder, pep_proj, log_temp,
                               train_mhc_tokens, pep_embs, train_pos,
                               pep_idx, device, optimizer=optimizer)
        val_loss   = run_epoch(mhc_encoder, pep_proj, log_temp,
                               val_mhc_tokens, pep_embs, val_pos,
                               pep_idx, device, optimizer=None)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"  epoch {epoch+1:2d}/{CFG['s2_n_epochs_final']}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  "
              f"temp={log_temp.exp().item():.4f}")

        if val_loss < best_val:
            best_val, patience = val_loss, 0
            best_weights = {
                "mhc_encoder": copy.deepcopy(mhc_encoder.state_dict()),
                "pep_proj":    copy.deepcopy(pep_proj.state_dict()),
                "log_temp":    log_temp.detach().clone(),
                "val_loss":    best_val,
            }
        else:
            patience += 1
            if patience >= CFG["s2_early_stop_patience"]:
                print(f"  early stop at epoch {epoch+1}")
                break

    torch.save(best_weights, os.path.join(results_dir, "best_model.pt"))
    print(f"Saved model to {results_dir}/best_model.pt  (val={best_val:.4f})")

    return train_losses, val_losses

# ── Per-model training ────────────────────────────────────────────────────────

def train_model(results_name, pretrained, freeze_backbone,
                train_mhc_tokens, val_mhc_tokens, pep_embs,
                train_pos, val_pos, pep_idx, device):
    results_dir = os.path.join(CFG["results_dir"], results_name)
    os.makedirs(results_dir, exist_ok=True)

    train_losses, val_losses = retrain_best(
        PARAMS, pretrained, freeze_backbone,
        train_mhc_tokens, val_mhc_tokens, pep_embs,
        train_pos, val_pos, pep_idx, device, results_dir,
    )

    _save_loss_plot(train_losses, val_losses, results_name, results_dir)

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

    train_mhc_tokens, val_mhc_tokens, test_mhc_tokens, \
        peptide_list, train_pos, val_pos, test_pos = load_data()

    print(f"Train MHCs: {train_mhc_tokens.shape[0]}  "
          f"Val MHCs: {val_mhc_tokens.shape[0]}  "
          f"Test MHCs: {test_mhc_tokens.shape[0]}  "
          f"Peptides: {len(peptide_list):,}")

    pep_embs = encode_all_peptides(peptide_list)

    torch.save({"mhc_tokens": test_mhc_tokens, "pos_dict": test_pos,
                "peptide_list": peptide_list},
               CFG["s2_test_split_path"])
    print(f"Saved test split to {CFG['s2_test_split_path']}")

    pep_idx = np.arange(len(peptide_list))

    shared = dict(train_mhc_tokens=train_mhc_tokens, val_mhc_tokens=val_mhc_tokens,
                  pep_embs=pep_embs, train_pos=train_pos, val_pos=val_pos,
                  pep_idx=pep_idx, device=device)

    # Model 1: MAE pretrained backbone + contrastive fine-tuning
    print(f"\n{'='*55}\n  Model: attn  (pretrained)\n{'='*55}")
    train_model("attn", pretrained=True, freeze_backbone=False, **shared)

    # Model 2: Random init backbone + contrastive fine-tuning (ablation)
    print(f"\n{'='*55}\n  Model: attn_baseline  (random init)\n{'='*55}")
    train_model("attn_baseline", pretrained=False, freeze_backbone=False, **shared)

    # Model 3: Frozen MAE backbone, only projection heads trained (linear probe)
    print(f"\n{'='*55}\n  Model: attn_probe  (frozen backbone)\n{'='*55}")
    train_model("attn_probe", pretrained=True, freeze_backbone=True, **shared)


if __name__ == "__main__":
    main()
