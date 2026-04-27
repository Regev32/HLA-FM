"""
Evaluate and compare trained MHC encoder checkpoints using Recall@K, Precision@K,
and ROC curves on held-out test peptides (saved by peptide_mhc.py).

Usage:
    from evaluate_peptide_mhc import evaluate

    evaluate({
        "attn":          "results/attn/best_model.pt",
        "attn_random": "results/attn_random/best_model.pt",
    })

    # or run directly (evaluates the default stage 2 checkpoint):
    python evaluate_peptide_mhc.py
"""

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
from encoders.utils import build_mlp

with open("config/train_config.json") as f:
    CFG = json.load(f)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_checkpoint(ckpt_path: str, device: str):
    """Load encoder + pep_proj from a stage 2 checkpoint."""
    ckpt        = torch.load(ckpt_path, map_location=device)
    arch        = ckpt.get("arch", {"embed_dim": CFG["s1_embed_dim"], "nhead": CFG["s1_nhead"],
                                    "num_layers": CFG["s1_num_layers"], "output_dim": CFG["s1_output_dim"],
                                    "dropout": CFG["s1_dropout"]})
    output_dim  = ckpt.get("output_dim", CFG["s2_output_dim"])
    proj_layers = ckpt.get("proj_layers", 1)

    encoder = MHCAttentionEncoder(
        vocab_size  = 23,
        embed_dim   = arch["embed_dim"],
        nhead       = arch["nhead"],
        num_layers  = arch["num_layers"],
        output_dim  = output_dim,
        dropout     = CFG["s2_dropout"],
        proj_layers = proj_layers,
    ).to(device)
    encoder.load_state_dict(ckpt["mhc_encoder"])
    encoder.eval()

    pep_proj = build_mlp(CFG["s2_peptide_dim"], output_dim, proj_layers, CFG["s2_dropout"]).to(device)
    pep_proj.load_state_dict(ckpt["pep_proj"])
    pep_proj.eval()

    return encoder, pep_proj


def _recall_at_k(scores: torch.Tensor, pos_set: set, k: int) -> float:
    topk = scores.topk(k).indices.tolist()
    return len(set(topk) & pos_set) / len(pos_set)


def _precision_at_k(scores: torch.Tensor, pos_set: set, k: int) -> float:
    topk = scores.topk(k).indices.tolist()
    return len(set(topk) & pos_set) / k


def _compute_roc_curve(scores: torch.Tensor, pos_dict: dict,
                       neg_ratio: int = 10, seed: int = 42):
    """
    Compute a mean ROC curve across alleles, sampling neg_ratio * |positives|
    negatives per allele so FPR is not dominated by the full negative pool.

    Returns (fpr_grid, mean_tpr, mean_auc).
    """
    rng      = np.random.default_rng(seed)
    fpr_grid = np.linspace(0, 1, 300)
    n_pep    = scores.shape[1]
    all_tprs = []
    all_aucs = []

    for i, pos_set in pos_dict.items():
        if len(pos_set) == 0:
            continue

        pos_list = list(pos_set)
        n_neg    = min(len(pos_set) * neg_ratio, n_pep - len(pos_set))

        neg_mask         = np.ones(n_pep, dtype=bool)
        neg_mask[pos_list] = False
        neg_pool         = np.where(neg_mask)[0]
        neg_sample       = rng.choice(neg_pool, size=n_neg, replace=False)

        subset_indices = np.concatenate([pos_list, neg_sample])
        subset_scores  = scores[i, subset_indices].numpy()
        labels         = np.array([1] * len(pos_list) + [0] * n_neg)

        order         = np.argsort(-subset_scores)
        sorted_labels = labels[order]

        cum_tp = np.cumsum(sorted_labels)
        cum_fp = np.cumsum(1 - sorted_labels)
        tpr    = np.concatenate([[0], cum_tp / len(pos_list)])
        fpr    = np.concatenate([[0], cum_fp / n_neg])

        all_tprs.append(np.interp(fpr_grid, fpr, tpr))
        all_aucs.append(float(np.trapezoid(tpr, fpr)))

    return fpr_grid, np.mean(all_tprs, axis=0), float(np.mean(all_aucs))

# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate(models: dict[str, str], device: str = None):
    if device is None:
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

    if not os.path.exists(CFG["s2_test_split_path"]):
        raise FileNotFoundError(
            f"Test split not found at {CFG['s2_test_split_path']}. Run peptide_mhc.py first."
        )

    test_data    = torch.load(CFG["s2_test_split_path"], map_location="cpu")
    mhc_tokens   = test_data["mhc_tokens"]       # all MHC alleles
    test_pep_idx = test_data["test_pep_idx"]      # global peptide indices in test set
    pos_dict     = test_data["pos_dict"]           # mhc_local → set of LOCAL test pep indices

    cached   = torch.load(CFG["s2_pep_embed_cache"], map_location="cpu")
    pep_embs = cached["embs"]
    if cached.get("peptide_list") != test_data.get("peptide_list"):
        raise RuntimeError(
            "Peptide list in cache does not match test split. "
            "Delete the cache and re-run peptide_mhc.py."
        )

    test_pep_embs = pep_embs[test_pep_idx]  # only test peptide embeddings
    n_test_pep    = len(test_pep_idx)
    n_mhc         = len(mhc_tokens)
    k_values      = CFG["eval_k_values"]

    # Transpose pos_dict: mhc→{pep} into pep→{mhc} for peptide-centric eval
    pep_pos_dict = {}
    for mhc_i, pep_set in pos_dict.items():
        for pep_j in pep_set:
            pep_pos_dict.setdefault(pep_j, set()).add(mhc_i)

    results = {}

    for name, ckpt_path in models.items():
        print(f"\nEvaluating: {name}  ({ckpt_path})")
        encoder, pep_proj = _load_checkpoint(ckpt_path, device)

        with torch.no_grad():
            mhc_emb = F.normalize(encoder(mhc_tokens.to(device)), dim=-1)

        # Score matrix: (N_test_pep, N_mhc) — for each peptide, score all alleles
        scores = torch.zeros(n_test_pep, n_mhc)
        batch  = CFG["s2_peptides_per_batch"]

        with torch.no_grad():
            for start in range(0, n_test_pep, batch):
                pep_batch = F.normalize(
                    pep_proj(test_pep_embs[start : start + batch].to(device)), dim=-1
                )
                scores[start : start + batch] = (pep_batch @ mhc_emb.T).cpu()

        recalls_per_k, precisions_per_k = [], []
        for k in k_values:
            valid      = [(i, ps) for i, ps in pep_pos_dict.items() if len(ps) > 0]
            recalls    = [_recall_at_k(scores[i], ps, k)    for i, ps in valid]
            precisions = [_precision_at_k(scores[i], ps, k) for i, ps in valid]

            mean_recall    = float(np.mean(recalls))
            mean_precision = float(np.mean(precisions))
            recalls_per_k.append(mean_recall)
            precisions_per_k.append(mean_precision)

            print(f"  Recall@{k:<5} = {mean_recall:.4f}  "
                  f"Precision@{k:<5} = {mean_precision:.4f}")

        fpr_grid, mean_tpr, mean_auc = _compute_roc_curve(scores, pep_pos_dict)
        results[name] = {"recall": recalls_per_k, "precision": precisions_per_k,
                         "fpr": fpr_grid, "tpr": mean_tpr, "auc": mean_auc}

    # ── Plots ─────────────────────────────────────────────────────────────────
    os.makedirs(CFG["results_dir"], exist_ok=True)
    _save_bar_plot(results, k_values, metric="recall",
                   title="Recall@K — per test peptide, ranking MHC alleles",
                   ylabel="Mean Recall",
                   out_path=os.path.join(CFG["results_dir"], "recall_at_k.png"))
    _save_bar_plot(results, k_values, metric="precision",
                   title="Precision@K — per test peptide, ranking MHC alleles",
                   ylabel="Mean Precision",
                   out_path=os.path.join(CFG["results_dir"], "precision_at_k.png"))
    _save_roc_plot(results,
                   out_path=os.path.join(CFG["results_dir"], "roc_all_models.png"))
    return results


def _save_bar_plot(results, k_values, metric, title, ylabel, out_path):
    x     = np.arange(len(k_values))
    width = 0.8 / max(len(results), 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (name, metrics) in enumerate(results.items()):
        ax.bar(x + i * width, metrics[metric], width, label=name)

    ax.set_xticks(x + width * (len(results) - 1) / 2)
    ax.set_xticklabels([f"@{k}" for k in k_values])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.legend()

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {out_path}")


def _save_roc_plot(results, out_path):
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, metrics in results.items():
        auc = metrics["auc"]
        ax.plot(metrics["fpr"], metrics["tpr"], label=f"{name}  (AUC={auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — per test peptide, ranking MHC alleles\n"
                 "(negatives sampled at 10x positives per peptide)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    evaluate({
        "attn":          os.path.join(CFG["results_dir"], "attn",          "best_model.pt"),
        "attn_random": os.path.join(CFG["results_dir"], "attn_random", "best_model.pt"),
        "attn_frozen":    os.path.join(CFG["results_dir"], "attn_frozen",    "best_model.pt"),
    })
