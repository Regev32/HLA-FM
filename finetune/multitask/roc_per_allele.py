"""
Standalone script: per-allele ROC curves for all trained multitask checkpoints.

For each task and each model checkpoint, computes a ROC curve by asking:
"given an allele, can the model rank its binders above non-binders in the test set?"

Run: python finetune/multitask/roc_per_allele.py
Outputs: results/finetune/multitask/{task}_roc_per_allele.png
"""

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
from finetune.multitask.evaluate import _load_checkpoint
from finetune.multitask.train import _to_device_if_fits
from finetune.binding.peptide_mhc import PeptideMHCTask
from finetune.binding.mhc_tcr import MHCTCRTask

with open(ROOT / "config/train_config.json") as f:
    CFG = json.load(f)

TASK_RESULTS_DIR = os.path.join(CFG["finetune_results_dir"], "multitask")
TASKS = [PeptideMHCTask(), MHCTCRTask()]


def _compute_roc_per_allele(scores_T, pos_dict, neg_ratio=10, seed=42):
    """
    scores_T : (n_mhc, n_test) — scores_T[j] = allele j's scores vs all test queries
    pos_dict : allele_j → set of local test query indices (positive binders)
    """
    rng      = np.random.default_rng(seed)
    fpr_grid = np.linspace(0, 1, 300)
    n_queries = scores_T.shape[1]
    all_tprs, all_aucs = [], []

    for j, pos_set in pos_dict.items():
        if len(pos_set) == 0:
            continue
        pos_list = list(pos_set)
        n_neg    = min(len(pos_set) * neg_ratio, n_queries - len(pos_set))
        neg_mask           = np.ones(n_queries, dtype=bool)
        neg_mask[pos_list] = False
        neg_sample         = rng.choice(np.where(neg_mask)[0], size=n_neg, replace=False)

        indices       = np.concatenate([pos_list, neg_sample])
        subset_scores = scores_T[j, indices].numpy()
        labels        = np.array([1] * len(pos_list) + [0] * n_neg)

        order         = np.argsort(-subset_scores)
        sorted_labels = labels[order]
        cum_tp = np.cumsum(sorted_labels)
        cum_fp = np.cumsum(1 - sorted_labels)
        tpr    = np.concatenate([[0], cum_tp / len(pos_list)])
        fpr    = np.concatenate([[0], cum_fp / n_neg])

        all_tprs.append(np.interp(fpr_grid, fpr, tpr))
        all_aucs.append(float(np.trapezoid(tpr, fpr)))

    mean_tpr = np.mean(all_tprs, axis=0)
    mean_auc = float(np.mean(all_aucs))
    return fpr_grid, mean_tpr, mean_auc


def main():
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    models = {"attn": os.path.join(TASK_RESULTS_DIR, "attn", "best_model.pt")}
    if CFG.get("compare_architectures", False):
        models["modern_attn"] = os.path.join(TASK_RESULTS_DIR, "modern_attn", "best_model.pt")
    models = {k: v for k, v in models.items() if os.path.exists(v)}

    for task in TASKS:
        split     = torch.load(task.test_split_path, map_location="cpu")
        mhc_tokens = split["mhc_tokens"].to(device)
        test_idx   = split["test_idx"]
        pos_dict   = split["pos_dict"]

        cached    = torch.load(task.embed_cache_path, map_location="cpu")
        test_embs = _to_device_if_fits(cached["embs"][test_idx], device, task.name)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="random")

        for model_name, ckpt_path in models.items():
            encoder, projectors = _load_checkpoint(ckpt_path, TASKS, device)
            encoder.eval()
            proj = projectors[task.name]

            with torch.no_grad():
                mhc_emb = F.normalize(encoder(mhc_tokens), dim=-1)

            n_test = test_embs.shape[0]
            bs     = task.batch_size
            scores = torch.zeros(n_test, mhc_tokens.shape[0])
            with torch.no_grad():
                for start in range(0, n_test, bs):
                    q = F.normalize(proj(test_embs[start : start + bs].to(device)), dim=-1)
                    scores[start : start + bs] = (q @ mhc_emb.T).cpu()

            fpr, tpr, auc = _compute_roc_per_allele(scores.T, pos_dict)
            print(f"  [{task.name}] {model_name}  per-allele AUC = {auc:.4f}")
            ax.plot(fpr, tpr, label=f"{model_name}  (AUC={auc:.4f})")

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC per allele — {task.name}")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.legend()
        out = os.path.join(TASK_RESULTS_DIR, f"{task.name}_roc_per_allele.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out}")


if __name__ == "__main__":
    main()
