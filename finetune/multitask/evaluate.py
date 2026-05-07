"""
Generic multi-task checkpoint evaluation.

Produces per-task Recall@K, Precision@K, and ROC plots for each model,
saved under results/finetune/multitask/.

    run(TASKS)   # TASKS = [PeptideMHCTask(), MHCTCRTask(), ...]
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
from encoders.attn_encoder import MHCAttentionEncoder
from encoders.modern_attn_encoder import ModernMHCEncoder
from encoders.utils import build_mlp

with open(ROOT / "config/train_config.json") as f:
    CFG = json.load(f)

TASK_RESULTS_DIR = os.path.join(CFG["finetune_results_dir"], "multitask")

# ── Checkpoint loading ────────────────────────────────────────────────────────

def _load_checkpoint(ckpt_path: str, tasks: list, device: str):
    ckpt        = torch.load(ckpt_path, map_location=device)
    output_dim  = ckpt.get("output_dim", CFG["s2_output_dim"])
    proj_layers = ckpt.get("proj_layers", 1)
    state       = ckpt["mhc_encoder"]

    encoder_type = ckpt.get("encoder_type",
                            "modern" if "rope_cos" in state else "classic")

    common = dict(
        vocab_size  = 23,
        embed_dim   = CFG["s1_embed_dim"],
        nhead       = CFG["s1_nhead"],
        num_layers  = CFG["s1_num_layers"],
        output_dim  = output_dim,
        dropout     = CFG["s2_dropout"],
        proj_layers = proj_layers,
    )

    if encoder_type == "modern":
        ffn_hidden_dim = state["transformer.layers.0.ffn.experts.0.gate.weight"].shape[0]
        encoder = ModernMHCEncoder(
            **common,
            use_moe        = CFG["s1_modern_use_moe"],
            num_experts    = CFG["s1_modern_num_experts"],
            top_k          = CFG["s1_modern_top_k"],
            ffn_hidden_dim = ffn_hidden_dim,
        ).to(device)
    else:
        encoder = MHCAttentionEncoder(**common).to(device)

    encoder.load_state_dict(state)
    encoder.eval()

    projectors = {}
    for task in tasks:
        proj = build_mlp(task.query_dim, output_dim, proj_layers, CFG["s2_dropout"]).to(device)
        proj.load_state_dict(ckpt["projectors"][task.name])
        proj.eval()
        projectors[task.name] = proj

    return encoder, projectors

# ── Metrics ───────────────────────────────────────────────────────────────────

def _recall_at_k(scores: torch.Tensor, pos_set: set, k: int) -> float:
    topk = scores.topk(k).indices.tolist()
    return len(set(topk) & pos_set) / len(pos_set)


def _precision_at_k(scores: torch.Tensor, pos_set: set, k: int) -> float:
    topk = scores.topk(k).indices.tolist()
    return len(set(topk) & pos_set) / k


def _compute_roc_curve(scores: torch.Tensor, query_pos: dict,
                        neg_ratio: int = 10, seed: int = 42):
    rng      = np.random.default_rng(seed)
    fpr_grid = np.linspace(0, 1, 300)
    n_items  = scores.shape[1]
    all_tprs = []
    all_aucs = []

    for i, pos_set in query_pos.items():
        if len(pos_set) == 0:
            continue
        pos_list = list(pos_set)
        n_neg    = min(len(pos_set) * neg_ratio, n_items - len(pos_set))
        neg_mask           = np.ones(n_items, dtype=bool)
        neg_mask[pos_list] = False
        neg_sample         = rng.choice(np.where(neg_mask)[0], size=n_neg, replace=False)

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


def _compute_metrics(scores, query_pos, k_values):
    recalls_per_k, precisions_per_k = [], []
    valid = [(i, ps) for i, ps in query_pos.items() if len(ps) > 0]
    for k in k_values:
        recalls    = [_recall_at_k(scores[i], ps, k)    for i, ps in valid]
        precisions = [_precision_at_k(scores[i], ps, k) for i, ps in valid]
        mean_r = float(np.mean(recalls))
        mean_p = float(np.mean(precisions))
        recalls_per_k.append(mean_r)
        precisions_per_k.append(mean_p)
        print(f"    Recall@{k:<5} = {mean_r:.4f}  Precision@{k:<5} = {mean_p:.4f}")

    fpr_grid, mean_tpr, mean_auc = _compute_roc_curve(scores, query_pos)
    print(f"    ROC AUC = {mean_auc:.4f}")
    return {"recall": recalls_per_k, "precision": precisions_per_k,
            "fpr": fpr_grid, "tpr": mean_tpr, "auc": mean_auc}

# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate(models: dict[str, str], tasks: list, device: str = None):
    if device is None:
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

    # ── Load test data for each task ─────────────────────────────────────────
    task_test_data = {}
    for task in tasks:
        split = torch.load(task.test_split_path, map_location="cpu")
        mhc_tokens = split["mhc_tokens"]
        test_idx   = split["test_idx"]        # list of global query indices
        pos_dict   = split["pos_dict"]        # mhc_i → set of local query indices

        cached = torch.load(task.embed_cache_path, map_location="cpu")
        all_embs   = cached["embs"]
        test_embs  = all_embs[test_idx]       # (n_test, query_dim)

        # Invert: local_query_i → set[mhc_i]
        query_pos = {}
        for mhc_i, q_set in pos_dict.items():
            for q_j in q_set:
                query_pos.setdefault(q_j, set()).add(mhc_i)

        task_test_data[task.name] = {
            "mhc_tokens": mhc_tokens.to(device),
            "test_embs":  test_embs,
            "query_pos":  query_pos,
            "batch_size": task.batch_size,
        }

    # task_name → model_name → metrics
    all_results = {task.name: {} for task in tasks}

    for model_name, ckpt_path in models.items():
        print(f"\nEvaluating: {model_name}  ({ckpt_path})")
        encoder, projectors = _load_checkpoint(ckpt_path, tasks, device)

        for task in tasks:
            td         = task_test_data[task.name]
            mhc_tokens = td["mhc_tokens"]
            test_embs  = td["test_embs"]
            query_pos  = td["query_pos"]
            bs         = td["batch_size"]
            k_values   = task.get_eval_k_values()

            with torch.no_grad():
                mhc_emb = F.normalize(encoder(mhc_tokens), dim=-1)

            n_test = test_embs.shape[0]
            scores = torch.zeros(n_test, mhc_tokens.shape[0])
            proj   = projectors[task.name]
            with torch.no_grad():
                for start in range(0, n_test, bs):
                    q = F.normalize(proj(test_embs[start : start + bs].to(device)), dim=-1)
                    scores[start : start + bs] = (q @ mhc_emb.T).cpu()

            print(f"  [{task.name}]")
            all_results[task.name][model_name] = _compute_metrics(scores, query_pos, k_values)

    os.makedirs(TASK_RESULTS_DIR, exist_ok=True)
    for task in tasks:
        results  = all_results[task.name]
        k_values = task.get_eval_k_values()
        _save_bar_plot(results, k_values, metric="recall",
                       title=f"Recall@K — {task.name}",
                       ylabel="Mean Recall",
                       out_path=os.path.join(TASK_RESULTS_DIR, f"{task.name}_recall_at_k.png"))
        _save_bar_plot(results, k_values, metric="precision",
                       title=f"Precision@K — {task.name}",
                       ylabel="Mean Precision",
                       out_path=os.path.join(TASK_RESULTS_DIR, f"{task.name}_precision_at_k.png"))
        _save_roc_plot(results,
                       title=f"ROC — {task.name}",
                       out_path=os.path.join(TASK_RESULTS_DIR, f"{task.name}_roc_all_models.png"))

    return all_results

# ── Plotting ──────────────────────────────────────────────────────────────────

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


def _save_roc_plot(results, title, out_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, metrics in results.items():
        ax.plot(metrics["fpr"], metrics["tpr"],
                label=f"{name}  (AUC={metrics['auc']:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {out_path}")

# ── Entry point ───────────────────────────────────────────────────────────────

def run(tasks: list):
    models = {
        "attn": os.path.join(TASK_RESULTS_DIR, "attn", "best_model.pt"),
    }
    if CFG.get("compare_architectures", False):
        models["modern_attn"] = os.path.join(TASK_RESULTS_DIR, "modern_attn", "best_model.pt")
    models = {k: v for k, v in models.items() if os.path.exists(v)}
    evaluate(models, tasks)


if __name__ == "__main__":
    from finetune.binding.peptide_mhc import PeptideMHCTask
    from finetune.binding.mhc_tcr import MHCTCRTask
    run([PeptideMHCTask(), MHCTCRTask()])
