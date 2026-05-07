"""
Generic multi-task contrastive finetuning for MHC binding tasks.

The MHC encoder is shared across all tasks. Each task has its own projector
head. Per step: loss = mean(task losses). Configured via main.py:

    TASKS = [PeptideMHCTask(), MHCTCRTask()]
    run(TASKS)

To add a new task: create finetune/binding/<task>.py, append to TASKS.
"""

import copy
import json
import os
import sys
from itertools import cycle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
from encoders.attn_encoder import MHCAttentionEncoder
from encoders.modern_attn_encoder import ModernMHCEncoder, find_matching_hidden_dim
from encoders.utils import build_mlp

with open(ROOT / "config/train_config.json") as f:
    CFG = json.load(f)

TASK_RESULTS_DIR = os.path.join(CFG["finetune_results_dir"], "multitask")

PARAMS = {
    "output_dim": CFG["s2_output_dim"],
    "lr":         CFG["s2_lr"],
    "dropout":    CFG["s2_dropout"],
}

# ── Device placement ──────────────────────────────────────────────────────────

def _to_device_if_fits(tensor: torch.Tensor, device: str, label: str) -> torch.Tensor:
    if not device.startswith("cuda"):
        return tensor
    size_gb = tensor.numel() * tensor.element_size() / 1e9
    free_bytes, _ = torch.cuda.mem_get_info(device)
    if tensor.numel() * tensor.element_size() < free_bytes * 0.8:
        print(f"  {label} cache ({size_gb:.2f} GB) fits — loading to {device}")
        return tensor.to(device)
    print(f"  {label} cache ({size_gb:.2f} GB) too large — keeping on CPU")
    return tensor

# ── Loss + masking ────────────────────────────────────────────────────────────

def multi_positive_infonce(mhc_emb, q_emb, pos_mask):
    logits = (mhc_emb @ q_emb.T) / CFG["s2_temp"]

    def _anchor_loss(lg, mask):
        log_denom = torch.logsumexp(lg, dim=1, keepdim=True)
        log_probs = lg - log_denom
        n_pos     = mask.float().sum(dim=1).clamp(min=1)
        loss_per  = -(log_probs * mask.float()).sum(dim=1) / n_pos
        return loss_per[mask.any(dim=1)].mean()

    return (_anchor_loss(logits, pos_mask) + _anchor_loss(logits.T, pos_mask.T)) / 2


def make_pos_mask(pos_dict, batch_indices, n_mhc, device):
    local_map = {int(g): l for l, g in enumerate(batch_indices)}
    batch_set = set(local_map)
    mask = torch.zeros(n_mhc, len(batch_indices), dtype=torch.bool)
    for mhc_idx, pos_items in pos_dict.items():
        for item_idx in pos_items & batch_set:
            mask[mhc_idx, local_map[item_idx]] = True
    return mask.to(device)

# ── Epoch loop ────────────────────────────────────────────────────────────────

def run_epoch(mhc_encoder, projectors, task_data, split, device, optimizer=None):
    """
    projectors : dict[task.name → nn.Module]
    task_data  : list of dicts with keys: task, mhc_tokens, embs,
                 train_idx, val_idx, train_pos, val_pos
    split      : "train" | "val"
    Returns    : (avg_loss, {task.name: loss})
    """
    training = optimizer is not None
    mhc_encoder.train(training)
    for proj in projectors.values():
        proj.train(training)

    ctx = torch.enable_grad() if training else torch.no_grad()

    # Build shuffled batch lists per task
    batch_lists = []
    for td in task_data:
        indices = td[f"{split}_idx"].copy() if training else td[f"{split}_idx"]
        if training:
            np.random.shuffle(indices)
        bs = td["task"].batch_size
        batch_lists.append([indices[s : s + bs] for s in range(0, len(indices), bs)])

    # Cycle shorter lists to match the longest
    max_len = max(len(bl) for bl in batch_lists)
    iters   = [iter(bl) if len(bl) == max_len else cycle(bl) for bl in batch_lists]

    total_loss  = 0.0
    task_losses = {td["task"].name: 0.0 for td in task_data}
    n_steps     = 0

    with ctx:
        for batch_tuple in zip(*iters):
            step_loss = torch.tensor(0.0, device=device)

            for td, batch_idx in zip(task_data, batch_tuple):
                n_mhc    = td["mhc_tokens"].shape[0]
                mhc_emb  = F.normalize(mhc_encoder(td["mhc_tokens"]), dim=-1)
                q_emb    = F.normalize(
                    projectors[td["task"].name](td["embs"][batch_idx].to(device)), dim=-1
                )
                pos_mask = make_pos_mask(td[f"{split}_pos"], batch_idx, n_mhc, device)

                t_loss = (multi_positive_infonce(mhc_emb, q_emb, pos_mask)
                          if pos_mask.any() else torch.tensor(0.0, device=device))
                step_loss = step_loss + t_loss
                task_losses[td["task"].name] += t_loss.item()

            step_loss = step_loss / len(task_data)

            if training:
                optimizer.zero_grad()
                step_loss.backward()
                optimizer.step()

            total_loss += step_loss.item()
            n_steps    += 1

    n = max(n_steps, 1)
    return total_loss / n, {k: v / n for k, v in task_losses.items()}

# ── Encoder builder ───────────────────────────────────────────────────────────

def build_encoder(params, device, pretrained=True, unfreeze_layers=None, arch="classic"):
    mae_dir       = (CFG["s1_modern_mae_dir"] if arch == "modern"
                     else CFG["s1_pretrained_mae_dir"])
    mae_ckpt_path = os.path.join(mae_dir, "mae_pretrained.pt")

    base_kwargs = dict(
        vocab_size  = 23,
        embed_dim   = CFG["s1_embed_dim"],
        nhead       = CFG["s1_nhead"],
        num_layers  = CFG["s1_num_layers"],
        dropout     = params.get("dropout", 0.1),
        output_dim  = params["output_dim"],
        proj_layers = CFG["s2_proj_layers"],
    )
    if arch == "modern":
        base_kwargs.update(
            use_moe     = CFG["s1_modern_use_moe"],
            num_experts = CFG["s1_modern_num_experts"],
            top_k       = CFG["s1_modern_top_k"],
        )

    EncoderClass = ModernMHCEncoder if arch == "modern" else MHCAttentionEncoder

    def _make(extra=None):
        kw = {**base_kwargs, **(extra or {})}
        if arch == "modern":
            ref = MHCAttentionEncoder(**{k: v for k, v in kw.items()
                                         if k not in ("use_moe","num_experts","top_k","ffn_hidden_dim")})
            target_n = sum(p.numel() for p in ref.parameters())
            kw["ffn_hidden_dim"] = find_matching_hidden_dim(
                target_n, {k: v for k, v in kw.items() if k != "ffn_hidden_dim"}
            )
        return EncoderClass(**kw).to(device)

    if pretrained and os.path.exists(mae_ckpt_path):
        ckpt = torch.load(mae_ckpt_path, map_location=device)
        sa   = ckpt["arch"]
        encoder = _make({"embed_dim": sa["embed_dim"], "nhead": sa["nhead"],
                          "num_layers": sa["num_layers"],
                          "dropout": params.get("dropout", sa["dropout"])})
        backbone_state = {k: v for k, v in ckpt["mhc_encoder"].items()
                          if not k.startswith("proj.")}
        encoder.load_state_dict(backbone_state, strict=False)
        print(f"  Loaded {arch} MAE pretrained backbone from {mae_ckpt_path}")
    else:
        if pretrained:
            print(f"  No checkpoint at {mae_ckpt_path} — falling back to random init")
        encoder = _make()

    if unfreeze_layers is not None:
        for p in encoder.parameters():
            p.requires_grad_(False)
        for p in encoder.proj.parameters():
            p.requires_grad_(True)
        if unfreeze_layers > 0:
            total = len(encoder.transformer.layers)
            for layer in encoder.transformer.layers[max(0, total - unfreeze_layers):]:
                for p in layer.parameters():
                    p.requires_grad_(True)
        n_train = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in encoder.parameters())
        print(f"  Unfroze proj + last {unfreeze_layers} layers "
              f"({n_train:,}/{n_total:,} params trainable)")

    return encoder

# ── Training run ──────────────────────────────────────────────────────────────

def retrain_best(params, pretrained, unfreeze_layers, task_data, device, results_dir, arch):
    mhc_encoder = build_encoder(params, device, pretrained=pretrained,
                                 unfreeze_layers=unfreeze_layers, arch=arch)
    projectors = {
        td["task"].name: build_mlp(td["task"].query_dim, params["output_dim"],
                                    CFG["s2_proj_layers"], CFG["s2_dropout"]).to(device)
        for td in task_data
    }

    all_proj_params = [p for proj in projectors.values() for p in proj.parameters()]
    optimizer = torch.optim.Adam(
        [p for p in mhc_encoder.parameters() if p.requires_grad] + all_proj_params,
        lr=params["lr"],
    )

    best_val     = float("inf")
    patience     = 0
    decays_left  = CFG["s2_lr_decays"]
    best_weights = {}
    train_losses, val_losses = [], []

    for epoch in range(CFG["s2_n_epochs_final"]):
        train_avg, train_per = run_epoch(mhc_encoder, projectors, task_data,
                                          "train", device, optimizer=optimizer)
        val_avg,   val_per   = run_epoch(mhc_encoder, projectors, task_data,
                                          "val",   device, optimizer=None)

        train_losses.append(train_avg)
        val_losses.append(val_avg)

        lr = optimizer.param_groups[0]["lr"]
        train_str = "  ".join(f"{k}={v:.4f}" for k, v in train_per.items())
        val_str   = "  ".join(f"{k}={v:.4f}" for k, v in val_per.items())
        print(f"  epoch {epoch+1:2d}/{CFG['s2_n_epochs_final']}  "
              f"train [{train_str}] avg={train_avg:.4f}  "
              f"val [{val_str}] avg={val_avg:.4f}  lr={lr:.1e}")

        if val_avg < best_val:
            best_val, patience = val_avg, 0
            best_weights = {
                "mhc_encoder":  copy.deepcopy(mhc_encoder.state_dict()),
                "projectors":   {k: copy.deepcopy(v.state_dict()) for k, v in projectors.items()},
                "val_loss":     best_val,
                "output_dim":   params["output_dim"],
                "proj_layers":  CFG["s2_proj_layers"],
                "encoder_type": arch,
                "task_names":   [td["task"].name for td in task_data],
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


def _train_model(results_name, pretrained, unfreeze_layers, task_data, device, arch):
    results_dir = os.path.join(TASK_RESULTS_DIR, results_name)
    os.makedirs(results_dir, exist_ok=True)
    train_losses, val_losses = retrain_best(
        PARAMS, pretrained, unfreeze_layers, task_data, device, results_dir, arch
    )
    _save_loss_plot(train_losses, val_losses, results_name, results_dir)
    return train_losses, val_losses

# ── Test split saving ─────────────────────────────────────────────────────────

def _save_test_split(task, mhc_tokens, query_list, test_idx, test_pos):
    g2l = {int(g): l for l, g in enumerate(test_idx)}
    remapped = {mhc_i: {g2l[q] for q in qs if q in g2l}
                for mhc_i, qs in test_pos.items()}
    os.makedirs(os.path.dirname(task.test_split_path), exist_ok=True)
    torch.save({
        "mhc_tokens":  mhc_tokens,
        "test_idx":    test_idx.tolist(),
        "pos_dict":    remapped,
        "query_list":  query_list,
        "query_key":   task.query_key,
    }, task.test_split_path)
    print(f"Saved {task.name} test split to {task.test_split_path}")

# ── Plotting ──────────────────────────────────────────────────────────────────

def _save_loss_plot(train_losses, val_losses, title, results_dir):
    epochs = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots()
    ax.plot(epochs, train_losses, label="train")
    ax.plot(epochs, val_losses,   label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (avg)")
    ax.set_title(title)
    ax.legend()
    fig.savefig(os.path.join(results_dir, "loss_curve.png"), dpi=150)
    plt.close(fig)


def _save_combined_loss_plot(all_losses, out_path):
    order   = ["attn", "modern_attn"]
    present = [n for n in order if n in all_losses]
    if not present:
        return
    fig, axes = plt.subplots(1, len(present), figsize=(6 * len(present), 5), sharey=True)
    if len(present) == 1:
        axes = [axes]
    for ax, name in zip(axes, present):
        tl, vl = all_losses[name]
        epochs = range(1, len(tl) + 1)
        ax.plot(epochs, tl, label="train")
        ax.plot(epochs, vl, label="val")
        ax.set_xlabel("Epoch")
        ax.set_title(name)
        ax.legend()
    axes[0].set_ylabel("Loss (avg)")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined loss plot to {out_path}")

# ── Main ──────────────────────────────────────────────────────────────────────

def run(tasks: list):
    np.random.seed(CFG["seed"])
    torch.manual_seed(CFG["seed"])
    torch.cuda.manual_seed_all(CFG["seed"])

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    task_data = []
    for task in tasks:
        mhc_tokens, query_list, train_idx, val_idx, test_idx, \
            train_pos, val_pos, test_pos = task.load_data()

        print(f"{task.name} — MHCs: {mhc_tokens.shape[0]}  "
              f"Queries: {len(query_list):,}  "
              f"(train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)})")

        embs = task.encode_queries(query_list)
        embs = _to_device_if_fits(embs, device, task.name)
        mhc_tokens = mhc_tokens.to(device)
        _save_test_split(task, mhc_tokens.cpu(), query_list, test_idx, test_pos)

        task_data.append({
            "task":      task,
            "mhc_tokens": mhc_tokens,
            "embs":      embs,
            "train_idx": train_idx,
            "val_idx":   val_idx,
            "train_pos": train_pos,
            "val_pos":   val_pos,
        })

    all_losses = {}

    print(f"\n{'='*60}\n  Model: attn  (classic pretrained)\n{'='*60}")
    all_losses["attn"] = _train_model(
        "attn", pretrained=True,
        unfreeze_layers=CFG["s2_unfreeze_layers"],
        task_data=task_data, device=device, arch="classic",
    )

    if CFG.get("compare_architectures", False):
        print(f"\n{'='*60}\n  Model: modern_attn  (modern pretrained)\n{'='*60}")
        all_losses["modern_attn"] = _train_model(
            "modern_attn", pretrained=True,
            unfreeze_layers=CFG["s2_unfreeze_layers"],
            task_data=task_data, device=device, arch="modern",
        )

    os.makedirs(TASK_RESULTS_DIR, exist_ok=True)
    _save_combined_loss_plot(all_losses, os.path.join(TASK_RESULTS_DIR, "models_loss.png"))


if __name__ == "__main__":
    from finetune.binding.peptide_mhc import PeptideMHCTask
    from finetune.binding.mhc_tcr import MHCTCRTask
    run([PeptideMHCTask(), MHCTCRTask()])
