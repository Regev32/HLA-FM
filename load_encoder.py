"""
Load a trained MHC encoder from a stage 1 or stage 2 checkpoint.

Usage:
    from load_encoder import load_encoder

    encoder = load_encoder("stage1")   # MAE pretrained backbone
    encoder = load_encoder("stage2")   # contrastive fine-tuned
"""

import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from encoders.attn_encoder import MHCAttentionEncoder
from encoders.modern_attn_encoder import ModernMHCEncoder

with open(ROOT / "config/train_config.json") as f:
    CFG = json.load(f)

CHECKPOINTS = {
    "stage1":        os.path.join(CFG["s1_pretrained_mae_dir"], "mae_pretrained.pt"),
    "stage1_modern": os.path.join(CFG["s1_modern_mae_dir"],     "mae_pretrained.pt"),
    "stage2":        os.path.join(CFG["finetune_results_dir"], "peptide_mhc", "attn",        "best_model.pt"),
    "stage2_modern": os.path.join(CFG["finetune_results_dir"], "peptide_mhc", "modern_attn", "best_model.pt"),
}


def load_encoder(stage: str, device: str = None):
    """
    Load a trained MHC encoder.

    Args:
        stage:  "stage1" | "stage1_modern" | "stage2" | "stage2_modern"
        device: torch device string. Auto-detected if None.

    Returns:
        MHCAttentionEncoder or ModernMHCEncoder in eval mode.
    """
    if stage not in CHECKPOINTS:
        raise ValueError(f"stage must be one of {list(CHECKPOINTS)}, got '{stage}'")

    if device is None:
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

    ckpt_path = CHECKPOINTS[stage]
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}. Train the model first.")

    ckpt  = torch.load(ckpt_path, map_location=device)
    state = ckpt["mhc_encoder"]

    # detect arch: saved tag → key inspection fallback
    encoder_type = ckpt.get("encoder_type",
                            "modern" if "rope_cos" in state else "classic")

    arch        = ckpt.get("arch", {"embed_dim": CFG["s1_embed_dim"], "nhead": CFG["s1_nhead"],
                                    "num_layers": CFG["s1_num_layers"], "output_dim": CFG["s1_output_dim"],
                                    "dropout": CFG["s1_dropout"]})
    output_dim  = ckpt.get("output_dim", arch.get("output_dim", CFG["s1_output_dim"]))
    proj_layers = ckpt.get("proj_layers", 1)

    common = dict(
        vocab_size  = 23,
        embed_dim   = arch["embed_dim"],
        nhead       = arch["nhead"],
        num_layers  = arch["num_layers"],
        output_dim  = output_dim,
        dropout     = arch.get("dropout", 0.1),
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

    print(f"Loaded {stage} ({encoder_type}) encoder from {ckpt_path}  (val_loss={ckpt['val_loss']:.4f})")
    return encoder
