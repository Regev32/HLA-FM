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

import torch

sys.path.insert(0, os.path.dirname(__file__))
from encoders.attn_encoder import MHCAttentionEncoder

with open("config/train_config.json") as f:
    CFG = json.load(f)

CHECKPOINTS = {
    "stage1": os.path.join(CFG["s1_pretrained_mae_dir"], "mae_pretrained.pt"),
    "stage2": os.path.join(CFG["results_dir"], "attn", "best_model.pt"),
}


def load_encoder(stage: str, device: str = None) -> MHCAttentionEncoder:
    """
    Load a trained MHC encoder.

    Args:
        stage:  "stage1" (MAE pretrained) or "stage2" (contrastive fine-tuned)
        device: torch device string. Auto-detected if None.

    Returns:
        MHCAttentionEncoder in eval mode.
    """
    if stage not in CHECKPOINTS:
        raise ValueError(f"stage must be 'stage1' or 'stage2', got '{stage}'")

    if device is None:
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

    ckpt_path = CHECKPOINTS[stage]
    if not os.path.exists(ckpt_path):
        script = "pretrain_mae.py" if stage == "stage1" else "train_mhc.py"
        raise FileNotFoundError(
            f"No checkpoint found at {ckpt_path}. Run {script} first."
        )

    ckpt = torch.load(ckpt_path, map_location=device)

    # Stage 1 saves arch in the checkpoint.
    # Stage 2 doesn't, so we read the backbone arch from config and infer
    # output_dim from the saved projection weight shape.
    arch       = ckpt.get("arch", {"embed_dim": CFG["s1_embed_dim"], "nhead": CFG["s1_nhead"], "num_layers": CFG["s1_num_layers"], "output_dim": CFG["s1_output_dim"], "dropout": CFG["s1_dropout"]})
    output_dim = ckpt["mhc_encoder"]["proj.weight"].shape[0]

    encoder = MHCAttentionEncoder(
        vocab_size = 23,
        embed_dim  = arch["embed_dim"],
        nhead      = arch["nhead"],
        num_layers = arch["num_layers"],
        output_dim = output_dim,
        dropout    = arch.get("dropout", 0.1),
    ).to(device)

    encoder.load_state_dict(ckpt["mhc_encoder"])
    encoder.eval()

    print(f"Loaded {stage} encoder from {ckpt_path}  (val_loss={ckpt['val_loss']:.4f})")
    return encoder
