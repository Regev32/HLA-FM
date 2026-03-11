import torch
import torch.nn as nn

from encoders.utils import PAD_IDX, _tokenize

# Safe upper bound for positional encoding (max MHC protein length ~379 + CLS)
MAX_POS = 512

DEFAULTS = dict(
    vocab_size = 22,
    embed_dim  = 128,
    nhead      = 8,
    num_layers = 4,
    output_dim = 256,
    dropout    = 0.1,
)


class MHCAttentionEncoder(nn.Module):
    def __init__(self, **kwargs):
        cfg        = {**DEFAULTS, **kwargs}
        vocab_size = cfg["vocab_size"]
        embed_dim  = cfg["embed_dim"]
        nhead      = cfg["nhead"]
        num_layers = cfg["num_layers"]
        output_dim = cfg["output_dim"]
        dropout    = cfg["dropout"]

        super().__init__()
        self.embed_dim = embed_dim  # exposed for external callers (e.g. MAE decoder)

        self.embedding     = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.pos_embedding = nn.Embedding(MAX_POS, embed_dim)
        self.cls_token     = nn.Parameter(torch.zeros(1, 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = embed_dim,
            nhead           = nhead,
            dim_feedforward = embed_dim * 4,
            dropout         = dropout,
            batch_first     = True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj = nn.Linear(embed_dim, output_dim)

    def forward(self, x: torch.Tensor, return_sequence: bool = False) -> torch.Tensor:
        """
        Args:
            x:               (batch, seq_len) token indices
            return_sequence: if True, return (batch, seq_len, embed_dim) token-level
                             embeddings for the original AA positions (no pooling/proj).
                             Used by the MAE decoder during pretraining.
        """
        h   = self.embedding(x)                            # (B, L, D)
        cls = self.cls_token.expand(h.size(0), -1, -1)
        h   = torch.cat([cls, h], dim=1)                   # (B, L+1, D)

        positions = torch.arange(h.size(1), device=h.device)
        h = h + self.pos_embedding(positions)

        pad_mask = (x == PAD_IDX)                          # (B, L)
        cls_mask = torch.zeros(x.size(0), 1, dtype=torch.bool, device=x.device)
        pad_mask = torch.cat([cls_mask, pad_mask], dim=1) # (B, L+1) — CLS never masked

        h = self.transformer(h, src_key_padding_mask=pad_mask)  # (B, L+1, D)

        if return_sequence:
            return h[:, 1:, :]                             # (B, L, D) — AA positions only

        return self.proj(h[:, 0, :])                       # (B, output_dim) — CLS token

    def encode(self, seqs: list[str], return_sequence: bool = False) -> torch.Tensor:
        tokens = _tokenize(seqs)
        with torch.no_grad():
            return self.forward(tokens, return_sequence=return_sequence)
