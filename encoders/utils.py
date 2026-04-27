import torch
import torch.nn as nn

AA_VOCAB = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWYX")}  # 20 standard AAs + X -> 0-20
PAD_IDX = 21


def _tokenize(seqs: list[str]) -> torch.Tensor:
    """Tokenize a list of AA sequences, padding to the max length in the batch."""
    tokens = [[AA_VOCAB.get(aa, PAD_IDX) for aa in seq] for seq in seqs]
    max_len = max(len(t) for t in tokens)
    padded = [t + [PAD_IDX] * (max_len - len(t)) for t in tokens]
    return torch.tensor(padded, dtype=torch.long)


def build_mlp(input_dim: int, output_dim: int, n_layers: int = 1,
              dropout: float = 0.0) -> nn.Module:
    """Build an MLP with gradually decreasing hidden dims.

    n_layers=1 returns a plain nn.Linear (backward-compatible state dict keys).
    n_layers>1 returns nn.Sequential with GELU activations between layers.
    """
    if n_layers <= 1:
        return nn.Linear(input_dim, output_dim)
    dims = [round(input_dim + (output_dim - input_dim) * i / n_layers)
            for i in range(n_layers + 1)]
    layers = []
    for i in range(n_layers):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < n_layers - 1:
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)
