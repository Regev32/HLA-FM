import torch

AA_VOCAB = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWYX")}  # 20 standard AAs + X -> 0-20
PAD_IDX = 21


def _tokenize(seqs: list[str]) -> torch.Tensor:
    """Tokenize a list of AA sequences, padding to the max length in the batch."""
    tokens = [[AA_VOCAB.get(aa, PAD_IDX) for aa in seq] for seq in seqs]
    max_len = max(len(t) for t in tokens)
    padded = [t + [PAD_IDX] * (max_len - len(t)) for t in tokens]
    return torch.tensor(padded, dtype=torch.long)
