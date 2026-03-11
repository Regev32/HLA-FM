import torch
from transformers import AutoTokenizer, AutoModel
from transformers import logging as transformers_logging

transformers_logging.set_verbosity_error()


class PeptideEncoder:
    DEFAULT_MODEL_PATH = "encoders/esm2_650M"

    def __init__(self, model_path: str = None, device: str = None):
        self.model_path = model_path or self.DEFAULT_MODEL_PATH
        self.device = device or self._detect_device()
        self.tokenizer = None
        self.model = None

    def _detect_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load(self) -> "PeptideEncoder":
        """Load the model and tokenizer. Returns self for chaining."""
        print(f"Loading model on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModel.from_pretrained(self.model_path).to(self.device)
        self.model.eval()
        print("Model loaded.")
        return self

    def _mean_pool(self, token_embs: torch.Tensor, attention_mask: torch.Tensor, batch: list[str]) -> torch.Tensor:
        """Mean pool token embeddings, excluding BOS/EOS tokens."""
        mask = attention_mask.clone()
        mask[:, 0] = 0  # BOS
        for j in range(len(batch)):
            seq_len = mask[j].sum().item()
            mask[j, int(seq_len)] = 0  # EOS

        mask = mask.unsqueeze(-1).float()
        return (token_embs * mask).sum(dim=1) / mask.sum(dim=1)

    def encode(self, peptides: list[str], batch_size: int = 32) -> torch.Tensor:
        """
        Encode a list of peptide sequences into embeddings.
        Returns a tensor of shape (N, 1280).
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        all_embeddings = []

        for i in range(0, len(peptides), batch_size):
            batch = peptides[i:i + batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            embeddings = self._mean_pool(outputs.last_hidden_state, inputs["attention_mask"], batch)
            all_embeddings.append(embeddings.cpu())

            print(f"Encoded {min(i + batch_size, len(peptides))}/{len(peptides)}")

        return torch.cat(all_embeddings, dim=0)

    def __repr__(self):
        status = "loaded" if self.model is not None else "not loaded"
        return f"PeptideEncoder(model='{self.model_path}', device='{self.device}', status={status})"


if __name__ == "__main__":
    encoder = PeptideEncoder()
    encoder.load()

    print(encoder)  # PeptideEncoder(model='...', device='mps', status=loaded)

    peptides = [
        "ACDEFGHIKL",
        "MNPQRSTVWY",
        "GILGFVFTL",
    ]

    embeddings = encoder.encode(peptides)
    print(f"\nEmbedding shape: {embeddings.shape}")  # (3, 1280)
    print(f"First embedding: {embeddings[0]}")
    print(embeddings)