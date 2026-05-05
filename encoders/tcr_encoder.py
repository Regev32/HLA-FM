import torch
from transformers import AutoTokenizer, AutoModel
from transformers import logging as transformers_logging

transformers_logging.set_verbosity_error()


class TCREncoder:
    """
    Wrapper around tcrBERT (wukevin/tcr-bert) for encoding TCR beta chains.

    Input format (tcrbert_input column):
        "<V_gene> <space-separated CDR3> <J_gene>"
        e.g. "TRBV28 C A S S L E W L P G N T I Y F TRBJ1-3"

    Output: (N, 768) float32 tensor — CLS token embedding.

    Download the model first:
        python encoders/download_tcrbert.py
    """

    DEFAULT_MODEL_PATH = "encoders/tcrbert"
    OUTPUT_DIM = 768

    def __init__(self, model_path: str = None, device: str = None):
        self.model_path = model_path or self.DEFAULT_MODEL_PATH
        self.device     = device or self._detect_device()
        self.tokenizer  = None
        self.model      = None

    def _detect_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load(self) -> "TCREncoder":
        """Load the model and tokenizer. Returns self for chaining."""
        print(f"Loading tcrBERT from {self.model_path} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model     = AutoModel.from_pretrained(self.model_path).to(self.device)
        self.model.eval()
        print("tcrBERT loaded.")
        return self

    def encode(self, sequences: list[str], batch_size: int = 64) -> torch.Tensor:
        """
        Encode a list of tcrbert_input strings into CLS embeddings.

        Args:
            sequences:  list of "<V_gene> <CDR3 AAs> <J_gene>" strings
            batch_size: number of sequences per forward pass

        Returns:
            Tensor of shape (N, 768)
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        all_embeddings = []

        for i in range(0, len(sequences), batch_size):
            batch  = sequences[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors = "pt",
                padding        = True,
                truncation     = True,
                max_length     = 128,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            # CLS token is the first token in BERT-style models
            cls_emb = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(cls_emb.cpu())

            print(f"Encoded {min(i + batch_size, len(sequences))}/{len(sequences)}")

        return torch.cat(all_embeddings, dim=0)

    def __repr__(self):
        status = "loaded" if self.model is not None else "not loaded"
        return f"TCREncoder(model='{self.model_path}', device='{self.device}', status={status})"


if __name__ == "__main__":
    encoder = TCREncoder()
    encoder.load()
    print(encoder)

    seqs = [
        "TRBV28 C A S S L E W L P G N T I Y F TRBJ1-3",
        "TRBV06 C A S S Y S G T G A Y E Q Y F TRBJ2-7",
        "TRBV12 C A S S S G T F Y G Y T F TRBJ1-2",
    ]

    embs = encoder.encode(seqs)
    print(f"\nEmbedding shape: {embs.shape}")  # (3, 768)
