"""
Modern MHC encoder — drop-in replacement for MHCAttentionEncoder.

Innovations vs the classic encoder:
  RMSNorm   — replaces LayerNorm (no mean-centering, no bias, cheaper)
  SwiGLU    — replaces GELU FFN  (x * silu(gate), ~2/3 hidden dim to match params)
  RoPE      — replaces learned positional embeddings (applied to Q & K per layer)
  MoE       — replaces dense FFN with a mixture of SwiGLU experts (optional)

Interface is identical to MHCAttentionEncoder so all training code is reusable:
  forward(x, return_sequence=False) → (B, output_dim)  or  (B, L, embed_dim)
  encode(seqs, ...)
  self.embed_dim, self.proj, self.transformer.layers
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from encoders.utils import PAD_IDX, _tokenize, build_mlp

MAX_SEQ = 512   # same upper bound as classic encoder

DEFAULTS = dict(
    vocab_size    = 22,
    embed_dim     = 128,
    nhead         = 8,
    num_layers    = 4,
    output_dim    = 256,
    dropout       = 0.1,
    proj_layers   = 1,
    use_moe       = True,
    num_experts   = 4,
    top_k         = 2,
    ffn_hidden_dim = None,   # if None, defaults to int(2/3 * 4 * embed_dim)
)


# ── RMSNorm ───────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


# ── RoPE helpers ──────────────────────────────────────────────────────────────

def _rope_freqs(head_dim: int, max_seq: int, theta: float = 10_000.0) -> tuple:
    """Precompute (cos, sin) of shape (max_seq, head_dim)."""
    half  = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half).float() / half))   # (half,)
    t     = torch.arange(max_seq).float()                               # (max_seq,)
    freqs = torch.outer(t, freqs)                                       # (max_seq, half)
    freqs = freqs.repeat(1, 2)                                          # (max_seq, head_dim)
    return freqs.cos(), freqs.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, nhead, L, head_dim) · cos/sin: (L, head_dim)"""
    c = cos.unsqueeze(0).unsqueeze(0)   # (1, 1, L, D)
    s = sin.unsqueeze(0).unsqueeze(0)
    return x * c + _rotate_half(x) * s


# ── RoPE Self-Attention ───────────────────────────────────────────────────────

class RoPESelfAttention(nn.Module):
    def __init__(self, embed_dim: int, nhead: int, dropout: float):
        super().__init__()
        assert embed_dim % nhead == 0
        self.nhead    = nhead
        self.head_dim = embed_dim // nhead
        self.scale    = self.head_dim ** -0.5

        self.qkv  = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out  = nn.Linear(embed_dim, embed_dim,     bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor,
                rope_cos: torch.Tensor, rope_sin: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.nhead, self.head_dim)
        q, k, v = qkv.unbind(2)                  # each (B, L, H, Dh)
        q = q.transpose(1, 2)                     # (B, H, L, Dh)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = _apply_rope(q, rope_cos[:L], rope_sin[:L])
        k = _apply_rope(k, rope_cos[:L], rope_sin[:L])

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, L, L)
        if key_padding_mask is not None:
            attn = attn.masked_fill(
                key_padding_mask[:, None, None, :], float("-inf")
            )
        attn = self.drop(attn.softmax(dim=-1))

        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.out(out)


# ── SwiGLU FFN ───────────────────────────────────────────────────────────────

class SwiGLU(nn.Module):
    """hidden_dim ≈ 2/3 * 4 * embed_dim keeps parameters ≈ standard GELU FFN."""
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.gate = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.up   = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, embed_dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


# ── Mixture of Experts ────────────────────────────────────────────────────────

class MoELayer(nn.Module):
    """
    Token-choice MoE: each token picks its top-k experts.
    aux_loss (load-balancing) is stored as an attribute after each forward;
    the training loop can add it to the main loss if desired.
    """
    def __init__(self, embed_dim: int, num_experts: int, top_k: int,
                 hidden_dim: int, dropout: float):
        super().__init__()
        assert top_k <= num_experts
        self.num_experts = num_experts
        self.top_k       = top_k
        self.router      = nn.Linear(embed_dim, num_experts, bias=False)
        self.experts     = nn.ModuleList([
            SwiGLU(embed_dim, hidden_dim, dropout) for _ in range(num_experts)
        ])
        self.aux_loss = 0.0   # updated each forward, for optional logging

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        flat    = x.reshape(-1, D)                         # (N, D)  N = B*L
        N       = flat.shape[0]

        router_logits = self.router(flat)                  # (N, E)
        scores        = router_logits.softmax(dim=-1)      # (N, E)
        top_vals, top_idx = scores.topk(self.top_k, dim=-1)
        top_vals = top_vals / top_vals.sum(dim=-1, keepdim=True)  # renormalise

        # Load-balancing auxiliary loss (switch-transformer style)
        # aux = num_experts * Σ_e (fraction of tokens → e) * (mean gate prob for e)
        one_hot = torch.zeros(N, self.num_experts, device=x.device)
        one_hot.scatter_(1, top_idx[:, :1], 1.0)          # top-1 assignment
        f_i = one_hot.mean(0)                              # (E,)
        p_i = scores.mean(0)                               # (E,)
        self.aux_loss = float((self.num_experts * (f_i * p_i).sum()).item())

        out = torch.zeros_like(flat)
        for k in range(self.top_k):
            ids    = top_idx[:, k]        # (N,)
            weight = top_vals[:, k : k+1] # (N, 1)
            for e in range(self.num_experts):
                mask = ids == e
                if mask.any():
                    out[mask] += weight[mask] * self.experts[e](flat[mask])

        return out.reshape(B, L, D)


# ── Modern Transformer Layer ──────────────────────────────────────────────────

class ModernTransformerLayer(nn.Module):
    """Pre-norm layer: RMSNorm → sublayer → residual."""
    def __init__(self, embed_dim: int, nhead: int, ffn_hidden_dim: int, dropout: float,
                 use_moe: bool, num_experts: int, top_k: int):
        super().__init__()
        self.norm1 = RMSNorm(embed_dim)
        self.attn  = RoPESelfAttention(embed_dim, nhead, dropout)
        self.norm2 = RMSNorm(embed_dim)
        self.ffn   = (MoELayer(embed_dim, num_experts, top_k, ffn_hidden_dim, dropout)
                      if use_moe else SwiGLU(embed_dim, ffn_hidden_dim, dropout))

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor,
                rope_cos: torch.Tensor, rope_sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), key_padding_mask, rope_cos, rope_sin)
        x = x + self.ffn(self.norm2(x))
        return x


# ── Param-matching utility ────────────────────────────────────────────────────

def find_matching_hidden_dim(target_params: int, base_kwargs: dict) -> int:
    """
    Binary-search for the ffn_hidden_dim that makes ModernMHCEncoder's parameter
    count as close as possible to target_params.
    base_kwargs must NOT include ffn_hidden_dim.
    """
    def count(h):
        return sum(p.numel() for p in
                   ModernMHCEncoder(**base_kwargs, ffn_hidden_dim=h).parameters())

    lo, hi = 1, 8192
    while lo < hi:
        mid = (lo + hi) // 2
        if count(mid) < target_params:
            lo = mid + 1
        else:
            hi = mid

    # Pick whichever of lo-1, lo is closer
    best = lo
    if lo > 1 and abs(count(lo - 1) - target_params) < abs(count(lo) - target_params):
        best = lo - 1

    actual = count(best)
    print(f"  [param match] ffn_hidden_dim={best}  "
          f"→ modern={actual:,}  target={target_params:,}  "
          f"diff={actual - target_params:+d}")
    return best


# ── Layers container (mirrors nn.TransformerEncoder.layers API) ───────────────

class _TransformerWrapper(nn.Module):
    def __init__(self, layers: list):
        super().__init__()
        self.layers = nn.ModuleList(layers)


# ── Modern MHC Encoder ────────────────────────────────────────────────────────

class ModernMHCEncoder(nn.Module):
    def __init__(self, **kwargs):
        cfg = {**DEFAULTS, **kwargs}
        vocab_size  = cfg["vocab_size"]
        embed_dim   = cfg["embed_dim"]
        nhead       = cfg["nhead"]
        num_layers  = cfg["num_layers"]
        output_dim  = cfg["output_dim"]
        dropout     = cfg["dropout"]
        proj_layers = cfg["proj_layers"]
        use_moe       = cfg["use_moe"]
        num_experts   = cfg["num_experts"]
        top_k         = cfg["top_k"]
        ffn_hidden_dim = cfg["ffn_hidden_dim"]

        super().__init__()
        self.embed_dim = embed_dim

        # ffn_hidden_dim can be set explicitly for parameter-matching experiments;
        # default ≈ 2/3 * 4*dim (SwiGLU convention that matches classic FFN params)
        if ffn_hidden_dim is None:
            ffn_hidden_dim = int(2 / 3 * 4 * embed_dim)

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Precomputed RoPE frequencies — not parameters, just buffers
        cos, sin = _rope_freqs(embed_dim // nhead, MAX_SEQ)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

        self.transformer = _TransformerWrapper([
            ModernTransformerLayer(embed_dim, nhead, ffn_hidden_dim, dropout,
                                   use_moe, num_experts, top_k)
            for _ in range(num_layers)
        ])
        self.norm_final = RMSNorm(embed_dim)
        self.proj       = build_mlp(embed_dim, output_dim, proj_layers, dropout)

    def forward(self, x: torch.Tensor, return_sequence: bool = False) -> torch.Tensor:
        h   = self.embedding(x)                             # (B, L, D)
        cls = self.cls_token.expand(h.size(0), -1, -1)
        h   = torch.cat([cls, h], dim=1)                    # (B, L+1, D)

        pad_mask = (x == PAD_IDX)                           # (B, L)
        cls_mask = torch.zeros(x.size(0), 1, dtype=torch.bool, device=x.device)
        pad_mask = torch.cat([cls_mask, pad_mask], dim=1)   # (B, L+1)

        for layer in self.transformer.layers:
            h = layer(h, pad_mask, self.rope_cos, self.rope_sin)

        h = self.norm_final(h)

        if return_sequence:
            return h[:, 1:, :]                              # (B, L, D) — AA positions

        return self.proj(h[:, 0, :])                        # (B, output_dim) — CLS

    def encode(self, seqs: list[str], return_sequence: bool = False) -> torch.Tensor:
        tokens = _tokenize(seqs)
        with torch.no_grad():
            return self.forward(tokens, return_sequence=return_sequence)
