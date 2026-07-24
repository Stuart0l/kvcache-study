"""Minimal decoder-only transformer with a hand-written, KV-cache-aware
causal self-attention. The attention function is shared by both the
no-cache and with-cache code paths so they are guaranteed to compute the
same thing, with or without a cache.
"""
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

LayerCache = Optional[Tuple[torch.Tensor, torch.Tensor]]  # (K, V) per layer


@dataclass
class ModelConfig:
    vocab_size: int
    n_layers: int = 1
    d_model: int = 64
    n_heads: int = 4
    max_seq_len: int = 2048


class CausalSelfAttention(nn.Module):
    """Hand-written multi-head causal self-attention.

    forward(x, cache) handles both generation modes with one code path:
    - no cache: x is the full sequence so far, cache is None -> standard
      causal attention over the whole sequence.
    - with cache: x is only the newest token(s), cache holds the K/V for
      everything before it -> new K/V are computed for x, appended to the
      cache, and the new queries attend over cached+new keys/values.
    A single mask construction covers both cases: keys coming from the
    cache are always in the past (no masking needed), keys coming from the
    new chunk get a standard causal sub-mask.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def _split_heads(self, t: torch.Tensor) -> torch.Tensor:
        B, T, _ = t.shape
        return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, T, Dh)

    def forward(self, x: torch.Tensor, cache: LayerCache = None):
        B, T_new, _ = x.shape
        q = self._split_heads(self.w_q(x))
        k_new = self._split_heads(self.w_k(x))
        v_new = self._split_heads(self.w_v(x))

        if cache is not None:
            k_prev, v_prev = cache
            k_full = torch.cat([k_prev, k_new], dim=2)
            v_full = torch.cat([v_prev, v_new], dim=2)
        else:
            k_full, v_full = k_new, v_new
        new_cache = (k_full, v_full)

        T_k = k_full.shape[2]
        T_cache = T_k - T_new
        scores = q @ k_full.transpose(-2, -1) / math.sqrt(self.d_head)  # (B, H, T_new, T_k)

        # Cached keys are always causally valid; only the new-chunk block
        # needs a causal sub-mask (this covers full no-cache forward too,
        # where T_cache == 0 and the sub-mask is the whole score matrix).
        causal_submask = torch.triu(
            torch.ones(T_new, T_new, dtype=torch.bool, device=x.device), diagonal=1
        )
        mask = torch.zeros(T_new, T_k, dtype=torch.bool, device=x.device)
        mask[:, T_cache:] = causal_submask
        scores = scores.masked_fill(mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        out = attn @ v_full  # (B, H, T_new, Dh)
        out = out.transpose(1, 2).contiguous().view(B, T_new, self.n_heads * self.d_head)
        return self.w_o(out), new_cache


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )

    def forward(self, x: torch.Tensor, cache: LayerCache = None):
        attn_out, new_cache = self.attn(self.ln1(x), cache)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, new_cache


class TinyGPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(
            [Block(cfg.d_model, cfg.n_heads, cfg.d_model * 4) for _ in range(cfg.n_layers)]
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(
        self,
        idx: torch.Tensor,
        cache: Optional[List[LayerCache]] = None,
        pos_offset: int = 0,
    ):
        """idx: (B, T_new) ids of only the NEW tokens for this call.
        cache: list of per-layer (K, V), or None on the first call.
        pos_offset: absolute sequence position of idx[:, 0].
        """
        B, T_new = idx.shape
        positions = torch.arange(pos_offset, pos_offset + T_new, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(positions)[None, :, :]

        if cache is None:
            cache = [None] * len(self.blocks)
        new_cache = []
        for block, layer_cache in zip(self.blocks, cache):
            x, layer_new_cache = block(x, layer_cache)
            new_cache.append(layer_new_cache)

        x = self.ln_f(x)
        return self.head(x), new_cache


@torch.no_grad()
def generate_no_cache(
    model: TinyGPT,
    prompt_ids: torch.Tensor,
    n_new_tokens: int,
    device: str,
    record_times: bool = False,
):
    """Recomputes the full forward pass over the whole sequence at every
    step. For standard full attention, this is O(n^3) attention work across
    n generated tokens because every step recomputes the whole prefix."""
    model.eval()
    idx = prompt_ids.clone().to(device)
    step_times = [] if record_times else None

    for _ in range(n_new_tokens):
        if record_times:
            if device == "mps":
                torch.mps.synchronize()
            t0 = _now()
        logits, _ = model(idx, cache=None, pos_offset=0)
        if record_times:
            if device == "mps":
                torch.mps.synchronize()
            step_times.append(_now() - t0)
        next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        idx = torch.cat([idx, next_id], dim=1)

    return (idx, step_times) if record_times else idx


@torch.no_grad()
def generate_with_cache(
    model: TinyGPT,
    prompt_ids: torch.Tensor,
    n_new_tokens: int,
    device: str,
    record_times: bool = False,
):
    """Forwards only the newest token at each step, reusing a per-layer
    K/V cache for everything before it. Standard full attention still does
    O(n^2) attention work across n generated tokens, but avoids recomputing
    old K/V projections and earlier-token activations."""
    model.eval()
    idx = prompt_ids.clone().to(device)
    T0 = idx.shape[1]
    step_times = [] if record_times else None

    if record_times:
        if device == "mps":
            torch.mps.synchronize()
        t0 = _now()
    logits, cache = model(idx, cache=None, pos_offset=0)
    if record_times:
        if device == "mps":
            torch.mps.synchronize()
        step_times.append(_now() - t0)

    pos = T0
    next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    idx = torch.cat([idx, next_id], dim=1)

    for _ in range(n_new_tokens - 1):
        if record_times:
            if device == "mps":
                torch.mps.synchronize()
            t0 = _now()
        logits, cache = model(next_id, cache=cache, pos_offset=pos)
        if record_times:
            if device == "mps":
                torch.mps.synchronize()
            step_times.append(_now() - t0)
        pos += 1
        next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        idx = torch.cat([idx, next_id], dim=1)

    return (idx, step_times) if record_times else idx


def _now() -> float:
    import time

    return time.perf_counter()
