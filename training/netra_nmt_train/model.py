"""
netra_nmt.model — custom encoder-decoder translation model (trained from scratch).

A compact NLLB-style seq2seq transformer, modernised with:
  * bidirectional encoder (no causal mask)
  * autoregressive decoder with causal self-attention + cross-attention
  * RoPE (rotary) positional encoding on self-attention — no learned position
    table, no fixed length cap
  * RMSNorm, Pre-Norm placement
  * SwiGLU feed-forward blocks
  * a single embedding table shared across encoder input, decoder input and the
    (tied) output projection
  * a KV cache for O(T) incremental decoding

Attention is implemented on top of ``F.scaled_dot_product_attention`` so training
picks up the FlashAttention-2 backend on CUDA automatically.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root-mean-square layer norm (no mean subtraction, no bias)."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


# ---------------------------------------------------------------------------
# Rotary positional embedding (RoPE)
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """Precomputes cos/sin tables for rotary position embeddings."""

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cached_len = 0
        self._cos: torch.Tensor | None = None
        self._sin: torch.Tensor | None = None

    def _ensure(self, seq_len: int, device, dtype):
        if (
            self._cos is None
            or seq_len > self._cached_len
            or self._cos.device != device
        ):
            self._cached_len = max(seq_len, self._cached_len)
            t = torch.arange(self._cached_len, device=device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)            # (L, hd/2)
            emb = torch.cat([freqs, freqs], dim=-1)          # (L, hd)
            self._cos = emb.cos()[None, None, :, :]          # (1,1,L,hd)
            self._sin = emb.sin()[None, None, :, :]

    def forward(self, start: int, length: int, device, dtype):
        """Return cos/sin for absolute positions [start, start+length)."""
        self._ensure(start + length, device, dtype)
        cos = self._cos[..., start:start + length, :].to(dtype)
        sin = self._sin[..., start:start + length, :].to(dtype)
        return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, T, hd); cos/sin: (1, 1, T, hd)."""
    return x * cos + _rotate_half(x) * sin


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention over SDPA.

    Handles three roles via flags:
      * encoder self-attention  (bidirectional, RoPE, no cache)
      * decoder self-attention  (causal, RoPE, KV cache)
      * decoder cross-attention (no RoPE, encoder K/V cached once)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        self_attention: bool = True,
        use_rope: bool = True,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.self_attention = self_attention
        self.use_rope = use_rope
        self.dropout = dropout

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B,H,T,hd)

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        B, H, T, hd = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * hd)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        is_causal: bool = False,
        rope: RotaryEmbedding | None = None,
        pos_offset: int = 0,
        cache: dict | None = None,
        cache_key: str = "self",
    ) -> torch.Tensor:
        B, Tq, _ = query.shape
        q = self._split(self.q_proj(query))                       # (B,H,Tq,hd)

        if self.self_attention:
            k = self._split(self.k_proj(key_value))
            v = self._split(self.v_proj(key_value))
            if self.use_rope and rope is not None:
                cos, sin = rope(pos_offset, Tq, query.device, q.dtype)
                q = apply_rotary(q, cos, sin)
                k = apply_rotary(k, cos, sin)
            if cache is not None:
                past = cache.get(cache_key)
                if past is not None:
                    pk, pv = past
                    k = torch.cat([pk, k], dim=2)
                    v = torch.cat([pv, v], dim=2)
                cache[cache_key] = (k, v)
        else:
            # cross-attention: encoder K/V are position-independent → cache once
            cached = cache.get(cache_key) if cache is not None else None
            if cached is not None:
                k, v = cached
            else:
                k = self._split(self.k_proj(key_value))
                v = self._split(self.v_proj(key_value))
                if cache is not None:
                    cache[cache_key] = (k, v)

        Tk = k.size(2)
        attn_mask = self._build_mask(key_padding_mask, is_causal, Tq, Tk, q.device, q.dtype)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.out_proj(self._merge(out))

    @staticmethod
    def _build_mask(key_padding_mask, is_causal, Tq, Tk, device, dtype):
        mask = None
        if is_causal and Tq > 1:
            # query i (absolute pos Tk-Tq+i) may attend to keys 0..(Tk-Tq)+i
            full = torch.full((Tq, Tk), float("-inf"), device=device, dtype=dtype)
            full = torch.triu(full, diagonal=(Tk - Tq) + 1)
            mask = full[None, None, :, :]
        if key_padding_mask is not None:
            kpm = torch.zeros_like(key_padding_mask, dtype=dtype)
            kpm = kpm.masked_fill(key_padding_mask, float("-inf"))  # (B,Tk)
            kpm = kpm[:, None, None, :]                             # (B,1,1,Tk)
            mask = kpm if mask is None else mask + kpm
        return mask


# ---------------------------------------------------------------------------
# Feed-forward
# ---------------------------------------------------------------------------

class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward: project to ffn_dim*2, gate with SiLU, project back."""

    def __init__(self, d_model: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, ffn_dim * 2, bias=False)
        self.w2 = nn.Linear(ffn_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.w1(x).chunk(2, dim=-1)
        return self.dropout(self.w2(F.silu(gate) * value))


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class EncoderBlock(nn.Module):
    """Bidirectional Pre-Norm encoder block (RoPE self-attention)."""

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(
            d_model, n_heads, dropout, self_attention=True, use_rope=True
        )
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, rope, src_key_padding_mask=None):
        residual = x
        x = self.norm1(x)
        x = self.self_attn(
            x, x, key_padding_mask=src_key_padding_mask, is_causal=False, rope=rope
        )
        x = self.dropout(x) + residual

        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        return x + residual


class Encoder(nn.Module):
    """Operates on embeddings (the shared embedding table lives on the model)."""

    def __init__(self, rope, d_model, n_layers, n_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.rope = rope
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, n_heads, ffn_dim, dropout) for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)

    def forward(self, x, src_key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, self.rope, src_key_padding_mask=src_key_padding_mask)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class DecoderBlock(nn.Module):
    """Pre-Norm decoder block: causal RoPE self-attn → cross-attn → SwiGLU FFN."""

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(
            d_model, n_heads, dropout, self_attention=True, use_rope=True
        )
        self.cross_attn = MultiHeadAttention(
            d_model, n_heads, dropout, self_attention=False, use_rope=False
        )
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.norm3 = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x,
        enc_out,
        rope,
        pos_offset=0,
        memory_key_padding_mask=None,
        layer_cache=None,
    ):
        # Causal self-attention (RoPE + optional KV cache)
        residual = x
        x = self.norm1(x)
        x = self.self_attn(
            x, x, is_causal=True, rope=rope, pos_offset=pos_offset,
            cache=layer_cache, cache_key="self",
        )
        x = self.dropout(x) + residual

        # Cross-attention over encoder output
        residual = x
        x = self.norm2(x)
        x = self.cross_attn(
            x, enc_out, key_padding_mask=memory_key_padding_mask,
            cache=layer_cache, cache_key="cross",
        )
        x = self.dropout(x) + residual

        # FFN
        residual = x
        x = self.norm3(x)
        x = self.ffn(x)
        return x + residual


class Decoder(nn.Module):
    """Operates on embeddings (the shared embedding table lives on the model)."""

    def __init__(self, rope, d_model, n_layers, n_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.rope = rope
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, n_heads, ffn_dim, dropout) for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)

    def forward(
        self,
        x,
        enc_out,
        memory_key_padding_mask=None,
        cache: list | None = None,
        pos_offset: int = 0,
    ):
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            x = layer(
                x, enc_out, self.rope, pos_offset=pos_offset,
                memory_key_padding_mask=memory_key_padding_mask,
                layer_cache=layer_cache,
            )
        return self.norm(x)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class NetraNMT(nn.Module):
    """
    Encoder-decoder translation model.

    Design notes:
      1. Bidirectional encoder, causal decoder.
      2. A single embedding table is shared across encoder input, decoder input
         and the tied ``lm_head`` (one 32k×d_model table for a shared vocab).
      3. RoPE on self-attention (encoder + decoder); cross-attention has no
         positional rotation. No learned position table, no length cap.
      4. RMSNorm, Pre-Norm placement.
      5. ``padding_idx`` matches ``pad_id`` (default 0).
      6. A per-layer KV cache enables O(T) incremental decoding.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        enc_layers: int = 12,
        dec_layers: int = 2,
        n_heads: int = 8,
        ffn_dim: int = 2048,
        max_len: int = 256,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.max_len = max_len

        # One embedding table, registered once on the model and shared by the
        # encoder, decoder and (tied) output head. Keeping it off the sub-modules
        # avoids duplicate state_dict keys pointing at shared storage.
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.emb_dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(d_model // n_heads)

        self.encoder = Encoder(self.rope, d_model, enc_layers, n_heads, ffn_dim, dropout)
        self.decoder = Decoder(self.rope, d_model, dec_layers, n_heads, ffn_dim, dropout)

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # tie output projection to embeddings

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Teacher-forced forward. Returns logits (B, T, vocab_size)."""
        enc_out = self.encode(input_ids, src_key_padding_mask=src_key_padding_mask)
        dec_x = self.emb_dropout(self.embed(decoder_input_ids))
        dec_out = self.decoder(
            dec_x,
            enc_out,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.lm_head(dec_out)

    # -----------------------------------------------------------------------
    # Incremental-decoding helpers (used by netra_nmt.decoding)
    # -----------------------------------------------------------------------

    def encode(self, input_ids, src_key_padding_mask=None):
        x = self.emb_dropout(self.embed(input_ids))
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)

    def new_cache(self) -> list[dict]:
        """Fresh per-layer KV cache for a decoding run."""
        return [dict() for _ in self.decoder.layers]

    def decode_step(
        self,
        step_ids: torch.Tensor,
        enc_out: torch.Tensor,
        cache: list,
        pos_offset: int,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        One incremental decode of ``step_ids`` (B, Tq) — usually Tq=1.
        Updates ``cache`` in place. Returns logits (B, Tq, vocab_size).
        """
        x = self.emb_dropout(self.embed(step_ids))
        dec_out = self.decoder(
            x,
            enc_out,
            memory_key_padding_mask=memory_key_padding_mask,
            cache=cache,
            pos_offset=pos_offset,
        )
        return self.lm_head(dec_out)

    # -----------------------------------------------------------------------
    # Greedy inference
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        bos_token_id: int,
        eos_token_id: int,
        max_new_tokens: int = 128,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Greedy autoregressive decoding with a KV cache."""
        self.eval()
        device = input_ids.device
        B = input_ids.size(0)

        enc_out = self.encode(input_ids, src_key_padding_mask=src_key_padding_mask)
        cache = self.new_cache()

        next_ids = torch.full((B, 1), bos_token_id, dtype=torch.long, device=device)
        generated = [next_ids]
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for step in range(max_new_tokens):
            logits = self.decode_step(
                next_ids, enc_out, cache, pos_offset=step,
                memory_key_padding_mask=src_key_padding_mask,
            )
            next_ids = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (B,1)
            generated.append(next_ids)
            finished |= (next_ids.squeeze(-1) == eos_token_id)
            if finished.all():
                break

        return torch.cat(generated, dim=1)
