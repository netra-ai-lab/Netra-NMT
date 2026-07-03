"""
netra_nmt.diffusion_model — non-autoregressive (masked-diffusion) EN↔KM model.

Architecturally IDENTICAL to the autoregressive ``NetraNMT``
(:mod:`netra_nmt.model`) — same encoder, same SwiGLU/LayerNorm decoder blocks,
same d_model/layers/heads/ffn, same weight tying — so a diffusion-vs-AR
comparison isolates the *decoding paradigm*, not the architecture. The only
differences:

  1. The decoder self-attention is **non-causal** (no triangular mask): a
     denoiser must see the whole partially-masked target at once.
  2. A **length predictor** head on mean-pooled encoder states, because a
     non-autoregressive decoder must choose the output length up front.
  3. One extra embedding row for the absorbing ``MASK`` state
     (``MASK_ID = vocab_size``). The output head is tied over ``vocab_size + 1``;
     ``MASK`` is simply never a gold target and is forbidden at decode time.

Reuses ``Encoder`` / ``EncoderBlock`` / ``DecoderBlock`` / ``SwiGLUFFN`` from
:mod:`netra_nmt.model` unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .model import Encoder, DecoderBlock


class NonCausalDecoder(nn.Module):
    """Same as model.Decoder but WITHOUT the causal mask (bidirectional)."""

    def __init__(self, vocab_size_with_mask, d_model, n_layers, n_heads,
                 ffn_dim, max_len=256, dropout=0.1):
        super().__init__()
        # +1 row already included in vocab_size_with_mask (for MASK_ID).
        self.embed = nn.Embedding(vocab_size_with_mask, d_model, padding_idx=1)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_ids, enc_out,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.dropout(self.embed(input_ids) + self.pos_embed(pos))
        for layer in self.layers:
            x = layer(
                x, enc_out,
                tgt_mask=None,                              # ← non-causal
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        return self.norm(x)


class DiffusionNMT(nn.Module):
    """Masked-diffusion encoder-decoder. ``MASK_ID == vocab_size``."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        enc_layers: int = 6,
        dec_layers: int = 6,
        n_heads: int = 8,
        ffn_dim: int = 2048,
        max_len: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_id = vocab_size              # absorbing state id
        self.max_len = max_len
        vmask = vocab_size + 1                  # embed/head include the MASK row

        self.encoder = Encoder(
            vocab_size, d_model, enc_layers, n_heads, ffn_dim, max_len, dropout
        )
        self.decoder = NonCausalDecoder(
            vmask, d_model, dec_layers, n_heads, ffn_dim, max_len, dropout
        )

        # Output head tied to the decoder embedding (over vocab_size + 1).
        self.lm_head = nn.Linear(d_model, vmask, bias=False)
        self.lm_head.weight = self.decoder.embed.weight

        # Length predictor: mean-pooled encoder state → length class in [1, max_len].
        self.length_head = nn.Linear(d_model, max_len)

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

    # ── building blocks used by both training and decoding ──────────────────
    def encode(self, input_ids, src_key_padding_mask=None):
        return self.encoder(input_ids, src_key_padding_mask=src_key_padding_mask)

    def _masked_mean(self, enc_out, src_key_padding_mask):
        if src_key_padding_mask is None:
            return enc_out.mean(dim=1)
        keep = (~src_key_padding_mask).unsqueeze(-1).to(enc_out.dtype)   # (B,S,1)
        summed = (enc_out * keep).sum(dim=1)
        counts = keep.sum(dim=1).clamp_min(1.0)
        return summed / counts

    def predict_length_logits(self, enc_out, src_key_padding_mask=None):
        return self.length_head(self._masked_mean(enc_out, src_key_padding_mask))  # (B, max_len)

    def denoise(self, tgt_ids, enc_out, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        dec_out = self.decoder(
            tgt_ids, enc_out,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.lm_head(dec_out)          # (B, T, vocab_size + 1)

    # ── training forward ────────────────────────────────────────────────────
    def forward(
        self,
        input_ids,
        corrupted_tgt_ids,
        src_key_padding_mask=None,
        tgt_key_padding_mask=None,
    ):
        """Returns (token_logits (B,T,vocab+1), length_logits (B,max_len))."""
        enc_out = self.encode(input_ids, src_key_padding_mask=src_key_padding_mask)
        token_logits = self.denoise(
            corrupted_tgt_ids, enc_out,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        length_logits = self.predict_length_logits(enc_out, src_key_padding_mask)
        return token_logits, length_logits
