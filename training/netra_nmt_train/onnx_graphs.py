"""The two graphs the package runs, written as thin PyTorch wrappers around a
trained :class:`~netra_nmt_train.model.NetraNMT` (same weights, same maths):

``Encoder``      tokens, mask -> cross_k, cross_v
                 Runs the 12-layer encoder and projects its output into every
                 decoder layer's cross-attention keys and values once, so the
                 decoder step never sees the encoder states.

``DecoderStep``  one token per sequence + the self-attention KV cache so far
                 -> logits for the next token + the cache with this step added.
                 The RoPE position is the cache length, exactly like
                 ``pos_offset=step`` in ``NetraNMT.decode_step``.

Shapes (L = decoder layers, H = heads, hd = head dim):
    tokens      (B, S) int64          mask        (B, S) int64, 1 = real token
    cross_k/v   (L, B, H, S, hd)      cross_mask  (B, S) float, 0 or -inf
    step_ids    (B, 1) int64          past_k/v    (L, B, H, P, hd)
    logits      (B, vocab)            present_k/v (L, B, H, P+1, hd)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import NetraNMT, apply_rotary


def _rope(model: NetraNMT, start, length: int | torch.Tensor):
    """cos/sin (1, 1, T, hd) for positions [start, start + length), traceable."""
    inv_freq = model.rope.inv_freq
    t = (torch.arange(length, device=inv_freq.device) + start).to(inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos()[None, None], emb.sin()[None, None]


class Encoder(nn.Module):
    def __init__(self, model: NetraNMT):
        super().__init__()
        self.model = model.eval()

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor):
        m = self.model
        x = m.embed(tokens)
        pad = torch.zeros(mask.shape, dtype=x.dtype).masked_fill(mask == 0, float("-inf"))
        attn_mask = pad[:, None, None, :]
        cos, sin = _rope(m, 0, tokens.shape[1])
        for layer in m.encoder.layers:
            att = layer.self_attn
            h = layer.norm1(x)
            q = apply_rotary(att._split(att.q_proj(h)), cos, sin)
            k = apply_rotary(att._split(att.k_proj(h)), cos, sin)
            v = att._split(att.v_proj(h))
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            x = x + att.out_proj(att._merge(o))
            x = x + layer.ffn(layer.norm2(x))
        enc = m.encoder.norm(x)
        ks, vs = [], []
        for layer in m.decoder.layers:
            ca = layer.cross_attn
            ks.append(ca._split(ca.k_proj(enc)))
            vs.append(ca._split(ca.v_proj(enc)))
        return torch.stack(ks), torch.stack(vs)


class DecoderStep(nn.Module):
    def __init__(self, model: NetraNMT):
        super().__init__()
        self.model = model.eval()

    def forward(self, step_ids, past_k, past_v, cross_k, cross_v, cross_mask):
        m = self.model
        x = m.embed(step_ids)
        cos, sin = _rope(m, past_k.shape[3], 1)
        mem_mask = cross_mask[:, None, None, :]
        new_k, new_v = [], []
        for i, layer in enumerate(m.decoder.layers):
            sa = layer.self_attn
            h = layer.norm1(x)
            q = apply_rotary(sa._split(sa.q_proj(h)), cos, sin)
            k = torch.cat([past_k[i], apply_rotary(sa._split(sa.k_proj(h)), cos, sin)], dim=2)
            v = torch.cat([past_v[i], sa._split(sa.v_proj(h))], dim=2)
            new_k.append(k)
            new_v.append(v)
            x = x + sa.out_proj(sa._merge(F.scaled_dot_product_attention(q, k, v)))

            ca = layer.cross_attn
            q = ca._split(ca.q_proj(layer.norm2(x)))
            o = F.scaled_dot_product_attention(q, cross_k[i], cross_v[i], attn_mask=mem_mask)
            x = x + ca.out_proj(ca._merge(o))
            x = x + layer.ffn(layer.norm3(x))
        logits = m.lm_head(m.decoder.norm(x))[:, -1, :]
        return logits, torch.stack(new_k), torch.stack(new_v)
