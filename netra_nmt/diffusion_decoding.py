"""
netra_nmt.diffusion_decoding — iterative parallel unmasking (Mask-Predict).

Non-autoregressive decode for :class:`netra_nmt.diffusion_model.DiffusionNMT`:

  1. Encode the source once.
  2. Predict target length L from the pooled encoder state.
  3. Start with an all-``MASK`` sequence of length L and run ``steps`` denoising
     passes. Each pass predicts every position in parallel; we keep the most
     confident tokens and re-mask the least confident ones, with the number of
     re-masked positions decaying linearly to 0. After the last pass nothing is
     masked.

Total cost is ``steps`` forward passes regardless of output length — the CPU
latency lever vs the AR model's one step per token.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def diffusion_decode(
    model,
    src_ids: torch.Tensor,
    src_key_padding_mask: torch.Tensor | None = None,
    *,
    steps: int = 10,
    pad_id: int = 0,
    eos_id: int = 3,
    length: torch.Tensor | None = None,
) -> list[list[int]]:
    """Return a list (len B) of decoded token-id lists (specials stripped).

    ``model`` may be compiled or raw; pass the unwrapped module to avoid
    recompiles on the varying target length. If ``length`` is given (B,), it
    overrides the model's length predictor (useful for oracle-length ablations).
    """
    model.eval()
    device = src_ids.device
    B = src_ids.size(0)
    mask_id = model.mask_id
    max_len = model.max_len

    enc_out = model.encode(src_ids, src_key_padding_mask=src_key_padding_mask)

    if length is None:
        len_logits = model.predict_length_logits(enc_out, src_key_padding_mask)
        length = len_logits.argmax(dim=-1) + 1        # class idx → length in [1, max_len]
    length = length.clamp(1, max_len).to(device)
    Lmax = int(length.max().item())

    pos = torch.arange(Lmax, device=device).unsqueeze(0)        # (1, Lmax)
    pad_mask = pos >= length.unsqueeze(1)                        # (B, Lmax) True = pad
    tokens = torch.full((B, Lmax), mask_id, dtype=torch.long, device=device)
    tokens = tokens.masked_fill(pad_mask, pad_id)
    conf = torch.zeros(B, Lmax, device=device)

    for k in range(steps):
        cur_mask = tokens.eq(mask_id)
        logits = model.denoise(
            tokens, enc_out,
            tgt_key_padding_mask=pad_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        logits[..., mask_id] = float("-inf")           # never emit the MASK symbol
        probs = logits.softmax(dim=-1)
        p, pred = probs.max(dim=-1)                     # (B, Lmax)

        # Commit predictions at currently-masked positions.
        tokens = torch.where(cur_mask, pred, tokens)
        conf = torch.where(cur_mask, p, conf)

        if k < steps - 1:
            # Number of positions to re-mask for the next pass, per example.
            num_mask = (length.float() * (1.0 - (k + 1) / steps)).long().clamp(min=0)
            conf_rank = conf.masked_fill(pad_mask, float("inf"))   # pad never re-masked
            order = conf_rank.argsort(dim=1)                       # asc by confidence
            ranks = torch.empty_like(order)
            ranks.scatter_(
                1, order,
                torch.arange(Lmax, device=device).unsqueeze(0).expand(B, Lmax),
            )
            remask = (ranks < num_mask.unsqueeze(1)) & (~pad_mask)
            tokens = tokens.masked_fill(remask, mask_id)

    results: list[list[int]] = []
    for b in range(B):
        ids = tokens[b, : int(length[b].item())].tolist()
        if eos_id in ids:
            ids = ids[: ids.index(eos_id)]
        ids = [t for t in ids if t != pad_id and t != mask_id]
        results.append(ids)
    return results
