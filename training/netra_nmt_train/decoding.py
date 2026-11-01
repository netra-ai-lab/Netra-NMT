"""
netra_nmt.decoding — decoding strategies (greedy, beam search, nucleus sampling).

Each function takes an encoded source and returns a list of generated token ids
(including the leading BOS). All strategies use the model's per-layer KV cache
for O(T) incremental decoding (see :meth:`NetraNMT.decode_step`).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import NetraNMT


def _reorder_cache(cache: list, index: torch.Tensor) -> None:
    """Reorder every cached K/V tensor along the batch dim (in place)."""
    for layer in cache:
        for key, (k, v) in layer.items():
            layer[key] = (k.index_select(0, index), v.index_select(0, index))


@torch.no_grad()
def greedy_decode(
    model: NetraNMT,
    input_ids: torch.Tensor,
    src_mask: torch.Tensor,
    bos_id: int,
    eos_id: int,
    max_new_tokens: int,
) -> list[int]:
    """Standard greedy decoding. Fastest, deterministic."""
    gen = model.generate(
        input_ids,
        bos_token_id=bos_id,
        eos_token_id=eos_id,
        max_new_tokens=max_new_tokens,
        src_key_padding_mask=src_mask,
    )
    return gen[0].tolist()


@torch.no_grad()
def beam_search(
    model: NetraNMT,
    input_ids: torch.Tensor,
    src_mask: torch.Tensor,
    bos_id: int,
    eos_id: int,
    max_new_tokens: int,
    beam_size: int = 5,
    length_penalty: float = 0.6,
) -> list[int]:
    """
    Beam search with a KV cache. Beams live in the batch dimension; the cache is
    reordered by parent beam each step. length_penalty follows the GNMT form
    ``((5 + len) / 6) ** length_penalty``.
    """
    device = input_ids.device

    enc_out = model.encode(input_ids, src_key_padding_mask=src_mask)   # (1, S, D)
    enc_out = enc_out.expand(beam_size, -1, -1).contiguous()
    mem_mask = (
        src_mask.expand(beam_size, -1).contiguous() if src_mask is not None else None
    )

    cache = model.new_cache()
    beam_scores = torch.full((beam_size,), float("-inf"), device=device)
    beam_scores[0] = 0.0                                              # only beam 0 active at first
    seqs = torch.full((beam_size, 1), bos_id, dtype=torch.long, device=device)
    next_ids = seqs

    def _norm(score: float, length: int) -> float:
        return score / (((5 + length) / 6) ** length_penalty)

    finished: list[tuple[float, list[int]]] = []

    for step in range(max_new_tokens):
        logits = model.decode_step(
            next_ids, enc_out, cache, pos_offset=step, memory_key_padding_mask=mem_mask
        )
        log_probs = F.log_softmax(logits[:, -1, :], dim=-1)          # (beam, V)
        V = log_probs.size(-1)

        cand = (beam_scores.unsqueeze(1) + log_probs).view(-1)       # (beam*V,)
        top_scores, top_idx = cand.topk(beam_size)
        parent = top_idx // V
        token = top_idx % V

        _reorder_cache(cache, parent)
        seqs = torch.cat([seqs.index_select(0, parent), token.unsqueeze(1)], dim=1)
        beam_scores = top_scores
        next_ids = token.unsqueeze(1)

        for b in range(beam_size):
            if token[b].item() == eos_id and beam_scores[b].item() > float("-inf"):
                finished.append((_norm(beam_scores[b].item(), seqs.size(1)), seqs[b].tolist()))
                beam_scores[b] = float("-inf")

        if beam_scores.max().item() == float("-inf"):
            break
        # Exact early stop (same rule as netra_nmt.inference.nmt): live scores only
        # fall, so their best reachable normalised score is score / penalty(max length).
        if finished and max(c[0] for c in finished) >= _norm(beam_scores.max().item(), 1 + max_new_tokens):
            break

    # Fold in any beams still running at the length cap.
    for b in range(beam_size):
        if beam_scores[b].item() > float("-inf"):
            finished.append((_norm(beam_scores[b].item(), seqs.size(1)), seqs[b].tolist()))

    best = max(finished, key=lambda c: c[0])
    return best[1]


@torch.no_grad()
def sample_decode(
    model: NetraNMT,
    input_ids: torch.Tensor,
    src_mask: torch.Tensor,
    bos_id: int,
    eos_id: int,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 0.95,
) -> list[int]:
    """Nucleus (top-p) sampling with temperature, using a KV cache."""
    device = input_ids.device

    enc_out = model.encode(input_ids, src_key_padding_mask=src_mask)
    cache = model.new_cache()

    generated = [bos_id]
    next_ids = torch.full((1, 1), bos_id, dtype=torch.long, device=device)

    for step in range(max_new_tokens):
        logits = model.decode_step(
            next_ids, enc_out, cache, pos_offset=step, memory_key_padding_mask=src_mask
        )
        logits = logits[0, -1, :] / temperature                      # (V,)

        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum_probs > top_p
        remove[0] = False                                            # always keep the top token
        sorted_logits[remove] = float("-inf")

        filtered = torch.full_like(logits, float("-inf"))
        filtered[sorted_idx] = sorted_logits
        next_token = torch.multinomial(F.softmax(filtered, dim=-1), 1).item()

        generated.append(next_token)
        next_ids = torch.tensor([[next_token]], dtype=torch.long, device=device)
        if next_token == eos_id:
            break

    return generated
