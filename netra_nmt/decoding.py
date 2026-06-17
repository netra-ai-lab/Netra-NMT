"""
netra_nmt.decoding — decoding strategies (greedy, beam search, nucleus sampling).

Each function takes an encoded source and returns a list of generated token ids
(including the leading BOS). They are decoupled from tokenisation and I/O so the
:class:`~netra_nmt.translator.NetraTranslator` can compose them freely.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import NetraNMT


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
    Beam search decoding.
    Each beam is a tuple of (cumulative_log_prob, token_id_list, finished).
    length_penalty: >1 favours longer outputs, <1 favours shorter.
    """
    device = input_ids.device

    # Encode source once; repeat enc_out beam_size times
    enc_out = model.encoder(input_ids, src_key_padding_mask=src_mask)   # (1, S, D)
    enc_out = enc_out.repeat(beam_size, 1, 1)                           # (B, S, D)
    src_mask = src_mask.repeat(beam_size, 1)                            # (B, S)

    beams: list[tuple[float, list[int], bool]] = [(0.0, [bos_id], False)]

    for _ in range(max_new_tokens):
        if all(done for _, _, done in beams):
            break

        active_idx = [i for i, (_, _, done) in enumerate(beams) if not done]

        # Build decoder input from current beam sequences
        max_t = max(len(beams[i][1]) for i in active_idx)
        dec_ids = torch.zeros(len(active_idx), max_t, dtype=torch.long, device=device)
        for j, i in enumerate(active_idx):
            seq = beams[i][1]
            dec_ids[j, :len(seq)] = torch.tensor(seq, device=device)

        enc_slice = enc_out[:len(active_idx)]
        mask_slice = src_mask[:len(active_idx)]

        pos = torch.arange(max_t, device=device).unsqueeze(0)
        x = model.decoder.dropout(
            model.decoder.embed(dec_ids) + model.decoder.pos_embed(pos)
        )
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            max_t, device=device, dtype=x.dtype
        )
        for layer in model.decoder.layers:
            x = layer(x, enc_slice, tgt_mask=causal_mask,
                      memory_key_padding_mask=mask_slice)
        x = model.decoder.norm(x)

        logits = model.lm_head(x[:, -1, :])           # (active, V)
        log_probs = F.log_softmax(logits, dim=-1)     # (active, V)

        # Keep finished beams as-is
        candidates: list[tuple[float, list[int], bool]] = [
            (score, seq, True) for score, seq, done in beams if done
        ]

        top_k_vals, top_k_ids = log_probs.topk(beam_size, dim=-1)  # (active, beam_size)
        for j, i in enumerate(active_idx):
            score, seq, _ = beams[i]
            for k in range(beam_size):
                new_token = top_k_ids[j, k].item()
                new_score = score + top_k_vals[j, k].item()
                candidates.append((new_score, seq + [new_token], new_token == eos_id))

        def norm_score(cand):
            s, seq, _ = cand
            lp = ((5 + len(seq)) / 6) ** length_penalty
            return s / lp

        candidates.sort(key=norm_score, reverse=True)
        beams = candidates[:beam_size]

    best = max(beams, key=lambda c: c[0] / (((5 + len(c[1])) / 6) ** length_penalty))
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
    """
    Nucleus (top-p) sampling with temperature.
    temperature < 1  →  sharper, more confident
    temperature > 1  →  flatter, more creative
    top_p            →  only sample from tokens whose cumulative prob ≥ top_p
    """
    device = input_ids.device
    enc_out = model.encoder(input_ids, src_key_padding_mask=src_mask)

    generated = [bos_id]

    for _ in range(max_new_tokens):
        T = len(generated)
        dec_ids = torch.tensor([generated], dtype=torch.long, device=device)
        pos = torch.arange(T, device=device).unsqueeze(0)

        x = model.decoder.dropout(
            model.decoder.embed(dec_ids) + model.decoder.pos_embed(pos)
        )
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=device, dtype=x.dtype
        )
        for layer in model.decoder.layers:
            x = layer(x, enc_out, tgt_mask=causal_mask,
                      memory_key_padding_mask=src_mask)
        x = model.decoder.norm(x)

        logits = model.lm_head(x[0, -1, :]) / temperature   # (V,)

        # Nucleus filtering
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum_probs > top_p
        remove[0] = False                          # always keep the top token
        sorted_logits[remove] = float("-inf")

        filtered_logits = torch.full_like(logits, float("-inf"))
        filtered_logits[sorted_idx] = sorted_logits
        next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), 1).item()

        generated.append(next_token)
        if next_token == eos_id:
            break

    return generated
