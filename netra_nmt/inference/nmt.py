"""The translation model on ONNX Runtime: token ids in, token ids out.

A model folder holds two graphs and the tokenizer (``netra-model.json`` names
them and gives the decoder's shape):

    encoder.onnx        tokens (B, S), mask (B, S) -> cross_k, cross_v (L, B, H, S, hd)
    decoder_step.onnx   step_ids (B, 1), past_k/v (L, B, H, P, hd), cross_k/v, cross_mask (B, S)
                        -> logits (B, V), present_k/v (L, B, H, P+1, hd)
    spm_32k.model       SentencePiece, shared by English and (word-segmented) Khmer

The three decoders below are ports of the PyTorch ones the model was evaluated
with (training/netra_nmt_train/decoding.py): same stopping rules, same GNMT
length penalty, same top-p filter.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .hub import load_sessions, model_file


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


class NMTModel:
    """One Netra NMT model folder, loaded on ``device``."""

    def __init__(self, source: str | Path, device: str | None = "auto"):
        self.root, self.config, self.variant, self.sessions, self.device = load_sessions(source, device)
        components = self.config["components"]
        dec = components["decoder"]
        self.layers, self.heads, self.head_dim = dec["layers"], dec["heads"], dec["head_dim"]
        self.pad_id, self.bos_id, self.eos_id = dec["pad_id"], dec["bos_id"], dec["eos_id"]
        self.max_len = components["preprocess"]["max_len"]
        self.lang_markers = components["preprocess"]["lang_markers"]

        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(model_file(self.root, components["tokenizer"]["model"])))
        self._on_gpu = self.device != "cpu"

    # -- tokens <-> text ---------------------------------------------------

    def encode_text(self, text: str, target_lang: str) -> list[int]:
        """``"<2km> text"`` -> ids, cut to the training length."""
        ids = self.sp.encode(f"{self.lang_markers[target_lang]} {text.strip()}", out_type=int)[: self.max_len]
        if not ids:
            raise ValueError("Input text encoded to zero tokens — is it empty?")
        return ids

    def decode_ids(self, ids: list[int]) -> str:
        specials = {self.pad_id, self.bos_id, self.eos_id}
        return self.sp.decode([t for t in ids if t not in specials])

    # -- graphs --------------------------------------------------------------

    def _encode(self, ids: list[int], keep: bool):
        tokens = np.asarray([ids], dtype=np.int64)
        out = self.sessions["encoder"].run(
            {"tokens": tokens, "mask": np.ones_like(tokens)},
            keep=("cross_k", "cross_v") if keep else (),
        )
        cross_mask = np.zeros((1, len(ids)), dtype=np.float32)
        return out["cross_k"], out["cross_v"], cross_mask

    def _empty_cache(self, batch: int) -> np.ndarray:
        return np.zeros((self.layers, batch, self.heads, 0, self.head_dim), dtype=np.float32)

    def _step(self, ids, past_k, past_v, cross_k, cross_v, cross_mask, keep: bool):
        out = self.sessions["decoder_step"].run(
            {"step_ids": np.asarray(ids, dtype=np.int64).reshape(-1, 1), "past_k": past_k, "past_v": past_v,
             "cross_k": cross_k, "cross_v": cross_v, "cross_mask": cross_mask},
            keep=("present_k", "present_v") if keep else (),
        )
        return out["logits"], out["present_k"], out["present_v"]

    # -- decoding ------------------------------------------------------------

    def greedy(self, ids: list[int], max_new_tokens: int = 128) -> list[int]:
        """Argmax at every step; stops at EOS. Returns BOS + tokens (+ EOS)."""
        keep = self._on_gpu
        cross_k, cross_v, cross_mask = self._encode(ids, keep)
        past_k = past_v = self._empty_cache(1)
        out, token = [self.bos_id], self.bos_id
        for _ in range(max_new_tokens):
            logits, past_k, past_v = self._step([token], past_k, past_v, cross_k, cross_v, cross_mask, keep)
            token = int(logits[0].argmax())
            out.append(token)
            if token == self.eos_id:
                break
        return out

    def beam(self, ids: list[int], max_new_tokens: int = 128, beam_size: int = 5,
             length_penalty: float = 0.6, early_stop: bool = True) -> list[int]:
        """Beam search, GNMT length penalty ``((5 + len) / 6) ** length_penalty``.

        Stops as soon as no live beam can beat the best finished hypothesis, which
        gives exactly the result of running all ``max_new_tokens`` steps.
        """
        cross_k, cross_v, cross_mask = self._encode(ids, keep=False)
        cross_k = np.repeat(cross_k, beam_size, axis=1)
        cross_v = np.repeat(cross_v, beam_size, axis=1)
        cross_mask = np.repeat(cross_mask, beam_size, axis=0)
        past_k = past_v = self._empty_cache(beam_size)

        scores = np.full(beam_size, -np.inf, dtype=np.float32)
        scores[0] = 0.0                                   # only beam 0 is live at first
        seqs = np.full((beam_size, 1), self.bos_id, dtype=np.int64)
        step_ids = seqs[:, 0]
        finished: list[tuple[float, list[int]]] = []

        def norm(score: float, length: int) -> float:
            return score / (((5 + length) / 6) ** length_penalty)

        max_len = 1 + max_new_tokens                      # BOS + every new token

        for _ in range(max_new_tokens):
            logits, past_k, past_v = self._step(step_ids, past_k, past_v, cross_k, cross_v, cross_mask, False)
            log_probs = log_softmax(logits.astype(np.float32))
            vocab = log_probs.shape[-1]
            cand = (scores[:, None] + log_probs).reshape(-1)
            top = np.argpartition(-cand, beam_size)[:beam_size]
            top = top[np.argsort(-cand[top], kind="stable")]
            parent, token = top // vocab, top % vocab

            past_k, past_v = past_k[:, parent], past_v[:, parent]
            seqs = np.concatenate([seqs[parent], token[:, None]], axis=1)
            scores = cand[top].astype(np.float32)
            step_ids = token

            for b in range(beam_size):
                if token[b] == self.eos_id and scores[b] > -np.inf:
                    finished.append((norm(float(scores[b]), seqs.shape[1]), seqs[b].tolist()))
                    scores[b] = -np.inf
            if scores.max() == -np.inf:
                break
            # Exact early stop: a live beam's score only falls (log-probs <= 0), so
            # the best it can still reach is score / penalty(longest length). Once
            # the best finished hypothesis beats that for every beam, nothing that
            # follows can win (and max() below keeps the first of equal scores).
            if early_stop and finished and max(c[0] for c in finished) >= norm(float(scores.max()), max_len):
                break

        for b in range(beam_size):                        # beams still running at the cap
            if scores[b] > -np.inf:
                finished.append((norm(float(scores[b]), seqs.shape[1]), seqs[b].tolist()))
        return max(finished, key=lambda c: c[0])[1]

    def sample(self, ids: list[int], max_new_tokens: int = 128, temperature: float = 1.0,
               top_p: float = 0.95, seed: int | None = None) -> list[int]:
        """Nucleus (top-p) sampling with temperature."""
        rng = np.random.default_rng(seed)
        cross_k, cross_v, cross_mask = self._encode(ids, keep=False)
        past_k = past_v = self._empty_cache(1)
        out, token = [self.bos_id], self.bos_id
        for _ in range(max_new_tokens):
            logits, past_k, past_v = self._step([token], past_k, past_v, cross_k, cross_v, cross_mask, False)
            logits = logits[0].astype(np.float64) / temperature
            order = np.argsort(-logits, kind="stable")
            sorted_logits = logits[order]
            probs = np.exp(sorted_logits - sorted_logits.max())
            probs /= probs.sum()
            remove = np.cumsum(probs) > top_p
            remove[0] = False                             # always keep the top token
            probs[remove] = 0.0
            probs /= probs.sum()
            token = int(order[rng.choice(len(probs), p=probs)])
            out.append(token)
            if token == self.eos_id:
                break
        return out
