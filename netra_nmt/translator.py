"""
netra_nmt.translator — high-level translation API.

Example
-------
    from netra_nmt import NetraTranslator

    t = NetraTranslator()                       # downloads weights on first use
    t.translate("Hello, how are you?", direction="en2km")
    t.translate("សួស្តី", direction="km2en")
"""

from __future__ import annotations

from pathlib import Path

import sentencepiece as spm
import torch
from safetensors.torch import load_file

from . import decoding
from .config import DEFAULT_DIRECTION, DIRECTIONS, LANG_MARKERS, ModelConfig
from .model import NetraNMT
from .weights import resolve


class NetraTranslator:
    """
    Loads the model + tokenizer once and translates text in either direction.

    Args:
        repo_id:   Hugging Face repo to pull weights from (defaults to the
                   packaged repo id / ``NETRA_NMT_REPO_ID``).
        local_dir: local export directory to load from instead of the Hub.
        device:    "cuda" | "cpu" | torch.device. Auto-detected if None.
    """

    def __init__(
        self,
        repo_id: str | None = None,
        local_dir: str | Path | None = None,
        device: str | torch.device | None = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        weights_path, config_path, spm_path = resolve(repo_id=repo_id, local_dir=local_dir)

        self.config = ModelConfig.from_json(config_path)

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(spm_path))

        # Build in fp32 and let the fp16 export upcast on load — fp32 inference
        # is robust on both CPU and GPU; the half precision is only for storage.
        model = NetraNMT(
            vocab_size=self.config.vocab_size,
            d_model=self.config.d_model,
            enc_layers=self.config.enc_layers,
            dec_layers=self.config.dec_layers,
            n_heads=self.config.n_heads,
            ffn_dim=self.config.ffn_dim,
            max_len=self.config.max_len,
            dropout=0.0,
        )
        state_dict = load_file(str(weights_path))
        # lm_head.weight is tied to decoder.embed.weight and intentionally not
        # stored, so strict=False is expected here.
        model.load_state_dict(state_dict, strict=False)
        self.model = model.to(self.device).eval()

    # -----------------------------------------------------------------
    # internals
    # -----------------------------------------------------------------

    def _trucase_en(self, text: str) -> str:
        try:
            import nltk
            words = text.split()
            cap_words = [w[0].upper() + w[1:] if w else w for w in words]
            tagged = nltk.pos_tag(cap_words)
            result = []
            for orig, (_, tag) in zip(words, tagged):
                if tag in ('NNP', 'NNPS'):
                    result.append(orig[0].upper() + orig[1:])
                elif orig.lower() == 'i':
                    result.append('I')
                else:
                    result.append(orig)
            return ' '.join(result)
        except ImportError:
            return text
        except LookupError:
            return text

    def _encode(self, text: str, target_lang: str):
        marker = LANG_MARKERS[target_lang]
        src = f"{marker} {text.strip()}"
        ids = self.sp.encode(src, out_type=int)[: self.config.max_len]
        if not ids:
            raise ValueError("Input text encoded to zero tokens — is it empty?")
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        src_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        return input_ids, src_mask

    def _decode(self, ids: list[int]) -> str:
        specials = {self.config.pad_id, self.config.bos_id, self.config.eos_id}
        clean = [t for t in ids if t not in specials]
        return self.sp.decode(clean)

    # -----------------------------------------------------------------
    # public API
    # -----------------------------------------------------------------

    def translate(
        self,
        text: str,
        direction: str = DEFAULT_DIRECTION,
        mode: str = "greedy",
        beam_size: int = 5,
        length_penalty: float = 0.6,
        temperature: float = 1.0,
        top_p: float = 0.95,
        max_new_tokens: int = 128,
    ) -> str:
        """
        Translate a single string.

        Args:
            direction: "en2km" (English→Khmer) or "km2en" (Khmer→English).
            mode:      "greedy" | "beam" | "sample".
        """
        if direction not in DIRECTIONS:
            raise ValueError(
                f"Unknown direction {direction!r}. Choose one of {sorted(DIRECTIONS)}."
            )
        _src_lang, tgt_lang = DIRECTIONS[direction]

        if direction == "en2km":
            text = self._trucase_en(text)

        input_ids, src_mask = self._encode(text, tgt_lang)
        bos_id, eos_id = self.config.bos_id, self.config.eos_id

        if mode == "greedy":
            tokens = decoding.greedy_decode(
                self.model, input_ids, src_mask, bos_id, eos_id, max_new_tokens
            )
        elif mode == "beam":
            tokens = decoding.beam_search(
                self.model, input_ids, src_mask, bos_id, eos_id, max_new_tokens,
                beam_size=beam_size, length_penalty=length_penalty,
            )
        elif mode == "sample":
            tokens = decoding.sample_decode(
                self.model, input_ids, src_mask, bos_id, eos_id, max_new_tokens,
                temperature=temperature, top_p=top_p,
            )
        else:
            raise ValueError(f"Unknown mode {mode!r}. Choose greedy | beam | sample.")

        return self._decode(tokens)

    def translate_batch(
        self,
        texts: list[str],
        direction: str = DEFAULT_DIRECTION,
        **kwargs,
    ) -> list[str]:
        """Translate a list of strings (sequentially). Returns a list of outputs."""
        return [self.translate(t, direction=direction, **kwargs) for t in texts]


# ---------------------------------------------------------------------------
# Module-level convenience wrapper with a lazily-built default translator.
# ---------------------------------------------------------------------------

_DEFAULT: NetraTranslator | None = None


def translate(text: str, direction: str = DEFAULT_DIRECTION, **kwargs) -> str:
    """
    One-shot translation using a cached default :class:`NetraTranslator`.

    Convenient for quick use; instantiate :class:`NetraTranslator` directly to
    control device, repo id, or to reuse across many calls.
    """
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = NetraTranslator()
    return _DEFAULT.translate(text, direction=direction, **kwargs)
