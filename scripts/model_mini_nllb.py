import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

class SwiGLUFFN(nn.Module):
    """
    SwiGLU feed-forward network.
    Projects to ffn_dim * 2, splits into gate + value, applies SiLU gating,
    then projects back to d_model.
    """

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
    """
    Standard (bidirectional) transformer encoder block with Pre-LN.
    No causal mask — the encoder attends to all positions freely.
    Accepts an optional src_key_padding_mask for padding tokens.
    """

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Pre-LN self-attention (no causal mask — bidirectional)
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(
            x, x, x,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
        )
        x = self.dropout(x) + residual

        # Pre-LN FFN
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        return x + residual


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        ffn_dim: int,
        max_len: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=1)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        input_ids: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)  # (1, T)
        x = self.dropout(self.embed(input_ids) + self.pos_embed(pos))

        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)

        return self.norm(x)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class DecoderBlock(nn.Module):
    """
    Transformer decoder block with Pre-LN.
    - Masked self-attention (causal mask passed in from Decoder.forward)
    - Cross-attention over encoder output
    - SwiGLU FFN
    """

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc_out: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Pre-LN masked self-attention (causal)
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(
            x, x, x,
            attn_mask=tgt_mask,                       # FIX 3: causal mask applied
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False,
        )
        x = self.dropout(x) + residual

        # Pre-LN cross-attention
        residual = x
        x = self.norm2(x)
        x, _ = self.cross_attn(
            x, enc_out, enc_out,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )
        x = self.dropout(x) + residual

        # Pre-LN FFN
        residual = x
        x = self.norm3(x)
        x = self.ffn(x)
        return x + residual


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        ffn_dim: int,
        max_len: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=1)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        input_ids: torch.Tensor,
        enc_out: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)  # (1, T)
        x = self.dropout(self.embed(input_ids) + self.pos_embed(pos))

        # FIX 3: generate the causal mask here once and pass it to every layer
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=input_ids.device, dtype=x.dtype
        )

        for layer in self.layers:
            x = layer(
                x,
                enc_out,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        return self.norm(x)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class MiniNLLB(nn.Module):
    """
    Minimal encoder-decoder translation model in the style of NLLB.

    Fixes applied vs. the original:
      1. Encoder uses no causal mask (bidirectional attention).
      2. Weight tying: lm_head ↔ decoder.embed  (not encoder.embed).
      3. Causal mask is generated inside Decoder.forward and passed to every
         decoder layer, so the model cannot peek at future tokens.
      4. Encoder and decoder embeddings are kept separate (different
         languages / roles); only the decoder embed is tied to lm_head.
      5. Padding mask support threaded through both encoder and decoder.
      6. Dropout added to SwiGLUFFN output for regularisation.
    """

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

        self.encoder = Encoder(
            vocab_size, d_model, enc_layers, n_heads, ffn_dim, max_len, dropout
        )
        self.decoder = Decoder(
            vocab_size, d_model, dec_layers, n_heads, ffn_dim, max_len, dropout
        )

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # FIX 2: tie lm_head to the *decoder* embedding, not the encoder's
        self.lm_head.weight = self.decoder.embed.weight

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform for linear layers, normal for embeddings."""
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
        """
        Args:
            input_ids:            (B, S) — source token ids
            decoder_input_ids:    (B, T) — target token ids, right-shifted
                                           (starts with <BOS>, ends before <EOS>)
            src_key_padding_mask: (B, S) bool — True where source is padding
            tgt_key_padding_mask: (B, T) bool — True where target is padding

        Returns:
            logits: (B, T, vocab_size)
        """
        enc_out = self.encoder(input_ids, src_key_padding_mask=src_key_padding_mask)

        dec_out = self.decoder(
            decoder_input_ids,
            enc_out,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )

        return self.lm_head(dec_out)  # (B, T, vocab_size)

    # -----------------------------------------------------------------------
    # Greedy inference helper
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
        """
        Greedy autoregressive decoding.

        Args:
            input_ids:            (B, S) source token ids
            bos_token_id:         id of the <BOS> / language token
            eos_token_id:         id of <EOS>
            max_new_tokens:       maximum tokens to generate
            src_key_padding_mask: (B, S) bool padding mask for the source

        Returns:
            generated: (B, T) token ids including the leading BOS token
        """
        self.eval()
        device = input_ids.device
        B = input_ids.size(0)

        # Encode source once
        enc_out = self.encoder(input_ids, src_key_padding_mask=src_key_padding_mask)

        # Start decoder input with BOS
        decoder_input_ids = torch.full((B, 1), bos_token_id, dtype=torch.long, device=device)

        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            T = decoder_input_ids.size(1)
            pos = torch.arange(T, device=device).unsqueeze(0)
            x = self.decoder.dropout(
                self.decoder.embed(decoder_input_ids) + self.decoder.pos_embed(pos)
            )

            causal_mask = nn.Transformer.generate_square_subsequent_mask(
                T, device=device, dtype=x.dtype
            )

            for layer in self.decoder.layers:
                x = layer(
                    x,
                    enc_out,
                    tgt_mask=causal_mask,
                    memory_key_padding_mask=src_key_padding_mask,
                )
            x = self.decoder.norm(x)

            # Greedy: pick the highest-probability token at the last position
            next_token_logits = self.lm_head(x[:, -1, :])          # (B, vocab)
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)  # (B, 1)

            decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)

            finished |= (next_token.squeeze(-1) == eos_token_id)
            if finished.all():
                break

        return decoder_input_ids


# # ---------------------------------------------------------------------------
# # Quick sanity check
# # ---------------------------------------------------------------------------

# if __name__ == "__main__":
#     VOCAB   = 32_000
#     B, S, T = 2, 20, 18

#     model = MiniNLLB(vocab_size=VOCAB, d_model=256, enc_layers=2, dec_layers=2,
#                      n_heads=4, ffn_dim=512, max_len=64)

#     src = torch.randint(2, VOCAB, (B, S))
#     tgt = torch.randint(2, VOCAB, (B, T))

#     # Padding masks: last 3 source tokens and last 2 target tokens are padding
#     src_pad = torch.zeros(B, S, dtype=torch.bool)
#     src_pad[:, -3:] = True
#     tgt_pad = torch.zeros(B, T, dtype=torch.bool)
#     tgt_pad[:, -2:] = True

#     logits = model(src, tgt, src_key_padding_mask=src_pad, tgt_key_padding_mask=tgt_pad)
#     print(f"logits shape : {logits.shape}")   # expect (2, 18, 32000)

#     # Verify weight tying
#     assert model.lm_head.weight.data_ptr() == model.decoder.embed.weight.data_ptr(), \
#         "lm_head and decoder embed are NOT tied!"
#     print("Weight tying : OK (lm_head ↔ decoder.embed)")

#     # Greedy generation
#     generated = model.generate(src, bos_token_id=2, eos_token_id=3,
#                                 max_new_tokens=10, src_key_padding_mask=src_pad)
#     print(f"Generated    : {generated.shape}")  # expect (2, ≤11)
#     print("All checks passed.")