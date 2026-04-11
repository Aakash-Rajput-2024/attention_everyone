from __future__ import annotations
import math
import torch
import torch.nn as nn
from model import DecoderLayer, Encoder, SinusoidalPositionalEncoding


class FullTransformer(nn.Module):

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_layers: int = 6,
        total_head: int = 8,
        dropout: float = 0.1,
        max_len: int = 5000,
        share_embeddings: bool = False,
    ):
        super().__init__()
        self.d_model = d_model

        self.src_tok_emb = nn.Embedding(src_vocab_size, d_model)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, d_model)
        if share_embeddings and src_vocab_size == tgt_vocab_size:
            self.tgt_tok_emb.weight = self.src_tok_emb.weight

        self.pos_enc = SinusoidalPositionalEncoding(
            d_model, max_len=max_len, dropout=dropout
        )

        self.encoder_layers = nn.ModuleList(
            [Encoder(d_model, total_head, dropout=dropout) for _ in range(num_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, total_head, dropout=dropout) for _ in range(num_layers)]
        )

        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        if share_embeddings and src_vocab_size == tgt_vocab_size:
            self.fc_out.weight = self.tgt_tok_emb.weight

    # ------------------------------------------------------------------
    # Helpers for efficient inference (encoder run only once per source)
    # ------------------------------------------------------------------

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        """Run the encoder and return memory. Cache this for decoding."""
        x = self.pos_enc(self.src_tok_emb(src) * math.sqrt(self.d_model))
        for enc in self.encoder_layers:
            x = enc(x)
        return x  # (B, S, d_model)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """Run one decoder pass given pre-computed encoder memory."""
        y = self.pos_enc(self.tgt_tok_emb(tgt) * math.sqrt(self.d_model))
        for dec in self.decoder_layers:
            y = dec(y, memory)
        return self.fc_out(y)  # (B, T, vocab_size)

    # ------------------------------------------------------------------
    # Standard forward (used during training — runs encoder + decoder)
    # ------------------------------------------------------------------

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        src: (B, S) source token ids
        tgt: (B, T) target input ids (shifted right for teacher forcing)
        Returns logits (B, T, tgt_vocab_size).
        """
        memory = self.encode(src)
        return self.decode(tgt, memory)


if __name__ == "__main__":
    B, S, T = 2, 14, 11
    vs = vt = 300
    m = FullTransformer(vs, vt, num_layers=6, share_embeddings=True)
    src = torch.randint(0, vs, (B, S))
    tgt = torch.randint(0, vt, (B, T))
    logits = m(src, tgt)
    assert logits.shape == (B, T, vt)
    print("ok", logits.shape)
