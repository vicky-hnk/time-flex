from math import ceil

import torch
import torch.nn as nn
from einops import rearrange, repeat
from src.models.layers.Crossformer_EncDec import (CrossDecoder,
                                                  CrossDecoderLayer,
                                                  CrossEncoder, scale_block)
from src.models.layers.embed import PatchEmbedding
from src.models.layers.SelfAttention_Family import (AttentionLayer,
                                                    FullAttention,
                                                    TwoStageAttentionLayer)


class CrossFormer(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=vSVLM2j9eie
    """

    def __init__(self, params):
        super().__init__()
        self.enc_in = params.enc_in
        self.seq_len = params.seq_len
        self.pred_len = params.pred_len
        self.seg_len = 12
        self.win_size = 2
        self.task_name = params.task_name

        # The padding operation to handle invisible segment length
        self.pad_in_len = ceil(
            1.0 * params.seq_len / self.seg_len) * self.seg_len
        self.pad_out_len = ceil(
            1.0 * params.pred_len / self.seg_len) * self.seg_len
        self.in_seg_num = self.pad_in_len // self.seg_len
        self.out_seg_num = ceil(
            self.in_seg_num / (
                        self.win_size ** (params.num_encoder_layers - 1)))
        self.head_nf = params.model_dim * self.out_seg_num

        # Embedding
        self.enc_value_embedding = PatchEmbedding(
            params.model_dim, self.seg_len, self.seg_len,
            self.pad_in_len - params.seq_len, 0)
        self.enc_pos_embedding = nn.Parameter(
            torch.randn(1, params.enc_in, self.in_seg_num, params.model_dim))
        self.pre_norm = nn.LayerNorm(params.model_dim)

        # Encoder
        self.encoder = CrossEncoder(
            [
                scale_block(params, 1 if l is 0 else self.win_size,
                            params.model_dim, params.num_heads, params.ff_dim,
                            1, params.dropout,
                            self.in_seg_num if l is 0 else ceil(
                                self.in_seg_num / self.win_size ** l),
                            params.factor
                            ) for l in range(params.num_encoder_layers)
            ]
        )
        # Decoder
        self.dec_pos_embedding = nn.Parameter(
            torch.randn(1, params.enc_in, (self.pad_out_len // self.seg_len),
                        params.model_dim))

        self.decoder = CrossDecoder(
            [
                CrossDecoderLayer(
                    TwoStageAttentionLayer(params,
                                           (self.pad_out_len // self.seg_len),
                                           params.factor, params.model_dim,
                                           params.num_heads,
                                           params.ff_dim, params.dropout),
                    AttentionLayer(
                        FullAttention(False, params.factor,
                                      attention_dropout=params.dropout,
                                      output_attention=False),
                        params.model_dim, params.num_heads),
                    self.seg_len,
                    params.model_dim,
                    params.ff_dim,
                    dropout=params.dropout,
                    # activation=params.activation,
                )
                for l in range(params.num_encoder_layers + 1)
            ],
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # embedding
        x_enc, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))
        x_enc = rearrange(x_enc,
                          '(b d) seg_num model_dim -> b d seg_num model_dim',
                          d=n_vars)
        x_enc += self.enc_pos_embedding
        x_enc = self.pre_norm(x_enc)
        enc_out, attns = self.encoder(x_enc)

        dec_in = repeat(self.dec_pos_embedding,
                        'b ts_d l d -> (repeat b) ts_d l d',
                        repeat=x_enc.shape[0])
        dec_out = self.decoder(dec_in, enc_out)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if (self.task_name == 'long_term_forecast' or
                self.task_name == 'short_term_forecast'):
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        return None

