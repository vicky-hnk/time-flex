"""
VCFormer original implementation: https://github.com/CSyyn/VCformer/tree/main
(published at IJCAI 2024)
"""

import torch
import torch.nn as nn
from src.models.layers.embed import DataEmbedding_inverted_vc
from src.models.layers.KTD import KTDlayer
from src.models.layers.SelfAttention_Family import (VarCorAttention,
                                                    VarCorAttentionLayer)
from src.models.layers.transformer_layers import Encoder
from src.models.layers.VCformer_Enc import VCEncoderLayer


class VCFormer(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2405.11470
    """

    def __init__(self, params):
        super().__init__()
        self.task_name = params.task_name
        self.seq_len = params.seq_len
        self.pred_len = params.pred_len
        self.output_attention = params.output_attention

        # Embedding
        self.enc_embedding = DataEmbedding_inverted_vc(
            c_in=params.seq_len, d_model=params.model_dim,
            embed_type=params.embed)

        # Encoder
        self.encoder = Encoder(
            [
                VCEncoderLayer(
                    VarCorAttentionLayer(
                        VarCorAttention(
                            params, False, params.factor,
                            attention_dropout=params.dropout,
                            output_attention=params.output_attention),
                        params.model_dim, params.num_heads),
                    KTDlayer(params, params.model_dim, params.snap_size,
                             params.proj_dim, params.hidden_dim,
                             params.hidden_layers),
                    params.model_dim,
                    params.ff_dim,
                    dropout=params.dropout,
                    activation=params.activation
                ) for l in range(params.num_encoder_layers)
            ],
            norm_layer=torch.nn.LayerNorm(params.model_dim)
        )

        # Decoder
        self.projection = nn.Linear(
            params.model_dim, params.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stand_dev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stand_dev

        _, _, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        enc_out, attns = self.encoder(enc_out)
        # reshape enc_out[B,D,T] -> dec_out[B,T,D]
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (
            stand_dev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (
            means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out, attns

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, attns = self.forecast(
            x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
