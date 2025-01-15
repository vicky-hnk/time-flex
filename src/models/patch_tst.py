import torch
from src.models.layers.embed import PatchEmbedding
from src.models.layers.SelfAttention_Family import (AttentionLayer,
                                                    FullAttention)
from src.models.layers.transformer_layers import Encoder, EncoderLayer
from torch import nn


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x model_dim x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class PatchTST(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """

    def __init__(self, params, patch_len=16, stride=8):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.task_name = params.task_name
        self.seq_len = params.seq_len
        self.pred_len = params.pred_len
        padding = stride

        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            d_model=params.model_dim, patch_len=patch_len, stride=stride,
            padding=padding, dropout=params.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False, params.factor,
                            attention_dropout=params.dropout,
                            output_attention=params.output_attention),
                        params.model_dim, params.num_heads),
                    params.model_dim,
                    params.ff_dim,
                    dropout=params.dropout,
                    activation=params.activation
                ) for _ in range(params.num_encoder_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1, 2),
                                     nn.BatchNorm1d(params.model_dim),
                                     Transpose(1, 2))
        )

        # Prediction Head
        self.head_nf = (
                params.model_dim *
                int((params.seq_len - patch_len) / stride + 2))
        if (self.task_name == 'long_term_forecast' or
                self.task_name == 'short_term_forecast'):
            self.head = FlattenHead(params.enc_in, self.head_nf,
                                    params.pred_len,
                                    head_dropout=params.dropout)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x model_dim]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x model_dim]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x model_dim]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x model_dim x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if (self.task_name == 'long_term_forecast' or
                self.task_name == 'short_term_forecast'):
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        return None
