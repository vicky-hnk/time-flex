"""
Original ICML 2022 implementation: https://github.com/MAZiqing/FEDformer
Paper link: https://proceedings.mlr.press/v162/zhou22g.html
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.layers.embed import DataEmbedding
from src.models.layers.decomp_layers import SeriesDecomposition
from src.models.layers.attention import AutoCorrelationLayer
from src.models.layers.fourier_correlation import (FourierBlock,
                                                   FourierCrossAttention)
from src.models.layers.multi_wavelet_correlation import (MultiWaveletCross,
                                                         MultiWaveletTransform)
from src.models.layers.transformer_layers import (AutoformerDecoderLayer,
                                                  AutoformerEncoderLayer,
                                                  Encoder, Decoder,
                                                  LayerNorm)


class FEDformer(nn.Module):
    """
    FEDformer performs the attention mechanism on frequency domain and
    achieved O(N) complexity.
    """

    def __init__(self, params, version='fourier', mode_select='random',
                 modes=32):
        """
        version: str, for FEDformer, there are two versions to choose,
        options: [Fourier, Wavelets].
        mode_select: str, for FEDformer, there are two mode selection method,
        options: [random, low].
        modes: int, modes to be selected.
        """
        super().__init__()
        self.task_name = params.task_name
        self.seq_len = params.seq_len
        self.seq_overlap = params.seq_overlap
        self.pred_len = params.pred_len

        self.version = version
        self.mode_select = mode_select
        self.modes = modes

        # Decomp
        self.decomp = SeriesDecomposition(params.window_size)
        self.enc_embedding = DataEmbedding(params.enc_in, params.model_dim,
                                           params.embed, params.freq,
                                           params.dropout)
        self.dec_embedding = DataEmbedding(params.dec_in, params.model_dim,
                                           params.embed, params.freq,
                                           params.dropout)

        if self.version == 'Wavelets':
            encoder_self_att = MultiWaveletTransform(ich=params.model_dim, L=1,
                                                     base='legendre')
            decoder_self_att = MultiWaveletTransform(ich=params.model_dim, L=1,
                                                     base='legendre')
            decoder_cross_att = MultiWaveletCross(
                in_channels=params.model_dim,
                out_channels=params.model_dim,
                seq_len_q=self.seq_len // 2 + self.pred_len,
                seq_len_kv=self.seq_len,
                modes=self.modes,
                ich=params.model_dim,
                base='legendre',
                activation='tanh')
        else:
            encoder_self_att = FourierBlock(
                in_channels=params.model_dim,
                out_channels=params.model_dim,
                seq_len=self.seq_len,
                modes=self.modes,
                mode_select_method=self.mode_select)
            decoder_self_att = FourierBlock(
                in_channels=params.model_dim, out_channels=params.model_dim,
                seq_len=self.seq_len // 2 + self.pred_len,
                modes=self.modes,
                mode_select_method=self.mode_select)
            decoder_cross_att = FourierCrossAttention(
                in_channels=params.model_dim,
                out_channels=params.model_dim,
                seq_len_q=self.seq_len // 2 + self.pred_len,
                seq_len_kv=self.seq_len,
                modes=self.modes,
                mode_select_method=self.mode_select,
                num_heads=params.num_heads)
        # Encoder
        self.encoder = Encoder(
            [
                AutoformerEncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,
                        # instead of multi-head attention in transformer
                        params.model_dim, params.num_heads),
                    params.model_dim,
                    params.ff_dim,
                    dropout=params.dropout,
                    activation=params.activation
                ) for _ in range(params.num_encoder_layers)
            ],
            norm_layer=LayerNorm(params.model_dim)
        )
        # Decoder
        # def __init__(self, auto_correlation, cross_attention, model_dim,
        #              output_feature_dim, kernel_size: int = 25,
        #              ff_dim=None, dropout: float = 0.1,
        #              activation: str = 'relu'):
        print("Output feature dim is ", params.c_out)
        self.decoder = Decoder(
            [
                AutoformerDecoderLayer(
                    AutoCorrelationLayer(
                        decoder_self_att,
                        params.model_dim, params.num_heads),
                    AutoCorrelationLayer(
                        decoder_cross_att,
                        d_model=params.model_dim, n_heads=params.num_heads),
                    model_dim=params.model_dim,
                    output_feature_dim=params.c_out,
                    ff_dim=params.ff_dim,
                    dropout=params.dropout,
                    activation=params.activation,
                )
                for _ in range(params.num_decoder_layers)
            ],
            norm_layer=LayerNorm(params.model_dim),
            projection=nn.Linear(params.model_dim, params.c_out, bias=True)
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len,
                                                            1)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat(
            [trend_init[:, -self.seq_overlap:, :], mean], dim=1)
        seasonal_init = F.pad(
            seasonal_init[:, -self.seq_overlap:, :], (0, 0, 0, self.pred_len))
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        enc_out, attns = self.encoder(enc_out)
        # dec
        seasonal_part, trend_part = self.decoder(dec_out, enc_out,
                                                 x_mask=None,
                                                 cross_mask=None,
                                                 trend=trend_init)
        # final
        return trend_part + seasonal_part

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if (self.task_name == 'long_term_forecast' or
                self.task_name == 'short_term_forecast'):
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        else:
            raise NotImplementedError
