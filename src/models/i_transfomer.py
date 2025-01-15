"""
Implementation of iTransformer model with inverse tokens for time series
forecasting.

References :
    https://github.com/thuml/iTransformer
    https://arxiv.org/pdf/2310.06625
"""

import torch
from src.models.layers.attention import AttentionLayer, FullAttention
from src.models.layers.embed import DataEmbeddingInverted
from src.models.layers.transformer_layers import Encoder, EncoderLayer
from torch import nn


class Itransformer(nn.Module):
    """
    Transformer model with inverse tokens for time series forecasting.
    A linear projection layer serves as decoder,
    reducing complexity.
    """

    def __init__(self, params) -> None:
        """
         Parameters:
            :param params (object): Collection of training and model-specific
            parameters. A description of all parameters can be found within
            the argument parser implementation (src.util.argument_parser.py).
                - seq_len (int): The length of the input time series sequence.
                - pred_len (int): The length of the prediction horizon.
                - window_size (int): The size of the kernel used within
                decomposition.
                - num_variates (int): The num of variates of time series.
        """
        super().__init__()
        self.seq_len = params.seq_len
        self.pred_len = params.pred_len
        self.window_size = params.window_size
        self.variates = params.num_variates
        # self.output_attention = params.output_attention
        self.use_norm = params.use_norm
        # Embedding
        self.enc_embedding = DataEmbeddingInverted(c_in=params.seq_len,
                                                   model_dim=params.model_dim,
                                                   dropout=params.dropout)
        # Encoder
        self.encoder = Encoder(
            [EncoderLayer(
                AttentionLayer(
                    attention=FullAttention(
                        False, params.factor,
                        attention_dropout=params.attention_dropout,
                        output_attention=params.output_attention),
                    model_dim=params.model_dim, num_heads=params.num_heads),
                params.model_dim, ff_dim=params.ff_dim,
                dropout=params.dropout, activation=params.activation
            ) for _ in range(params.num_encoder_layers)],
            norm_layer=torch.nn.LayerNorm(params.model_dim)
        )
        # Decoder
        self.projector = nn.Linear(params.model_dim, params.pred_len,
                                   bias=True)

        # set device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor) \
            -> torch.Tensor:
        #  Normalization
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            std = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= std

        _, _, num_variates = x_enc.shape  # dim [batch_size, seq_len, variates]
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        # Encoding
        enc_out, _ = self.encoder(enc_out)
        # Decoding
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :num_variates]
        # dim is [batch, pred_len, variates]
        # De-Normalization
        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (
                std[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (
                means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out  # dim is [batch, seq_len, variates]
