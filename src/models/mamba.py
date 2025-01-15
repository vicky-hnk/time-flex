"""
This Mamba implementation is a version with minor changes from:
https://github.com/thuml/Time-Series-Library
However, that implementation only works for seq_len>=pred_len.
To maintain comparability with existing implementations, we set
seq_len=pred_len in our experiments.
"""
import math

import torch
import torch.nn as nn
from mamba_ssm import Mamba
from src.models.layers.embed import DataEmbedding


class MambaModel(nn.Module):

    def __init__(self, params):
        """
        Parameters:
            :param params.ff_dim:  size of the intermediate layer in the
            feedforward neural network component of the model
            :param params.conv_dim:  width of the convolutional layers used
            within the model to capture local dependencies
            :param params.expand: factor  to expand the model dimension to a
            larger internal representation before processing (higher
            dimensional space)
        """
        super().__init__()
        # set device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.pred_len = params.pred_len

        self.d_inner = params.model_dim * params.expand
        self.dt_rank = math.ceil(params.model_dim / 16)

        self.embedding = DataEmbedding(params.enc_in, params.model_dim,
                                       params.embed, params.freq,
                                       params.dropout)

        self.mamba = Mamba(
            d_model=params.model_dim,
            d_state=params.ff_dim,
            d_conv=params.conv_dim,
            expand=params.expand,
        )

        self.out_layer = nn.Linear(params.model_dim, params.output_feature_dim,
                                   bias=False)

    def forecast(self, x_enc, x_mark_enc):
        mean_enc = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True,
                                       unbiased=False) + 1e-5).detach()
        x_enc = x_enc / std_enc

        x = self.embedding(x_enc, x_mark_enc)
        x = self.mamba(x)
        x_out = self.out_layer(x)

        x_out = x_out * std_enc + mean_enc
        return x_out

    def forward(self, x_enc, x_mark_enc):
        x_out = self.forecast(
            x_enc.to(self.device), x_mark_enc.to(self.device))
        return x_out[:, -self.pred_len:, :]
