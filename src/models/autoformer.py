"""
This module implements the Autoformer architecture described in "Autoformer:
Decomposition Transformers with
Auto-Correlation for Long-Term Series Forecasting"

References:
    https://arxiv.org/pdf/2106.13008.pdf.
    https://github.com/thuml/Autoformer

Architecture:
    The time series is decomposed into trend and seasonal components.
    In contrast to common Transformer architectures Autoformer uses
    cross-correlation instead of Attention. l which is
    calculated on the Fourier transform of the time series. We use the torch
    implementation of FFT to pose the tensor
    in the GPU to accelerate the computation.
"""
import torch
from src.models.layers.attention import AttentionLayer, AutoCorrelation
from src.models.layers.decomp_layers import SeriesDecomposition
from src.models.layers.embed import PerformEmbedding
from src.models.layers.normalization import NormalizationLayer
from src.models.layers.transformer_layers import (AutoformerDecoderLayer,
                                                  AutoformerEncoderLayer)
from torch import nn


class AutoformerEncoder(nn.Module):
    """
    The encoder consists of a residual auto-correlation block, followed by a
    series decomposition and a residual forward
    block. It's output just contains a seasonal component.
        S1 = Decomposition(Auto-Correlation(X) + X)
        S2 = Decomposition(Forward(S1) + S1)
    """

    def __init__(self, attn_layers, conv_layers=None,
                 normalization_layer=None):
        """
        Parameters:
            :param attn_layers: A collection of all attention layers
            (here specifically an auto-correlation layers), the
            number of encoder layers can be defined as hyperparameter.
            :param conv_layers: A collection of additional convolution layers
            applied after each attention block,
            default value is None.
            :param normalization_layer: If not None, a normalization layer is
            applied. This is highly recommended to
            maintain consistency of layer input distributions across epochs
            (prevent internal covariate shift and gradient
            vanishing).

        Remark:
            This implementation differs from the original paper: in the
            original implementation the attention and
            convolution layers are zipped and applied pairwise. However,
            if the number of convolution is lower than the
            number of attention layers not all attention layers are applied.
            We thus enumerate the attention layers and
            apply as many convolution layers as provided. If the number of
            convolution layers is larger than the number
            of attention layers, they are not applied.
        """
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(
            conv_layers) if conv_layers is not None else None
        self.normalization_layer = normalization_layer

    def forward(self, x, attention_mask=None):
        """
        Application of the encoder and collection of attention values.
        Attention and convolution layers are applied
        alternately. Optionally, a normalization layer is applied.
        """
        # collect attentions among attention layers
        attentions = []
        # Process attention and convolution layers
        # to be decided: for layer in self.layers would process any defined
        # layer, not robust against a mismatching
        # number of conv layers, code would be way shorter though
        for i, attn_layer in enumerate(self.attn_layers):
            x, attn = attn_layer(x,  attention_mask=attention_mask)
            attentions.append(attn)
            if self.conv_layers is not None and i < len(self.conv_layers):
                # Apply convolution layer if it exists for current index
                x = self.conv_layers[i](x)
        if self.normalization_layer is not None:
            x = self.normalization_layer(x)
        return x, attentions


class AutoformerDecoder(nn.Module):
    """
      The decoder takes a seasonal and trend initialization as well as the
      encoder output as input.
      In contrast to the encoder, a second Auto-Correlation block is applied
      using the keys and values of the encoder
      output.
          S1, T1 = Decomposition(Auto-Correlation(X) + X)
          S2, T2 = Decomposition(Auto-Correlation(S1) + S1)
          S3, T3 = Decomposition(Forward(S2) + S2)
          T = T0 + W1*T1 + W2*T2 + W3*T3
      """

    def __init__(self, layers, normalization_layer=None,
                 projection_layer=None):
        """
        Parameters:
            :param layers: A collection of all encoder layers.
            :param normalization_layer: Optional normalization layer.
            Defaults to None.
            :param projection_layer: Optional projection layer.
            Defaults to None.
        """
        super().__init__()
        self.decoder_layers = nn.ModuleList(layers)
        self.normalization_layer = normalization_layer
        self.projection_layer = projection_layer

    def forward(self, x, cross, x_mask, cross_mask, trend=None):
        """
        The decoder processes the trend in form of a residual block. Optionally,
        a normalization and a projection
        layer are applied.
        """

        for layer in self.decoder_layers:
            x, residual = layer(x, cross, x_mask, cross_mask)
            trend = trend + residual
        if self.normalization_layer:
            x = self.normalization_layer(x)
        # projection applied addition. to projection in single decoder layers
        if self.projection_layer:
            x = self.projection_layer(x)
        return x, trend


class Autoformer(nn.Module):
    """
    Main class of the Autoformer model with encoder and decoder.
    """

    def __init__(self, params) -> None:
        """
        Parameters:
            :param params (object): Collection of training and model-specific
            parameters. A description of all parameters can be
            found within the argument parser implementation
            (src.util.argument_parser.py).
        """
        super().__init__()
        self.seq_len = params.seq_len
        self.pred_len = params.pred_len
        self.label_len = params.seq_overlap
        self.window_size = params.window_size
        self.variates = params.num_variates
        self.output_feature_dim = params.output_feature_dim
        self.decomposition = SeriesDecomposition(self.window_size)
        self.frequency = params.freq
        self.dropout = params.dropout
        # set device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        # layer initialization
        self.encoder_embedding = PerformEmbedding(
            input_dim=params.enc_in, model_dim=params.model_dim,
            frequency=self.frequency, dropout=self.dropout)
        self.decoder_embedding = PerformEmbedding(input_dim=params.dec_in,
                                                  model_dim=params.model_dim)
        self.encoder = AutoformerEncoder([AutoformerEncoderLayer(
            auto_correlation=AttentionLayer(attention=AutoCorrelation(
                factor=params.factor,
                attention_dropout=params.attention_dropout),
                model_dim=params.model_dim, num_heads=params.num_heads),
            model_dim=params.model_dim,
            kernel_size=params.kernel_size,
            ff_dim=params.ff_dim,
            dropout=params.dropout, activation='relu')
            for _ in range(params.num_encoder_layers)],
            normalization_layer=NormalizationLayer(params.model_dim))
        self.decoder = AutoformerDecoder([AutoformerDecoderLayer(
            auto_correlation=AttentionLayer(
                attention=AutoCorrelation(factor=params.factor),
                model_dim=params.model_dim, num_heads=params.num_heads),
            cross_attention=AttentionLayer(
                attention=AutoCorrelation(factor=params.factor),
                model_dim=params.model_dim, num_heads=params.num_heads),
            model_dim=params.model_dim,
            output_feature_dim=params.output_feature_dim,
            kernel_size=params.kernel_size,
            ff_dim=params.ff_dim,
            dropout=params.dropout, activation='relu')
            for _ in range(params.num_decoder_layers)],
            normalization_layer=NormalizationLayer(params.model_dim),
            projection_layer=nn.Linear(params.model_dim,
                                       params.output_feature_dim, bias=True))

    def forward(self, x_enc: torch.Tensor, mark_enc: torch.Tensor,
                x_dec: torch.Tensor, mark_dec: torch.Tensor,
                dec_dec_mask=None, dec_enc_mask=None) -> torch.Tensor:
        """
        Processing of the Autoformer model.
        """
        if x_enc.dim() != 3:
            raise ValueError(
                f"Encoder input tensor must be 3-dimensional, "
                f"got {x_enc.dim()} dimensions")
        if x_dec.dim() != 3:
            raise ValueError(
                f"Decoder input tensor must be 3-dimensional, "
                f"got {x_dec.dim()} dimensions")
        # input shape is [batch, seq_len, variates]
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len,
                                                            1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]],
                            device=x_enc.device)
        # get seasonal and trend of the input data
        seasonal_init, trend_init = self.decomposition(x_enc)
        # pad the seasonal tensor with zeros of same length as prediction
        # length and trend part with mean of encoder
        # input  of same length as prediction length
        # trend_init and seasonal_init have total_len = label_len+pred_len
        trend_init = torch.cat(
            [trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat(
            [seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        # application of encoder embedding and encoding
        embedded_encoder = self.encoder_embedding(x=x_enc, mark=mark_enc)
        encoder_output, _ = self.encoder(x=embedded_encoder)
        # application of decoder embedding and decoding
        embedded_decoder = self.decoder_embedding(x=seasonal_init,
                                                  mark=mark_dec)
        seasonal_decoder_output, trend_decoder_output = self.decoder(
            x=embedded_decoder, cross=encoder_output,
            x_mask=dec_dec_mask, cross_mask=dec_enc_mask,
            trend=trend_init)
        # add seasonal and trend part
        final_output = seasonal_decoder_output + trend_decoder_output
        return final_output
