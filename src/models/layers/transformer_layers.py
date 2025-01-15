""" Module implementing different transformer layers."""
import torch
import torch.nn as nn
import torch.nn.functional as func
from src.models.layers.decomp_layers import SeriesDecomposition


class EncoderLayer(nn.Module):
    """Encoder layer for a transformer-based model."""

    def __init__(self, attention, model_dim, ff_dim=None,
                 dropout: float = 0.1, activation: str = "relu"):
        super().__init__()
        ff_dim = ff_dim or 4 * model_dim
        self.attention = attention
        self.conv_layer_1 = nn.Conv1d(in_channels=model_dim,
                                      out_channels=ff_dim,
                                      kernel_size=1)
        self.conv_layer_2 = nn.Conv1d(in_channels=ff_dim,
                                      out_channels=model_dim,
                                      kernel_size=1)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = func.relu if activation == "relu" else func.gelu

    def forward(self, x, attention_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(x, x, x, attention_mask=attention_mask,
                                     tau=tau, delta=delta)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(
            self.activation(self.conv_layer_1(y.transpose(-1, 1))))
        y = self.dropout(self.conv_layer_2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class AutoformerEncoderLayer(nn.Module):
    """
    Implementation of single encoder layer using the following processing:
    - auto-correlation as attention mechanism with residual connection
    - decomposition layer
    - feedforward block with 2 convolutional layers and residual connection
    - decomposition layer
    """

    def __init__(self, auto_correlation, model_dim, kernel_size: int = 25,
                 ff_dim=None, dropout: float = 0.1,
                 activation: str = 'relu'):
        """
        :param auto_correlation: defines the auto-correlation calculation
        block that shall be used
        :param model_dim: dimension of the input token
        :param kernel_size: kernel sie for decomposition layer, not! the kernel
        size of convolutions in feedforward
        :param ff_dim: dimension of the feedforward layer, in
        SotA-implementations usually set to 4 times
        the dimension of the model_dim
        :param dropout: dropout rate for feedforward (default is 0.1)
        :param activation: activation function of feedforward block
        (default is relu)
        """
        super().__init__()
        self.auto_correlation = auto_correlation
        self.ff_dim = ff_dim if ff_dim else 4 * model_dim
        self.decomposition_layer_1 = SeriesDecomposition(kernel_size)
        self.decomposition_layer_2 = SeriesDecomposition(kernel_size)
        # bias in convolutions is set to false, normalization is applied
        # afterwards, which subtracts the mean and thus
        # nullifies bias b (y=Wx+b)
        self.conv_layer_1 = nn.Conv1d(in_channels=model_dim,
                                      out_channels=self.ff_dim, kernel_size=1,
                                      bias=False)
        self.conv_layer_2 = nn.Conv1d(in_channels=self.ff_dim,
                                      out_channels=model_dim, kernel_size=1,
                                      bias=False)
        self.dropout = nn.Dropout(dropout)
        if activation == 'relu':
            self.activation = func.relu
        elif activation == 'leaky_relu':
            self.activation = func.leaky_relu
        else:
            self.activation = func.gelu

    def forward(self, x, attention_mask=None, tau=None, delta=None):
        x_corr, attn = self.auto_correlation(x, x, x)
        # first residual
        # dim of x_corr is [batch, seq_len, model_dim]
        x = x + self.dropout(x_corr)
        # first decomposition
        x, _ = self.decomposition_layer_1(x)
        x_forward = x
        # feedforward block
        x_forward = self.dropout(self.activation(
            self.conv_layer_1(x_forward.transpose(-1, 1))))
        # resulting dim [batch, dim_ff, seq_len]
        x_forward = self.dropout(self.conv_layer_2(x_forward).transpose(-1, 1))
        # resulting dim [batch, seq_len, model_dim]
        # decomposition of the second residual
        x_out, _ = self.decomposition_layer_2(x + x_forward)
        return x_out, attn


class Encoder(nn.Module):
    """Encoder for a Transformer model."""

    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) \
            if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x):
        # x [B, L, D]
        attentions = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers,
                                              self.conv_layers):
                x, attn = attn_layer(x)
                x = conv_layer(x)
                attentions.append(attn)
            x, attn = self.attn_layers[-1](x)
            attentions.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x)
                attentions.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attentions


class AutoformerDecoderLayer(nn.Module):
    """
    Implements a decoder layer for the Autoformer architecture.
    ---
    Parameters:
        :param auto_correlation: A module for computing auto-correlation.
        :param cross_attention: A module for computing cross-attention
        between inputs.
        :param model_dim: The dimensionality of the model (input features).
        :param output_feature_dim: The dimensionality of the output features.
        :param kernel_size: The kernel size for the series decomposition layers.
        Defaults to 25.
        :param ff_dim: The dimensionality of the feedforward network,
        defaults to 4 times model_dim.
        :param dropout: Dropout rate for regularization. Defaults to 0.1.
        :param activation: The type of activation function to use
        ('relu', 'leaky_relu', or 'gelu'). Defaults to 'relu'.
    ---
    Inputs:
        - x: The input tensor for self-attention.
        - cross: The input tensor for cross-attention.
        - x_mask: Optional mask for the input tensor. Defaults to None.
        - cross_mask: Optional mask for the cross-attention tensor.
        Defaults to None.
    ---
    Returns:
        : return: A tuple containing the transformed input and the aggregated
        residual trend components.
        """

    def __init__(self, auto_correlation, cross_attention, model_dim,
                 output_feature_dim, kernel_size: int = 25,
                 ff_dim=None, dropout: float = 0.1,
                 activation: str = 'relu'):
        super().__init__()
        self.auto_corr = auto_correlation
        self.cross_attention = cross_attention
        self.ff_dim = ff_dim if ff_dim else 4 * model_dim
        self.decomposition_layer_1 = SeriesDecomposition(kernel_size)
        self.decomposition_layer_2 = SeriesDecomposition(kernel_size)
        self.decomposition_layer_3 = SeriesDecomposition(kernel_size)
        # bias in convolutions is set to false, normalization is applied
        # afterwards, which subtracts the mean and thus
        # nullifies bias b (y=Wx+b)
        self.conv_layer_1 = nn.Conv1d(in_channels=model_dim,
                                      out_channels=self.ff_dim, kernel_size=1,
                                      bias=False)
        self.conv_layer_2 = nn.Conv1d(in_channels=self.ff_dim,
                                      out_channels=model_dim, kernel_size=1,
                                      bias=False)
        self.dec_projection_layer = nn.Conv1d(in_channels=model_dim,
                                              out_channels=output_feature_dim,
                                              kernel_size=3, stride=1,
                                              padding=1,
                                              padding_mode='circular',
                                              bias=False)
        self.dropout = nn.Dropout(dropout)
        if activation == 'relu':
            self.activation = func.relu
        elif activation == 'leaky_relu':
            self.activation = func.leaky_relu
        else:
            self.activation = func.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x_corr1 = self.auto_corr(x, cross, cross, attention_mask=x_mask)
        # first residual
        x = x + self.dropout(x_corr1[0])
        x, trend1 = self.decomposition_layer_1(x)
        x_corr2 = self.auto_corr(x, cross, cross, attention_mask=cross_mask)
        x = x + self.dropout(x_corr2[0])
        x, trend2 = self.decomposition_layer_2(x)
        x_forward = x
        x_forward = self.dropout(self.activation(
            self.conv_layer_1(x_forward.transpose(-1, 1))))
        x_forward = self.dropout(self.conv_layer_2(
            x_forward).transpose(-1, 1))
        x, trend3 = self.decomposition_layer_3(x + x_forward)
        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.dec_projection_layer(
            residual_trend.permute(0, 2, 1)).transpose(1, 2)
        # dim is [batch, label+pred length, variates]
        return x, residual_trend


class Decoder(nn.Module):
    """
    Autoformer decoder.
    """

    def __init__(self, layers, norm_layer=None, projection=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask=x_mask,
                                      cross_mask=cross_mask)
            trend = trend + residual_trend

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x, trend


class LayerNorm(nn.Module):
    """
    Special designed layer norm for the seasonal part.
    """

    def __init__(self, channels):
        super().__init__()
        self.layer_norm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layer_norm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1],
                                                            1)
        return x_hat - bias
