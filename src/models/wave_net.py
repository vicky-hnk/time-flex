"""
The module contains a time series forecasting WaveNet model that implements
the WaveNet architecture originally developed for signal processing:
https://arxiv.org/abs/1609.03499
The model can be applied on integer or complex Pytorch tensors.
"""

import torch
import torch.nn.functional as func
from src.models.layers.complex_layers import (ComplexAdaptiveAvgPool1d,
                                              ComplexConv1d, ComplexDropout,
                                              relu_complex, sigmoid_complex,
                                              tanh_complex)
from src.util.train_utils import set_seeds
from torch import nn


class BaseWaveNet(nn.Module):
    """Dilated Convolutional Neural Network architecture for multi-variate
    time series.
        The original architecture was designed for signal processing using
        Wavelet transform to encode the signal.
        This implementation uses a seq2seq-similar approach.
        Parameters:
            :param param.num_filters: Number of filters in the convolutional
            layers, i.e. the dimension of the output
            space. For model consistency the number of filters for residual
            and dilated convolutional layers are
            equal.
            :param param.kernel_size: size of convolution kernel
            :param param.num_dilated_layers: Number of hidden layers with
            dilated convolution. For each layer a
            convolution rate is calculated as follows: 2**i with i being the
            number of the layer. Since the
            convolution rate defines how inputs shall be skipped using base 2
            to calculate it results in an
            exponentially increasing rate. This limits the increase of
            computational complexity with the rise in the
            number of layers.
        Remarks:
            Pytorch Conv1D takes the following input and output dimension:
            (N, C, L) with N=batch size, C= number of
            channels (i.e. number of features) and L = sequence length.
    """

    def __init__(self, params, tanh_activation, sigmoid_activation,
                 relu_activation, complex_bool: bool = False):
        super().__init__()
        # Initialize shared attributes
        self.seq_len = params.seq_len
        self.pred_len = params.pred_len
        self.num_filters = params.num_filters
        self.skip_channels = params.skip_channels
        self.kernel_size = params.kernel_size
        # if params.separate features are processed separately,
        # thus one WaveNet block per feature
        if params.separate:
            self.num_in_features = 1
            self.out_features = 1
        # if features processed together: define whether uni or
        # multi variate prediction
        else:
            self.num_in_features = params.num_variates
            if params.features == 'S' or params.features == 'MS':
                self.out_features = 1
            elif params.features == 'M':
                self.out_features = self.num_in_features
            else:
                raise ValueError('valid features are strings: M,S and MS')
        # calculate dilation rates
        self.num_dilated_layers = params.num_dilated_layers
        self.dilation_rates = [2 ** i for i in range(self.num_dilated_layers)]
        self.dropout = params.dropout
        if self.num_dilated_layers <= 1 and self.dropout > 0:
            print("WARNING: Dropout option adds dropout after all but last "
                  "recurrent layer, so non-zero dropout expects"
                  "num_layers greater than 1.")
        # set device
        self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        # initialize activation functions
        self.tanh = tanh_activation
        self.sigmoid = sigmoid_activation
        self.relu = relu_activation
        # call factory method to create layers
        self.Conv1d = self.layer_factory('conv1d',
                                         complex_layers=complex_bool)
        self.Dropout = self.layer_factory('dropout',
                                          complex_layers=complex_bool)
        self.AdaptiveAvgPool1d = self.layer_factory(
            'adaptive_avg_pool1d', complex_layers=complex_bool)
        # initialize layers of residual blocks
        self.start_convolution = self.Conv1d(in_channels=self.num_in_features,
                                             out_channels=self.num_filters,
                                             kernel_size=1)
        # create dilated convolution layers
        self.filter_convolutions = nn.ModuleList([
            self.Conv1d(in_channels=self.num_filters,
                        out_channels=self.num_filters,
                        kernel_size=self.kernel_size,
                        padding=0, dilation=dilation_rate)
            for dilation_rate in self.dilation_rates
        ])
        self.gate_convolutions = nn.ModuleList([
            self.Conv1d(in_channels=self.num_filters,
                        out_channels=self.num_filters,
                        kernel_size=self.kernel_size,
                        padding=0, dilation=dilation_rate)
            for dilation_rate in self.dilation_rates
        ])
        # create residual and skip convolution layers
        self.residual_convolutions = nn.ModuleList([
            self.Conv1d(in_channels=self.num_filters,
                        out_channels=self.num_filters,
                        kernel_size=1)
            for _ in self.dilation_rates
        ])
        self.skip_convolutions = nn.ModuleList([
            self.Conv1d(in_channels=self.num_filters,
                        out_channels=self.skip_channels,
                        kernel_size=1)
            for _ in self.dilation_rates
        ])
        # initialize output layers
        self.adaptive_pool = self.AdaptiveAvgPool1d(self.pred_len)
        self.conv1d_dense = self.Conv1d(
            in_channels=self.skip_channels,
            out_channels=2 ** self.num_dilated_layers,
            kernel_size=1, padding=0)
        self.dropout_layer = self.Dropout(self.dropout)
        self.final_conv1d = self.Conv1d(
            in_channels=2 ** self.num_dilated_layers,
            out_channels=self.out_features,
            kernel_size=1, padding=0)

    @staticmethod
    def layer_factory(layer_type, complex_layers=False):
        if complex_layers:
            if layer_type == 'conv1d':
                return ComplexConv1d
            elif layer_type == 'dropout':
                return ComplexDropout
            elif layer_type == 'adaptive_avg_pool1d':
                return ComplexAdaptiveAvgPool1d
        else:
            if layer_type == 'conv1d':
                return nn.Conv1d
            elif layer_type == 'dropout':
                return nn.Dropout
            elif layer_type == 'adaptive_avg_pool1d':
                return nn.AdaptiveAvgPool1d

    @staticmethod
    def initial_dim_check(tensor):
        if tensor.dim() != 3:
            raise ValueError(
                f"Input tensor must be 3-dimensional, "
                f"got {tensor.dim()} dimensions")
        # input dimension is [batch, seq len, num_variates/channels]
        # but Pytorch Conv1D takes [batch size, channels, seq length]
        tensor = tensor.permute(0, 2, 1)
        return tensor

    def forward(self, x):
        x = self.initial_dim_check(x)
        len_diff = self.pred_len - self.seq_len
        # Pad x if the prediction length is greater than the sequence length
        # Early padding to ensure that all subsequent layers work with a
        # consistently sized input.
        # Possible drawback is feature dilution (zero-padding introduces areas
        # in the input that are empty,
        # convolution over these areas might lead to diluted feature).
        if len_diff > 0:
            x = func.pad(x, (0, len_diff))
        x = self.start_convolution(x)
        # dimension is  [batch size, num of filters, prediction length]
        skip = 0
        for i, dilation_rate in enumerate(self.dilation_rates):
            # Calculate padding for causality
            pad = (self.kernel_size - 1) * dilation_rate
            x_padded = func.pad(x, (pad, 0))  # Apply causal padding
            # perform gated activation unit (GAU)
            x_filter = self.filter_convolutions[i](x_padded)
            # dimension is [1, number of filters, prediction length]
            x_gated = self.gate_convolutions[i](x_padded)
            # dimension is [1, number of filters, prediction length]
            tanh_out = self.tanh(x_filter)
            sigmoid_out = self.sigmoid(x_gated)
            gau = sigmoid_out * tanh_out
            # append skip connections
            skip_out = self.skip_convolutions[i](gau)
            # dimension is [1, number of filters, prediction length]
            skip = skip + skip_out
            # add residual part as input for next layer
            # if desired add additional convolution
            # residual_out = self.residual_convolutions[i](gau)
            # x = x + residual_out
            x = x + gau
        # application of output layers
        skip = self.adaptive_pool(skip)
        out = self.relu(skip)
        out = self.conv1d_dense(out)
        # dimension is [1, 2** number of dilated layers, prediction length]
        out = self.relu(out)
        out = self.dropout_layer(out)
        out_seq = self.final_conv1d(out)
        # dimension is [batch, number of output features, pred_len]
        # dim is [batch, pred_len, num of out-features] after! permutation
        return out_seq.permute(0, 2, 1)


class WaveNet(BaseWaveNet):
    def __init__(self, params):
        super().__init__(params, tanh_activation=torch.tanh,
                         sigmoid_activation=torch.sigmoid,
                         relu_activation=func.relu, complex_bool=False)


class ComplexWaveNet(BaseWaveNet):
    def __init__(self, params):
        """
        Complex version of the WaveNet model implemented above.
        The data type of the input tensor is complex float.
        """
        super().__init__(params, tanh_activation=tanh_complex,
                         sigmoid_activation=sigmoid_complex,
                         relu_activation=relu_complex, complex_bool=True)
