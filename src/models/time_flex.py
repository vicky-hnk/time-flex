"""
Decomposition Dilated Convolution Model for multi-variate time series
forcasting.
The model architecture is composed of the following blocks:
    * A decomposition block which calculates the moving average to split the
    time series to a trend/ average part
    and a seasonal part. This block was originally introduced in
    https://arxiv.org/pdf/2205.13504.pdf and used in
    https://arxiv.org/pdf/2106.13008.pdf.
    * The seasonal part of the time series is processed in a dilated
    convolution block as introduced in
    https://arxiv.org/pdf/1609.03499.pdf. The original model was used
    for audio signals but adapts well to time series
    forcasting.
    * The postprocessing block is similar to the originally one in the WaveNet
    publication.
    * A weighted sum combines trend and season of all features.
If separate is true:
    To capture the trend of every feature separately, a feedforward block
    with one linear layer per feature is
    applied. The same is done for the seasonal part
    (one "WaveNet" block per feature).
    This separation is especially useful for time series with the following
    characteristics.
* The time series of the different feature have very distinct patterns
    (e.g. environmental temperature has a strong yearly pattern while traffic
    has a strong weekly pattern).
* The correlation between variates is relatively low.
* The influence of variates is unknown which can be learned using the weighted
sum function.
---
Possible use-case specific adaptions:
* Time series transformation
    * FFT for strong emphasis on periodic patterns (high frequency resolution,
    suited for stationary data).
    * No transform to reduce complexity.
"""

import torch
from torch import nn

from src.models.layers.decomp_layers import SeriesDecomposition
from src.models.wave_net import WaveNet, ComplexWaveNet
from src.models.layers.normalization import RevIN


class TimeFlex(nn.Module):
    """
    The TimeFlex class is a PyTorch module for multi-variate forecasting
    supporting different transformations
    (either Fast Fourier Transform (FFT) or no transformation).
    This module can be configured to process multiple variates separately
    or together, and it allows for the use of e on the input data.

    """

    def __init__(self, params):
        """
        Parameters:
            :param params (object): Collection of training and model-specific
            parameters. A description of all parameters can be found within
            the argument parser implementation (src.util.argument_parser.py).
                - seq_len (int): The length of the input time series sequence.
                - pred_len (int): The length of the prediction horizon.
                - num_filters (int): The number of filters to use in the
                convolutional layers.
                - skip_channels (int): The number of skip channels in the
                network architecture.
                - kernel_size (int): The size of the kernel in the
                convolutional layers.
                - window_size (int): The size of the kernel used within
                decomposition.
                - separate (bool): Flag to determine if the variates should be
                processed separately.
                - transform (str): The type of transformation to apply to the
                input series. Valid options are 'FFT' or 'None'.
                - num_variates (int): The number of vars in the input series.

        Attributes:
            - decomposition (SeriesDecomposition): Module for initial
            series decomposition.
            - linear_trend (nn.Linear or nn.ModuleList): Linear layer(s)
            for trend prediction.
            - dilated_convolutions (WaveNet/ComplexWaveNet or nn.ModuleList):
            Convolutional network(s) for series
            decomposition and prediction.
        """
        super().__init__()
        self.seq_len = params.seq_len
        self.pred_len = params.pred_len
        self.num_filters = params.num_filters
        self.skip_channels = params.skip_channels
        self.window_size = params.window_size
        self.kernel_size = params.kernel_size
        self.separate = params.separate
        self.transform = params.transform
        self.variates = params.num_variates
        # set device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print("Revin application:", params.use_revin)
        self.revin = RevIN(
            num_features=params.num_variates) if params.use_revin else None
        # initialize layers
        self.decomposition = SeriesDecomposition(self.window_size)
        # if separate initialize one layer/block per variate
        if self.separate:
            # Initialize linear trends for each variate
            self.linear_trend = nn.ModuleList(
                [nn.Linear(self.seq_len, self.pred_len)
                 for _ in range(self.variates)])

            # Initialize WaveNet or ComplexWaveNet based on transform type
            self.dilated_convolutions = nn.ModuleList([
                ComplexWaveNet(params=params) if self.transform in ['FFT',
                                                                    '2dFFT']
                else WaveNet(params=params)
                for _ in range(self.variates)
            ])

        # process all variates together
        else:
            self.linear_trend = nn.Linear(self.seq_len, self.pred_len)
            if self.transform == 'FFT' or self.transform == '2dFFT':
                self.dilated_convolutions = ComplexWaveNet(params=params)
            else:
                self.dilated_convolutions = WaveNet(params=params)
        # Initialize learnable weights for the sum
        self.weight_season = nn.Parameter(torch.ones(1), requires_grad=True)
        self.weight_trend = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):
        """
        Hint:
            Applying the Fourier transform the Tensor contains complex floats.
            Most Pytorch methods are not
            compatible with complex floats. src.models.layers.complex_layers.py
            contains an implementation of common layers and activation
            functions for complex float tensors.
        """
        # RevIn if desired
        x = self.revin(x, 'norm') if self.revin else x
        # DECOMPOSE
        season_input, trend_input = self.decomposition(x)
        season_input, trend_input = (season_input.permute(0, 2, 1),
                                     trend_input.permute(0, 2, 1))
        # TREND PART
        trend_output = torch.zeros([trend_input.size(0), trend_input.size(1),
                                    self.pred_len],
                                   dtype=trend_input.dtype).to(self.device)
        if self.separate:
            for i in range(self.variates):
                trend_output[:, i, :] = self.linear_trend[i](
                    trend_input[:, i, :])
        else:
            trend_output = self.linear_trend(trend_input)
        # SEASONAL PART
        # [batch, variates, seq length]
        if self.transform == 'FFT':
            season_input = torch.fft.fft(season_input, dim=-1)
        elif self.transform == '2dFFT':
            season_input = torch.fft.fft2(season_input)
        else:
            pass
        season_output = torch.zeros([season_input.size(0),
                                     season_input.size(1), self.pred_len],
                                    dtype=season_input.dtype).to(self.device)
        if self.separate:
            for i in range(self.variates):
                season_output[:, i, :] = self.dilated_convolutions[i](
                    season_input[:, i, :].unsqueeze(-1)).squeeze(-1)
        else:
            season_output = self.dilated_convolutions(season_input)
        # inverse transformation if applied
        if self.transform == 'FFT':
            season_output = torch.fft.ifft(season_output).real
        elif self.transform == '2dFFT':
            season_output = torch.fft.ifft2(season_output).real
        else:
            pass
        # WEIGHTED SUM
        out = (self.weight_season * season_output + self.weight_trend *
               trend_output)
        out = out.permute(0, 2, 1)  # to [batch, pred_len, variates]
        out = self.revin(out, 'denorm') if self.revin else out
        return out
