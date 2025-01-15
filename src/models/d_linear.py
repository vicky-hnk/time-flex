"""
Linear Network with decomposition layer
(Recreation of model architecture in: "Are Transformers Effective for
Time Series Forecasting?")

References :
    https://arxiv.org/pdf/2205.13504.pdf
    https://github.com/honeywell21/DLinear
"""

import torch
from src.models.layers.decomp_layers import SeriesDecomposition
from src.util.train_utils import set_seeds
from torch import nn


class DLinearModel(nn.Module):
    """
    Network with decomposition layer which splits the time series to trend
    and seasonal parts.
    If separate is true a fully connected layer is learned for every variate
    separately. Otherwise, on linear layer is
    learned for the seasonal and the trend part.
    """

    def __init__(self, params) -> None:
        """
         Parameters:
            :param params (object): Collection of training and model-specific
            parameters. A description of all parameters can be
            found within the argument parser implementation
            (src.util.argument_parser.py).
                - seq_len (int): The length of the input time series sequence.
                - pred_len (int): The length of the prediction horizon.
                - window_size (int): The size of the kernel used within
                decomposition.
                - num_variates (int): The number of variates of
                the time series.
                - separate (bool): Decides whether to use channel dependence
                or channel independence.
        """
        super().__init__()
        self.seq_len = params.seq_len
        self.pred_len = params.pred_len
        self.window_size = params.window_size
        self.variates = params.num_variates
        self.decomposition = SeriesDecomposition(self.window_size)
        self.separate = params.separate

        # set device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        if self.separate:
            self.linear_seasonal = nn.ModuleList()
            self.linear_trend = nn.ModuleList()

            for _ in range(self.variates):
                # separate linear layers for season and trend for each variate
                self.linear_seasonal.append(nn.Linear(
                    self.seq_len, self.pred_len))
                self.linear_trend.append(nn.Linear(
                    self.seq_len, self.pred_len))

        else:
            self.linear_seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.linear_trend = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        """
        Applies seasonal decomposition and a linear layer/ multiple linear
        layers, if separate is true.
        """
        if x.dim() != 3:
            raise ValueError(f"Input tensor must be 3-dimensional, "
                             f"got {x.dim()} dimensions")
        # input shape is [batch, seq_len, variates]
        season_input, trend_input = self.decomposition(x)
        season_input, trend_input = (season_input.permute(0, 2, 1),
                                     trend_input.permute(0, 2, 1))
        season_output, trend_output = None, None

        # Process each variable separately
        if self.separate:
            season_output = torch.zeros(
                [season_input.size(0),
                 season_input.size(1), self.pred_len],
                dtype=season_input.dtype).to(self.device)
            trend_output = torch.zeros(
                [trend_input.size(0), trend_input.size(1), self.pred_len],
                dtype=trend_input.dtype).to(self.device)

            for i in range(self.variates):
                season_output[:, i, :] = self.linear_seasonal[i](
                    season_input[:, i, :])
                trend_output[:, i, :] = self.linear_trend[i](
                    trend_input[:, i, :])

        # Process shared linear layers
        else:
            season_output = self.linear_seasonal(season_input)
            trend_output = self.linear_trend(trend_input)

        x = season_output + trend_output
        return x.permute(0, 2, 1)  # to [batch, pred_len, variates]
