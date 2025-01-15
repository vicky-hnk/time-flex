"""
Module contains a collection of reusable decomposition blocks.
"""
import torch
from torch import nn


class MoveAverage(nn.Module):
    """
    Implementation similar to: https://github.com/thuml/Autoformer
    Slightly modified because original implementation is only applicable
    in case of odd kernel sizes.
    """

    def __init__(self, kernel_size, stride: int = 1) -> None:
        super().__init__()
        # average over kernel size, default stride is one to avoid info loss
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride,
                                padding=0)
        self.pad_front = (self.kernel_size - 1) // 2  # Front padding
        self.pad_end = (self.kernel_size - 1) - self.pad_front  # End padding

    def forward(self, x):
        # padding on the both ends of time series to use "full window size"
        front = x[:, 0:1, :].repeat(1, self.pad_front, 1)
        end = x[:, -1:, :].repeat(1, self.pad_end, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))  # to [Batch, Variates, Input length]
        x = x.permute(0, 2, 1)
        return x


class SeriesDecomposition(nn.Module):
    """
    Decomposes a time series into season and trend by calculating the mean
    over a sliding window.
    """

    def __init__(self, kernel_size, stride: int = 1) -> None:
        super().__init__()
        self.stride = stride
        self.moving_avg = MoveAverage(kernel_size, stride=self.stride)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class MultiScaleDecomposition(nn.Module):
    def __init__(self, scales, combi_method: str = 'sum', stride: int = 1,
                 combine=True):
        super().__init__()
        self.method = combi_method
        self.window_sizes = scales
        self.num_scales = len(scales)
        self.combine = combine
        self.moving_avg_blocks = nn.ModuleList(
            [MoveAverage(kernel, stride=stride) for kernel in scales])

        initial_weights = torch.rand(self.num_scales)
        self.weights = torch.nn.Parameter(
            initial_weights / initial_weights.sum())

    def forward(self, x):
        trend_parts = []
        seasonal_parts = []

        for block in self.moving_avg_blocks:
            moving_avg = block(x)
            trend_parts.append(moving_avg.unsqueeze(0))
            seasonal_parts.append((x - moving_avg).unsqueeze(0))

        trend_parts = torch.cat(trend_parts, dim=0)
        seasonal_parts = torch.cat(seasonal_parts, dim=0)

        if self.combine:
            if self.method == 'mean':
                combined_trend = torch.mean(trend_parts, dim=0)
            elif self.method == 'sum':
                combined_trend = torch.sum(trend_parts, dim=0)
            else:
                raise ValueError("Unsupported method. Use 'mean' or 'sum'.")

            combined_residual = x - combined_trend
            return combined_residual, combined_trend
        else:
            # Return all components separately
            return seasonal_parts, trend_parts
