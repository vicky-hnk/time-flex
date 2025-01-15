"""
SparseTSF implementation from:

Period lengths:
    Electricity, ETTh, traffic 24
    ETTm, Weather 4
"""

import torch
import torch.nn as nn


class SparseTSF(nn.Module):
    """
    A PyTorch model for time series forecasting that utilizes 1D convolution
    to aggregate information across periods in the input sequence.
    The model performs down-sampling and sparse forecasting using
    a linear layer, and then performs up-sampling to return the output in the
    same format as the input.

    Attributes:
        seq_len (int): Length of the input sequence.
        pred_len (int): Length of the prediction/output sequence.
        enc_in (int): Number of input channels (input features per time step).
        period_len (int): Length of a period or segment used for down-sampling.
        seg_num_x (int): Number of segments in the input sequence after
        division by period_len.
        seg_num_y (int): Number of segments in the output sequence after
        division by period_len.
    """

    def __init__(self, params):
        """
        Initializes the model with the provided parameters.

        Args:
            params: Object containing various model hyperparameters, including:
                - seq_len (int): Length of the input sequence.
                - pred_len (int): Length of the prediction/output sequence.
                - enc_in (int): Number of input channels (input features per time step).
                - period_len (int): Length of each period or segment for down-sampling.
        """
        super().__init__()

        # Get parameters from the input
        self.seq_len = params.seq_len
        self.pred_len = params.pred_len
        self.enc_in = params.enc_in
        self.period_len = params.period_len

        # Calculate the number of segments in the input and output sequences
        # based on the period length
        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len

        # Aggregate features over the time sequence
        self.conv1d = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=1 + 2 * (self.period_len // 2),
            # Kernel size adjusted for period length
            stride=1,
            padding=self.period_len // 2,
            # Padding is based on half of the period length
            padding_mode="zeros",
            bias=False
        )
        self.linear = nn.Linear(self.seg_num_x, self.seg_num_y, bias=False)

    def forward(self, x):
        batch_size = x.shape[0]
        # Shape after normalization: (batch_size, enc_in, seq_len)
        seq_mean = torch.mean(x, dim=1).unsqueeze(1)
        x = (x - seq_mean).permute(0, 2, 1)
        # Apply 1D convolution to aggregate features across the sequence
        # Reshape x for convolution to (batch_size * enc_in, 1, seq_len)
        # Then, reshape it back to (batch_size, enc_in, seq_len)
        x = self.conv1d(x.reshape(
            -1, 1, self.seq_len)).reshape(-1, self.enc_in, self.seq_len) + x

        x = x.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)
        y = self.linear(x)
        y = y.permute(0, 2, 1).reshape(batch_size, self.enc_in, self.pred_len)
        # Add the sequence mean back and return the final prediction output
        return y.permute(0, 2, 1) + seq_mean
