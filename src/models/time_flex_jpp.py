"""
This module implements the TimeFlex Network variant with joint post-processing
after the weighted sum. See time_flex.py for the original implementation of
TimeFlex.
"""
import torch
from torch import nn
import torch.nn.functional as func

from src.models.layers.decomp_layers import (SeriesDecomposition,
                                             MultiScaleDecomposition)
from src.models.layers.normalization import RevIN
from src.models.layers.complex_layers import (ComplexConv1d,
                                              ComplexAdaptiveAvgPool2d,
                                              relu_complex, sigmoid_complex,
                                              tanh_complex)


class LinearBlock(nn.Module):
    """
    Linear Layers to process the trend.
    """

    def __init__(self, params):
        super().__init__()
        self.pred_len = params.pred_len
        self.variates = params.num_variates

        # set device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.linear_trend = nn.ModuleList(
            [nn.Linear(params.seq_len, self.pred_len)
             for _ in range(self.variates)])

    def forward(self, x):
        trend_output = torch.zeros([x.size(0), x.size(1), self.pred_len],
                                   dtype=x.dtype).to(self.device)
        for i in range(self.variates):
            trend_output[:, i, :] = self.linear_trend[i](x[:, i, :])
        return trend_output


class DilatedConvolutions(nn.Module):
    """
    Dilated Convolutional Neural Network for Multi-variate
    Time Series Forecasting.
    This class implements a dilated convolutional neural network (CNN)
    designed for multi-variate time series forecasting.

    Parameters:
        :param params.kernel_size: Size of the convolutional kernel.
        :param params.pred_len:  Prediction length, i.e., the number of future
        time steps the model is expected to forecast.
        :param params.seq_len: Sequence length, i.e., the number of past time
        steps used as input to the model.
        :param params.num_variates : Number of variates (features) in the input
        time series data.
        :param params.num_dilated_layers : Number of hidden layers with dilated
        convolutions. The dilation rate for each layer is calculated as 2**i,
        with 'i' being the layer number. This results in an exponentially
        increasing dilation rate, which helps to capture long-term dependencies
        without a significant increase in computational complexity.
    Remarks:
        PyTorch Conv1D takes the following input and output dimensions:
        - Input: (N, C, L) where N is the batch size, C is the number of
        channels (i.e., number of features), and L is the sequence length.
        - Output: (N, C_out, L_out) where C_out is the number of output
        channels (filters).
    """

    def __init__(self, params):
        super().__init__()
        self.kernel_size = params.kernel_size
        self.pred_len = params.pred_len
        self.seq_len = params.seq_len
        self.variates = params.num_variates
        self.num_dilated_layers = params.num_dilated_layers
        self.dilation_rates = [2 ** i for i in range(self.num_dilated_layers)]

        # set device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # initialize activation & layers
        if params.transform == 'FFT' or params.transform == '2dFFT':
            self.tanh = tanh_complex
            self.sigmoid = sigmoid_complex
            self.relu = relu_complex
        else:
            self.tanh = torch.tanh
            self.sigmoid = torch.sigmoid
            self.relu = torch.relu
        # initialize layers
        self._init_layers(params)

    def _init_layers(self, params):
        """Initialize dilated conv layers."""
        if params.transform == 'FFT' or params.transform == '2dFFT':
            self.Conv1d = ComplexConv1d
            self.AdaptiveAvgPool2d = ComplexAdaptiveAvgPool2d
        else:
            self.Conv1d = nn.Conv1d
            self.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d

        def create_conv1d(in_channels, out_channels, kernel_size,
                          dilation_rate=1, padding=0):
            return self.Conv1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, dilation=dilation_rate,
                               padding=padding)

        def create_conv1d_layers(num_layers, in_channels, out_channels,
                                 kernel_size, dilation_rates, padding=0):
            return nn.ModuleList([
                nn.ModuleList([
                    create_conv1d(in_channels, out_channels, kernel_size,
                                  dilation_rate, padding)
                    for dilation_rate in dilation_rates
                ])
                for _ in range(num_layers)
            ])

        self.start_convolution = create_conv1d(
            in_channels=1, out_channels=params.num_filters, kernel_size=1)

        self.filter_convolutions = create_conv1d_layers(
            params.num_variates, params.num_filters, params.num_filters,
            self.kernel_size, self.dilation_rates)

        self.gate_convolutions = create_conv1d_layers(
            params.num_variates, params.num_filters, params.num_filters,
            self.kernel_size, self.dilation_rates)

        self.skip_convolutions = create_conv1d_layers(
            params.num_variates, params.num_filters, params.num_filters,
            1, [1] * len(self.dilation_rates))

        self.residual_convolutions = create_conv1d_layers(
            params.num_variates, params.num_filters, params.num_filters,
            1, [1] * len(self.dilation_rates))

        self.adaptive_pool = self.AdaptiveAvgPool2d((1, self.pred_len))

    def forward(self, x):
        len_diff = self.pred_len - self.seq_len
        if len_diff > 0:
            x = func.pad(x, (0, len_diff))
        variate_skips = []
        for i in range(self.variates):
            variate_input = x[:, i, :].unsqueeze(1)
            # [batch, 1, pred len]
            variate_input = self.start_convolution(variate_input)
            skip_outputs = []
            for j, dilation_rate in enumerate(self.dilation_rates):
                # Calculate padding for causality
                pad = (self.kernel_size - 1) * dilation_rate
                # Apply causal padding
                x_padded = func.pad(variate_input, (pad, 0))
                # perform gated activation unit (GAU)
                x_filter = self.filter_convolutions[i][j](x_padded)
                x_gated = self.gate_convolutions[i][j](x_padded)
                # dim [batch, num_filters, pred_len]
                tanh_out = self.tanh(x_filter)
                sigmoid_out = self.sigmoid(x_gated)
                gau = sigmoid_out * tanh_out
                # Append skip connections
                skip_out = self.skip_convolutions[i][j](gau)
                skip_outputs.append(skip_out)
                # Add residual part as input for next layer
                # if desired: add 1d convolution to residual
                # residual_out = self.residual_convolutions[i][j](gau)
                # variate_input = variate_input + residual_out
                variate_input = variate_input + gau
            skip_tensor = torch.stack(skip_outputs, dim=0)  # [l, b, f, pred]
            skip_tensor = skip_tensor.permute(1, 0, 2, 3)
            skip_tensor_pooled = self.adaptive_pool(skip_tensor)
            skip_tensor_pooled = skip_tensor_pooled.squeeze()  # [b, l, pred]
            variate_skips.append(skip_tensor_pooled)
        out = torch.stack(variate_skips, dim=0)
        return out  # [vars, batch, layers, pred_len]


class EnhanceLayers(nn.Module):
    def __init__(self, params):
        super().__init__()

        # set device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        features_map = {'S': 1, 'MS': 1, 'M': params.num_variates}
        self.out_features = features_map.get(params.features, 1)

        # Set activation function
        self.activation = func.gelu if (
                params.activation == 'gelu') else func.relu

        # initialize layers
        self.conv1d_dense = nn.Conv1d(
            in_channels=params.num_variates,
            out_channels=2 ** params.num_dilated_layers,
            kernel_size=1, padding=0)
        self.dropout_layer = nn.Dropout(params.dropout)
        self.final_conv1d = nn.Conv1d(
            in_channels=2 ** params.num_dilated_layers,
            out_channels=self.out_features,
            kernel_size=1, padding=0)

    def forward(self, x):
        x = self.activation(x)
        x = self.conv1d_dense(x)
        # dimension is [1, 2** number of dilated layers, prediction length]
        x = self.activation(x)
        out = self.dropout_layer(x)
        return self.final_conv1d(out)  # dim is [batch, out_features, pred_len]


class TimeFlexJPP(nn.Module):
    """
    This model processes trend and season separately as well as the features
    (i.e. channel independence).
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
                input series. Valid options are 'FFT', 'DWT', or 'None'.
                - num_variates (int): The number of vars in the input series.
                """
        super().__init__()

        # initialize parameters
        self.seq_len = params.seq_len
        self.pred_len = params.pred_len
        self.num_filters = params.num_filters
        self.skip_channels = params.skip_channels
        self.window_size = params.window_size
        self.kernel_size = params.kernel_size
        self.num_dilated_layers = params.num_dilated_layers
        self.transform = params.transform
        self.variates = params.num_variates
        self.decomp = params.decomp_method
        self.scales = params.scales

        # set device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # initialize layers and weights
        self._init_weights(params)
        self._reset_parameters()
        self._init_layers(params)

    def _init_weights(self, params):
        """Initialize model weights."""
        self.weights_season = nn.Parameter(
            torch.Tensor(params.num_variates, params.num_dilated_layers),
            requires_grad=True)
        self.weights_trend = nn.Parameter(torch.Tensor(1, params.num_variates),
                                          requires_grad=True)

    def _reset_parameters(self):
        """Uses kaiming initialization to reset weights."""
        nn.init.kaiming_normal_(self.weights_season, mode='fan_out',
                                nonlinearity='relu')
        nn.init.kaiming_normal_(self.weights_trend, mode='fan_out',
                                nonlinearity='relu')

    def _init_layers(self, params):
        """Initialize model layers."""
        print("Revin application:", params.use_revin)
        self.revin = RevIN(
            num_features=params.num_variates) if params.use_revin else None
        self.linear_block = LinearBlock(params)
        self.dilated_convolutions = DilatedConvolutions(params)
        self.final_enhance = EnhanceLayers(params)
        if self.decomp == 'moving_avg':
            self.decomposition = SeriesDecomposition(self.window_size)
        elif self.decomp == 'multi_scale':
            self.decomposition = MultiScaleDecomposition(self.scales)
        else:
            raise ValueError("Unknown decomposition- valid options: "
                             "'moving_avg', 'multi_scale'.")

    def _create_output_tensors(self, season_out, trend_out):
        """Initialize output tensors."""
        weighted_season = torch.zeros(season_out.shape[:4]).to(self.device)
        weighted_trend = torch.zeros(trend_out.shape[:3]).to(self.device)
        summed_season = torch.zeros(season_out.shape[:3]).to(self.device)
        return weighted_season, weighted_trend, summed_season

    @staticmethod
    def _ensure_4d_tensor(tensor):
        """Ensure the tensor has 4 dimensions.
        (during testing: only one batch)"""
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(1)
        tensor = tensor.permute(1, 0, 3, 2)
        if tensor.dim() != 4:
            raise ValueError(
                f"Expected tensor to have 4 dimensions, "
                f"but got {tensor.dim()}")
        return tensor

    def forward_season(self, x):
        """Forward method for seasonal part."""
        # apply transformation
        if self.transform == 'FFT':
            x = torch.fft.fft(x)
        elif self.transform == '2dFFT':
            x = torch.fft.fft2(x)
        else:
            pass
        # apply dilated convolutions
        season_processed = self.dilated_convolutions(x)
        # inverse transformation
        if self.transform == 'FFT':
            season_processed = torch.fft.ifft(season_processed).real
        elif self.transform == '2dFFT':
            season_processed = torch.fft.ifft2(season_processed).real
        else:
            pass
        return season_processed

    def forward_trend(self, x):
        """Forward method for trend part."""
        trend_processed = self.linear_block(x)
        return trend_processed

    def forward(self, x):
        # RevIn if desired
        x = self.revin(x, 'norm') if self.revin else x

        # Series Decomposition to TREND and SEASON
        season_input, trend_input = self.decomposition(x)
        season_input, trend_input = (season_input.permute(0, 2, 1),
                                         trend_input.permute(0, 2, 1))
        # process TREND and SEASON at single scale
        trend_output = self.forward_trend(trend_input)
        season_output = self.forward_season(season_input)
        season_output = self._ensure_4d_tensor(season_output)

        # WEIGHTED SUM for tensors of dim [batch, vars, pred_len]
        weighted_season, weighted_trend, summed_season = (
            self._create_output_tensors(season_output, trend_output))
        for i in range(self.variates):
            for j in range(self.num_dilated_layers):
                weighted_season[:, i, :, j] = (self.weights_season[i, j] *
                                               season_output[:, i, :, j])
            # [batch, vars, pred_len, layers]
            summed_season = torch.sum(weighted_season, dim=3, keepdim=False)
            # [batch, vars, pred_len]
            weighted_trend[:, i, :] = (self.weights_trend[:, i] *
                                       trend_output[:, i, :])
        sum_output = summed_season + weighted_trend  # [batch, vars, pred_len]

        # POSTPROCESSING with activation and 1d convolution
        out = self.final_enhance(sum_output)
        out = out.permute(0, 2, 1)  # to [batch, pred_len, variates]
        out = self.revin(out, 'denorm') if self.revin else out
        return out
