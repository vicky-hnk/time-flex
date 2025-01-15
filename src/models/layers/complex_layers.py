"""
This module provides a collection of modules, layers and functions for complex tensors.
"""
import torch
from torch import nn
import torch.nn.functional as func


# ---Activation functions---

def sigmoid_complex(x):
    """Compute the sigmoid function of the complex tensors."""
    return torch.complex(torch.sigmoid(x.real), torch.sigmoid(x.imag))


def relu_complex(x):
    """Compute the relu function of the complex tensors."""
    return torch.complex(torch.relu(x.real), torch.relu(x.imag))


def tanh_complex(x):
    """Compute the tanh function of the complex tensors."""
    return torch.complex(torch.tanh(x.real), torch.tanh(x.imag))


# ---Other functions---

def dropout_func_complex(x, dropout=0.2, inplace: bool = False):
    """
    Applies dropout to the real and imaginary parts of a complex
    tensor separately.

    Parameters:
    - x: Input tensor of a complex dtype (e.g., torch.complex64).
    - p: Dropout rate.
    - inplace: If `True`, modify the input tensor in-place
    (not recommended for complex tensors).

    Returns:
    - A tensor with the same shape as the input, with dropout applied.
    """

    if not inplace:
        x = x.clone()
    mask = func.dropout(torch.ones_like(x.real), p=dropout)
    # Apply the same mask to both real and imaginary parts
    x_real = x.real * mask
    x_imag = x.imag * mask
    return torch.complex(x_real, x_imag)


# ---Layers and Models---

class ComplexLinearLayer(nn.Module):
    """
    This module implements a linear layer for complex input and output.
    ---
    Remark:
    Complex multiplication is applied: (a+bi)*(c+di) = (ac-bd) + i(ad+bc)
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear_real = nn.Linear(in_features, out_features)
        self.linear_imaginary = nn.Linear(in_features, out_features)

    def forward(self, x):
        real_part = (self.linear_real(x.real) - self.linear_imaginary(x.imag))
        imag_part = (self.linear_real(x.imag) - self.linear_imaginary(x.real))
        return torch.complex(real_part, imag_part)


class ComplexConv1d(nn.Module):
    """
        This module implements a 1D Convolutional layer for complex input
        and output.
        ---
        Remark:
        Complex multiplication is applied: (a+bi)*(c+di) = (ac-bd) + i(ad+bc)
        """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0,
                 dilation: int = 1, groups: int = 1, bias: bool = True):
        super().__init__()
        self.real_conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation, groups, bias)
        self.imag_conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation, groups, bias)

    def forward(self, x):
        real_part = self.real_conv(x.real) - self.imag_conv(x.imag)
        imag_part = self.real_conv(x.imag) + self.imag_conv(x.real)
        return torch.complex(real_part, imag_part)


class ComplexDropout(nn.Module):
    """
    This module implements a dropout layer for complex input and output.
    Applies dropout only during training, as does the original Pytorch
    Dropout Module.
    ---
    Remark:
    Complex multiplication is applied: (a+bi)*(c+di) = (ac-bd) + i(ad+bc)
    """

    def __init__(self, dropout: float = 0.2, inplace: bool = False):
        """
        :param dropout: Dropout probability.
        :param inplace: Whether to perform the dropout operation inplace.
        """
        super().__init__()
        self.p = dropout
        self.inplace = inplace

    def forward(self, x):
        if self.training:
            # Generate a dropout mask.
            mask = func.dropout(torch.ones_like(x.real), self.p,
                                self.training, self.inplace)
            real = x.real * mask
            imag = x.imag * mask
            return torch.complex(real, imag)
        return x


class ComplexAdaptiveAvgPool1d(nn.Module):
    """
    This module implements a 1D Adaptive Average Pooling layer for
    complex inputs.
    It separates the real and imaginary parts of the input, applies adaptive
    average pooling
    to each part independently, and then recombines them into a complex output.
    """

    def __init__(self, output_size: int):
        """
        :param output_size: Size of the pooling output.
        """
        super().__init__()
        self.output_size = output_size
        self.real_pool = nn.AdaptiveAvgPool1d(output_size)
        self.imag_pool = nn.AdaptiveAvgPool1d(output_size)

    def forward(self, x):
        real = self.real_pool(x.real)
        imag = self.imag_pool(x.imag)
        return torch.complex(real, imag)


class ComplexAdaptiveAvgPool2d(nn.Module):
    """
    This module implements a 1D Adaptive Average Pooling layer for
    complex inputs.
    It separates the real and imaginary parts of the input, applies adaptive
    average pooling
    to each part independently, and then recombines them into a complex output.
    """

    def __init__(self, output_size: int):
        """
        :param output_size: Size of the pooling output, i.e. a tuple HxW.
        """
        super().__init__()
        self.output_size = output_size
        self.real_pool = nn.AdaptiveAvgPool2d(output_size)
        self.imag_pool = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, x):
        real = self.real_pool(x.real)
        imag = self.imag_pool(x.imag)
        return torch.complex(real, imag)
