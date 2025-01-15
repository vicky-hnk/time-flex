"""
This file contains several methods applying transformations and encodings.
"""
import torch
from pytorch_wavelets import DWT1DForward, DWT1DInverse

# see: https://github.com/fbcotter/pytorch_wavelets


def get_fft2(x: torch.Tensor) -> torch.Tensor:
    """
    Applies Fast Fourier Transform to a PyTorch tensor.
    """
    return torch.fft.fft2(x)


def get_inverse_fft2(x: torch.Tensor) -> torch.Tensor:
    """
    Applies inverse Fast Fourier Transform to a PyTorch tensor.
    """
    return torch.fft.ifft2(x)


def get_dwt(x: torch.Tensor, j: int = 3, wavelet: str = 'db1'):
    """
    Applies Discrete Wavelet Transform to a PyTorch tensor.
    ---
    Parameters:
    :param x: PyTorch tensor
    :param wavelet: A str indicating the wavelet type.
        - Daubechies Wavelets (dbN): N denotes the order, Higher order
        Daubechies wavelets are smoother and have more.
    :param j: parameter in the DWTForward class specifying the number of
    levels of decomposition that will be performed
        on the input data. Each level of decomposition breaks down the signal
        into further approximation and detail
        coefficients, allowing for multi-resolution analysis of the signal.
        - The mode parameter specifies how the signal is extended at its
        boundaries during the convolution with
        wavelet filters (padding with zeros is a common way).

    Returns:
    :return: Wavelet coefficients.
    """
    dwt = DWT1DForward(J=j, mode='zero', wave=wavelet)
    dwt.cuda()
    coefficients_low, coefficients_high = dwt(x.unsqueeze(0))
    print("Test DWT dimensions", coefficients_low, coefficients_high)
    return coefficients_low, coefficients_high


def inverse_dwt(coefficients_low, coefficients_high, wavelet: str = 'db1'):
    """
    Calculate the inverse wavelet transform.
    """
    i_dwt = DWT1DInverse(mode='zero', wave=wavelet)
    i_dwt.cuda()
    reconstructed_x = i_dwt(coefficients_low, coefficients_high)
    return reconstructed_x.squeeze(0)
