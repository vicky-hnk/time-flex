"""
This module contains positional encoding for time series forecasting
Transformers.
"""

import torch
from torch import nn


def sinusoidal_encoding(positional_encoding):
    """
    Generates sinusoidal positional encoding for a Pytorch tensor.
    ---
    Parameters:
    :param positional_encoding: An initialized tensor of shape
    [input_dim, model_dim].
    ---
    Returns:
    :return positional_encoding: A Pytorch tensor representing
    the sinusoidal encoding.
    """
    input_dim, model_dim = positional_encoding.size(0), positional_encoding.size(1)
    # k is position of an object in the input sequence, 0 <= k <= L/2
    k = (torch.arange(0, input_dim).float().unsqueeze(1) // 2).float()
    # calculate frequencies
    div_term = torch.pow(10000, 2 * k / model_dim)
    # apply sine to even indices and cosine to odd indices
    positional_encoding[0::2] = torch.sin(k[0::2] / div_term[0::2])
    positional_encoding[1::2] = torch.cos(k[1::2] / div_term[1::2])
    return positional_encoding


def tape_encoding(positional_encoding, seq_len):
    """
    Generates temporal absolute positional encoding for a Pytorch tensor -
    as introduced in:
     Foumani et al. Improving position encoding of transformers for
     multivariate time series classification.
     Data Min Knowl Disc 38, 22–48 (2024).
     https://doi.org/10.1007/s10618-023-00948-2.
     tAPE is an adaption of the classical sinusoidal encoding,
     with a different frequency calculation considering the
     sequence length of the input. It better captures temporal context.
    ---
    Parameters:
    :param positional_encoding: An initialized tensor of shape
    [input_dim, model_dim].
    :param seq_len: The sequence length of the input sequence.
    ---
    Returns:
    :return positional_encoding: A Pytorch tensor representing
    the tAPE encoding.
    """
    model_dim = positional_encoding.size(1)
    k = (torch.arange(0, seq_len).float().unsqueeze(1) // 2).float()
    div_term = torch.pow(10000, 2 * k / model_dim)
    positional_encoding[0::2] = torch.sin((k[0::2] / div_term[0::2]) *
                                          (model_dim / seq_len))
    positional_encoding[1::2] = torch.cos((k[1::2] / div_term[1::2]) *
                                          (model_dim / seq_len))
    return positional_encoding


class SinusoidalEncoding(nn.Module):
    """
    This class implements the classical sinusoidal positional encoding
    used for Transformer models.
    """

    def __init__(self, input_dim, model_dim):
        super().__init__()
        sinus_pos_encoding = torch.zeros(input_dim, model_dim).float()
        sinus_pos_encoding.require_grad = False
        sinus_pos_encoding = sinusoidal_encoding(sinus_pos_encoding)
        # Pytorch nn.Embedding is a LUT to store embeddings taking the
        # dictionary size and the size of embedding
        # vectors as arguments
        self.embedding = nn.Embedding(input_dim, model_dim)
        # setting of embedding (wrapped in parameter object)
        # hint: do not use weight.data directly if you do not put
        # requires_grad=False, this can lead to problems during gradient
        # computation since Autograd cannot track these direct operations
        self.embedding.weight = nn.Parameter(sinus_pos_encoding, False)

    def forward(self, x):
        self.embedding = self.embedding.to('cpu')
        # detach embedding from GPU since no gradient computation is needed
        return self.embedding(x).detach()


class TapeEncoding(nn.Module):
    """
    This class implements the time absolute positional encoding used for
    Transformer models. The transform logic is
    implemented in src.util.transforms_encoding.
    Todo: run test on tape encoding.
    """

    def __init__(self, model_dim, seq_len, input_dim):
        super().__init__()
        t_abs_pos_encoding = torch.zeros(seq_len, model_dim).float()
        t_abs_pos_encoding.require_grad = False
        t_abs_pos_encoding = tape_encoding(t_abs_pos_encoding, seq_len)
        self.embedding = nn.Embedding(input_dim, model_dim)
        self.embedding.weight = nn.Parameter(t_abs_pos_encoding, False)

    def forward(self, x):
        self.embedding = self.embedding.to('cpu')
        # detach embedding from GPU since no gradient computation is needed
        return self.embedding(x).detach()

