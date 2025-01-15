import torch
from torch import nn
import math

from src.models.layers.encoding import SinusoidalEncoding


class TemporalEmbedding(nn.Module):
    """
    This class implements the temporal embedding used in AutoFormer
    (NeurIPS 2021).
    We do not embed seconds and years. The embedding type is sinusoidal.
    """

    def __init__(self, model_dim: int, frequency: str,
                 embedding='SinusoidalEncoding'):
        super().__init__()
        # define embedding type
        if embedding == 'SinusoidalEncoding':
            self.embedding = SinusoidalEncoding
        elif embedding == 'fixed':
            self.embedding = FixedEmbedding
        else:
            print(f'Unrecognized embedding {embedding}.')
            raise NotImplementedError
        # define temporal embedding sizes
        minute_size = 4  # hour divided to quarters
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13
        self.model_dim = model_dim
        self.frequency = frequency
        # initialize embedding
        if self.frequency in {'min', 'sec'}:
            self.minute_embed = self.embedding(minute_size, self.model_dim)
        self.hour_embed = self.embedding(hour_size, self.model_dim)
        self.weekday_embed = self.embedding(weekday_size, self.model_dim)
        self.day_embed = self.embedding(day_size, self.model_dim)
        self.month_embed = self.embedding(month_size, self.model_dim)

    def forward(self, x):
        # input indices shall be integer types (int64)
        x = x.long()
        # x should have dimension [batch, features, temporal feature]
        # Note: embeds minutes 0, 15, 30, 45 as 0, 1, 2, 3
        minute_x = self.minute_embed(x[:, :, 5] // 15) \
            if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 4])
        weekday_x = self.weekday_embed(x[:, :, 3])
        day_x = self.day_embed(x[:, :, 2])
        month_x = self.month_embed(x[:, :, 1])
        return hour_x + weekday_x + day_x + month_x + minute_x


class TokenEmbedding(nn.Module):
    """
    Embedding using Conv1D as "token-window" to capture local context with
    fewer parameters than a dense layer.
    Circular padding is applied to maintain periodicity.
    """

    def __init__(self, input_size, model_dim):
        super().__init__()
        self.input_size = input_size
        self.model_dim = model_dim
        self.token_conv = nn.Conv1d(in_channels=self.input_size,
                                    out_channels=self.model_dim,
                                    kernel_size=3, padding=1,
                                    padding_mode='circular', bias=False)
        # in case of nested submodules the following loops ensures all
        # convolutional layers are initialized properly
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_in',
                                        nonlinearity='leaky_relu')

    def forward(self, x):
        # Conv1D expects input shape [batch_size, variates, length]
        # but initial shape is [batch_size, length, variates]
        x = self.token_conv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x)


class TimeFeatureEmbedding(nn.Module):
    """
    Hint: data with freq h, min, D, W are encoded with
    [year, month, weekday, day, hour, minute]
    data with freq sec are encoded with
    [year, month, weekday, day, hour, minute, second]
    """

    def __init__(self, model_dim, freq='h'):
        super().__init__()

        freq_map = {'W': 6, 'D': 6, 'h': 6, 'min': 6, 'sec': 7}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, model_dim, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h',
                 dropout=0.1):
        super().__init__()

        self.value_embedding = TokenEmbedding(c_in, d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(
            model_dim=d_model, embedding=embed_type, frequency=freq) \
            if embed_type != 'timeF' else TimeFeatureEmbedding(
            model_dim=d_model, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        current_device = next(self.parameters()).device
        if x is None and x_mark is not None:
            x_mark = x_mark.to(current_device)
            return self.temporal_embedding(x_mark)
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x_mark = x_mark.to(current_device)
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class PerformEmbedding(nn.Module):
    """
    Uses token-embedding for input sequence tensor and temporal_embedding
    for corresponding marks.
    ---
    Parameters:
    :param input_dim: Dimensionality of the input sequence tensor.
    :param model_dim: Dimensionality of the output embedding vector,
    which typically matches
      the dimensionality expected by subsequent model layers.
    :param frequency: The granularity of the temporal encoding.
    :param dropout: Dropout rate applied to the final output.
    :param embedding_type: The type of temporal embedding to use.
    """

    def __init__(self, input_dim: int, model_dim: int, frequency: str = 'min',
                 dropout: float = 0.1,
                 embedding_type='SinusoidalEncoding'):
        super().__init__()
        self.frequency = frequency
        self.token_embedding = TokenEmbedding(input_size=input_dim,
                                              model_dim=model_dim)
        self.temporal_embedding = TemporalEmbedding(model_dim=model_dim,
                                                    frequency=self.frequency,
                                                    embedding=embedding_type)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, mark: torch.Tensor):
        x = self.token_embedding(x) + self.temporal_embedding(mark).to('cuda')
        return self.dropout(x)


class DataEmbedding(nn.Module):
    def __init__(self, input_size, model_dim, embed_type='fixed', freq='h',
                 dropout=0.1):
        super().__init__()

        self.value_embedding = TokenEmbedding(input_size=input_size,
                                              model_dim=model_dim)
        self.position_embedding = PositionalEmbedding(d_model=model_dim)
        self.temporal_embedding = TemporalEmbedding(
            model_dim=model_dim, embedding=embed_type, frequency=freq) \
            if embed_type != 'timeF' else TimeFeatureEmbedding(
            model_dim=model_dim, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        current_device = next(
            self.parameters()).device
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x_mark = x_mark.to(
                current_device)  # Move x_mark to the current device
            x = self.value_embedding(x) + self.temporal_embedding(
                x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbeddingInverted(nn.Module):
    """Normal embedding class for vanilla Transformer."""

    def __init__(self, c_in: int, model_dim: int, dropout: float = 0.1):
        super().__init__()
        self.value_embedding = nn.Linear(c_in, model_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)  # x: [batch, variate, time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x_mark = x_mark.to(x.device)
            # the potential to take co-variates (e.g. timestamps) as tokens
            x = self.value_embedding(torch.cat(
                [x, x_mark.permute(0, 2, 1)], 1))
        # x has dim [batch, variate, model dim]
        return self.dropout(x)


class DataEmbedding_inverted_vc(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h',
                 dropout=0.1):
        super().__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            # if not time feature
            x = self.value_embedding(x)
        else:
            # if have time feature
            x_mark = x_mark.to(x.device)
            x = self.value_embedding(
                torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a
        # d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars
