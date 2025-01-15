"""
Module contains a collection of attention layers.
"""
import numpy as np
import torch
from torch import nn


class AttentionLayer(nn.Module):
    """
    This layer projects inputs into separate subspaces for queries, keys,
    and values to perform multi-head
    attention/auto-correlation, followed by aggregating the outputs back
    to the model's dimension.
    ---
    Parameters:
    :param attention: A callable function or module that computes the
    attention/auto-correlation.
    :param model_dim: The dimensionality of the input and output of the model.
    :param num_heads: The number of attention heads for multi-head attention.
    :param keys_dim: The dimensionality of each key. Defaults to
    model_dim // num_heads.
    :param values_dim: The dimensionality of each value. Defaults to
    model_dim // num_heads.
    ---
    Inputs:
    - queries, keys, values: The input tensors for the auto-correlation
    computation.
    - attention_mask: A mask for the attention mechanism.
    ---
    Outputs:
    - out, attn: The output tensor after applying auto-correlation and
    attention aggregation,
    and the attention weights tensor.
    """

    def __init__(self, attention, model_dim: int, num_heads: int,
                 keys_dim=None, values_dim=None):
        """
        :param attention: A callable function or module that computes the
        attention/auto-correlation.
        :param model_dim: The dimensionality of the input and output.
        :param num_heads: The number of attention heads for multi-head
        attention.
        :param keys_dim: The dimensionality of each key tensor.
        :param values_dim: The dimensionality of each value tensor.
        """
        super().__init__()
        self.attention = attention

        # split whole input dimension to multiple separate layers
        # (num_heads defines the number of layers)
        keys_dim = keys_dim or (model_dim // num_heads)
        values_dim = values_dim or (model_dim // num_heads)

        # project input dimensions to dimension of keys, queries and values
        self.query_projection = nn.Linear(model_dim, keys_dim * num_heads)
        self.key_projection = nn.Linear(model_dim, keys_dim * num_heads)
        self.value_projection = nn.Linear(model_dim, values_dim * num_heads)

        # project output of all heads to model dimension
        self.out_projection = nn.Linear(values_dim * num_heads, model_dim)
        self.num_heads = num_heads

    def forward(self, queries, keys, values, attention_mask=None, tau=None,
                delta=None):
        batch_size, query_seq_len, _ = queries.shape
        _, key_seq_len, _ = keys.shape
        # reshape Q, K and V for multi-head attention
        # (distribute on multiple heads)
        # --> seq_length x (model_dim/num_heads)
        queries = self.query_projection(queries).reshape(
            batch_size, query_seq_len, self.num_heads, -1)
        keys = self.key_projection(keys).reshape(
            batch_size, key_seq_len, self.num_heads, -1)
        values = self.value_projection(values).reshape(
            batch_size, key_seq_len, self.num_heads, -1)

        out, attn = self.attention(
            queries, keys, values, attention_mask, tau, delta)
        out = out.reshape(batch_size, query_seq_len, -1)
        return self.out_projection(out), attn


class AutoCorrelation(nn.Module):
    """
    Calculates the auto-correlation for Autoformer model.
    ---
    Parameters:
        :param factor: higher factor results in a larger top_k
        :param attention_dropout: Dropout rate after attention layer.
    """

    def __init__(self, factor: int = 1, attention_dropout: float = 0.1,
                 output_attention=False):
        """
        :param factor: higher factor results in a larger top_k.
        :param attention_dropout: Dropout rate after attention layer.
        :param output_attention: Whether to return the attention.
        """
        super().__init__()
        self.factor = factor
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def time_delay_agg_training(self, values, correlation):
        """
        Auto-Correlation with batch-normalization for the training phase.
        Dimension of input values is [batch, head, model_dim/heads, seq_len]
        """
        head, variates, length = (values.shape[1], values.shape[2],
                                  values.shape[3])
        # find top k period lengths
        top_k = int(self.factor * np.log(length))
        # calculate mean along 2 dimensions
        mean_value = torch.mean(torch.mean(correlation, dim=1), dim=1)
        # extract k highest values (torch.topk returns top values [0]
        # and top indices [1])
        # correspond to the k periods with highest correlation values
        topk_index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[
            1]
        correlation_weights = torch.stack([mean_value[:, topk_index[i]]
                                           for i in range(top_k)], dim=-1)
        # normalize correlation weights
        temp_correlation = torch.softmax(correlation_weights, dim=-1)
        # aggregation, dimension of temp_values is
        # [batch, head, model_dim/heads, seq_len]
        temp_values = values
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            # shift elements in temp_values beyond last position and
            # re-introduced at first position
            # number of shifted places is the (negative) current top k index
            # -> align period with sequence start
            pattern = torch.roll(temp_values, -int(topk_index[i]), -1)
            unsqueezed_correlation = (
                temp_correlation[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).
                repeat(1, head, variates, length))
            # multiply period with correlation weights
            delays_agg = delays_agg + pattern * unsqueezed_correlation
        return delays_agg

    def time_delay_agg_inference(self, values, correlation):
        batch, head, variates, length = values.shape[0], values.shape[1], \
        values.shape[2], values.shape[3]
        # initialize index tensor with length of values along dimensions
        # of batch, head and channel/variates
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(
            0) \
            .repeat(batch, head, variates, 1).to(values.device)
        # find top k and normalize correlation
        top_k = int(self.factor * np.log(length))
        mean_value = torch.mean(torch.mean(correlation, dim=1), dim=1)
        weights, delay_index = torch.topk(mean_value, top_k, dim=-1)
        temp_correlation = torch.softmax(weights, dim=-1)
        # aggregation
        # double size of the last dimension by replicating its elements once
        temp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            # add delay index to shift original tensor
            temp_delay = (
                        init_index + delay_index[:, i].unsqueeze(1).unsqueeze(
                    1).unsqueeze(1).
                        repeat(1, head, variates, length))
            pattern = torch.gather(temp_values, dim=-1, index=temp_delay)
            unsqueezed_correlation = (
                temp_correlation[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).
                repeat(1, head, variates, length))
            # multiply period with correlation weights
            delays_agg = delays_agg + pattern * unsqueezed_correlation
        return delays_agg

    def forward(self, queries, keys, values, attention_mask=None, tau=None,
                delta=None):
        # attention masking not implemented
        _, query_seq_len, _, _ = queries.shape
        _, value_seq_len, _, _ = values.shape
        if query_seq_len > value_seq_len:
            # padding with zeros if dimension of Q is larger than the
            # dimension of K and V
            zeros = torch.zeros_like(
                queries[:, :(query_seq_len - value_seq_len), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            # cut dimension of K and V if larger than dimension of Q
            values = values[:, :query_seq_len, :, :]
            keys = keys[:, :query_seq_len, :, :]

        # computes 1-dimensional fast fourier transform of real input,
        # outputs the non-negative frequency components.
        # ensure the fourier transforms are stored as single,
        # uninterrupted block of memory (contiguous)
        query_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(),
                                   dim=-1)
        key_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        # auto-correlation of discrete-time signal is original signal times
        # conjugate of lagged signal
        corr_result = query_fft * torch.conj(key_fft)
        # inverse of the fft
        correlation = torch.fft.irfft(corr_result, n=query_seq_len, dim=-1)
        # time delay aggregation (different for training and inference)
        if self.training:
            out_values = self.time_delay_agg_training(
                values.permute(0, 2, 3, 1).contiguous(),
                correlation).permute(0, 3, 1, 2)
        else:
            out_values = self.time_delay_agg_inference(
                values.permute(0, 2, 3, 1).contiguous(), correlation)
            out_values = out_values.permute(0, 3, 1, 2)

        if self.output_attention:
            return out_values.contiguous(), correlation.permute(0, 3, 1, 2)

        return out_values.contiguous(), None


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1,
                 output_attention=False):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values,  attention_mask=None, tau=None,
                delta=None):
        batch_size, query_seq_len, _, attn_head_dim = queries.shape
        scale = self.scale or 1. / torch.sqrt(attn_head_dim)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attention_mask is None:
                attention_mask = TriangularCausalMask(batch_size,
                                                      query_seq_len,
                                                      device=queries.device)
            # weight zero if mask value True
            scores.masked_fill_(attention_mask.mask, -np.inf)

        out_attention = self.dropout(torch.softmax(scale * scores, dim=-1))
        output = torch.einsum("bhls,bshd->blhd", out_attention, values)

        if self.output_attention:
            return output.contiguous(), out_attention
        else:
            return output.contiguous(), None


class AutoCorrelationLayer(nn.Module):
    def __init__(self, correlation, d_model, n_heads, d_keys=None,
                 d_values=None):
        super().__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attention_mask=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_correlation(
            queries,
            keys,
            values,
            attention_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
    

class TriangularCausalMask:
    def __init__(self, batch_size, length, device='cpu'):
        mask_shape = [batch_size, 1, length, length]
        with torch.no_grad():
            # boolean mask with upper triangular being true and lower
            # being false
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool),
                                    diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask
