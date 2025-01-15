"""
Implementation of TimeMixer modeĺ from:
https://github.com/kwuking/TimeMixer/tree/main
Decomposition either FFT-based or using moving average.
"""
import torch
import torch.nn as nn
from src.models.layers.decomp_layers import SeriesDecomposition
from src.models.layers.normalization import Normalize
from src.models.layers.embed import DataEmbedding_wo_pos


class DftSeriesDecomposition(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, top_k=5):
        super().__init__()
        self.top_k = top_k

    def forward(self, x):
        xf = torch.fft.rfft(x)
        freq = abs(xf)
        freq[0] = 0
        top_k_freq, top_list = torch.topk(freq, 5)
        xf[freq <= top_k_freq.min()] = 0
        x_season = torch.fft.irfft(xf)
        x_trend = x - x_season
        return x_season, x_trend


class MultiScaleSeasonMixing(nn.Module):
    """
    Bottom-up mixing season pattern
    """

    def __init__(self, params):
        super().__init__()

        self.down_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        params.seq_len // (params.down_sampling_window ** i),
                        params.seq_len // (
                                params.down_sampling_window ** (i + 1)),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        params.seq_len // (
                                params.down_sampling_window ** (i + 1)),
                        params.seq_len // (
                                params.down_sampling_window ** (i + 1)),
                    ),

                )
                for i in range(params.down_sampling_layers)
            ]
        )

    def forward(self, season_list):

        # mixing high->low
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]

        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))

        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    """
    Top-down mixing trend pattern
    """

    def __init__(self, params):
        super().__init__()

        self.up_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        params.seq_len // (
                                params.down_sampling_window ** (i + 1)),
                        params.seq_len // (params.down_sampling_window ** i),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        params.seq_len // (params.down_sampling_window ** i),
                        params.seq_len // (params.down_sampling_window ** i),
                    ),
                )
                for i in reversed(range(params.down_sampling_layers))
            ])

    def forward(self, trend_list):

        # mixing low->high
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        return out_trend_list


class PastDecomposableMixing(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.seq_len = params.seq_len
        self.pred_len = params.pred_len
        self.down_sampling_window = params.down_sampling_window

        self.layer_norm = nn.LayerNorm(params.model_dim)
        self.dropout = nn.Dropout(params.dropout)
        self.channel_independence = params.separate

        if params.decomp_method == 'moving_avg':
            self.decomposition = SeriesDecomposition(params.window_size)
        elif params.decomp_method == "dft_decomp":
            self.decomposition = DftSeriesDecomposition(params.top_k)
        else:
            raise ValueError('decomposition is error')

        if not params.separate:
            self.cross_layer = nn.Sequential(
                nn.Linear(in_features=params.model_dim,
                          out_features=params.ff_dim),
                nn.GELU(),
                nn.Linear(in_features=params.ff_dim,
                          out_features=params.model_dim),
            )

        # Mixing season
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(params)

        # Mixing trend
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(params)

        self.out_cross_layer = nn.Sequential(
            nn.Linear(in_features=params.model_dim,
                      out_features=params.ff_dim),
            nn.GELU(),
            nn.Linear(in_features=params.ff_dim,
                      out_features=params.model_dim),
        )

    def forward(self, x_list):
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)

        # Decompose to obtain the season and trend
        season_list = []
        trend_list = []
        for x in x_list:
            season, trend = self.decomposition(x)
            if self.channel_independence == 0:
                season = self.cross_layer(season)
                trend = self.cross_layer(trend)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))

        # bottom-up season mixing
        out_season_list = self.mixing_multi_scale_season(season_list)
        # top-down trend mixing
        out_trend_list = self.mixing_multi_scale_trend(trend_list)

        out_list = []
        for ori, out_season, out_trend, length in zip(x_list, out_season_list,
                                                      out_trend_list,
                                                      length_list):
            out = out_season + out_trend
            if self.channel_independence:
                out = ori + self.out_cross_layer(out)
            out_list.append(out[:, :length, :])
        return out_list


class TimeMixer(nn.Module):

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.task_name = params.task_name
        self.seq_len = params.seq_len
        self.label_len = params.seq_overlap
        self.pred_len = params.pred_len
        self.down_sampling_window = params.down_sampling_window
        self.channel_independence = params.separate
        print("Channel independence is set to ", self.channel_independence)
        self.pdm_blocks = nn.ModuleList([PastDecomposableMixing(params)
                                         for _ in range(params.num_layers)])

        self.preprocess = SeriesDecomposition(params.window_size)
        self.enc_in = params.enc_in
        self.use_future_temporal_feature = params.use_future_temporal_feature

        if self.channel_independence:
            self.enc_embedding = DataEmbedding_wo_pos(
                c_in=1, d_model=params.model_dim, freq=params.freq,
                dropout=params.dropout)
        else:
            self.enc_embedding = DataEmbedding_wo_pos(
                c_in=params.enc_in, d_model=params.model_dim, freq=params.freq,
                dropout=params.dropout)

        self.layer = params.num_layers
        if (self.task_name == 'long_term_forecast' or
                self.task_name == 'short_term_forecast'):
            self.predict_layers = torch.nn.ModuleList(
                [
                    torch.nn.Linear(
                        params.seq_len // (params.down_sampling_window ** i),
                        params.pred_len,
                    )
                    for i in range(params.down_sampling_layers + 1)
                ]
            )

            if self.channel_independence:
                self.projection_layer = nn.Linear(
                    params.model_dim, 1, bias=True)
            else:
                self.projection_layer = nn.Linear(
                    params.model_dim, params.output_feature_dim, bias=True)

                self.out_res_layers = torch.nn.ModuleList([
                    torch.nn.Linear(
                        params.seq_len // (params.down_sampling_window ** i),
                        params.seq_len // (params.down_sampling_window ** i),
                    )
                    for i in range(params.down_sampling_layers + 1)
                ])

                self.regression_layers = torch.nn.ModuleList(
                    [
                        torch.nn.Linear(
                            params.seq_len // (
                                    params.down_sampling_window ** i),
                            params.pred_len,
                        )
                        for i in range(params.down_sampling_layers + 1)
                    ]
                )

            self.normalize_layers = torch.nn.ModuleList(
                [
                    Normalize(self.params.enc_in, affine=True, non_norm=True
                    if params.use_norm == 0 else False)
                    for i in range(params.down_sampling_layers + 1)
                ]
            )

    def out_projection(self, dec_out, i, out_res):
        dec_out = self.projection_layer(dec_out)
        out_res = out_res.permute(0, 2, 1)
        out_res = self.out_res_layers[i](out_res)
        out_res = self.regression_layers[i](out_res).permute(0, 2, 1)
        dec_out = dec_out + out_res
        return dec_out

    def pre_enc(self, x_list):
        if self.channel_independence:
            return x_list, None
        else:
            out1_list = []
            out2_list = []
            for x in x_list:
                x_1, x_2 = self.preprocess(x)
                out1_list.append(x_1)
                out2_list.append(x_2)
            return out1_list, out2_list

    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
        """
        Remark:
        In the original implementation for pooling (AvgPool1d), the output size
        is computed using integer division, effectively flooring the result
        (e.g., 23 // 2 = 11).
        For slicing (::self.params.down_sampling_window), the step size
        includes the element at every self.params.down_sampling_window-th
        position, which can result in rounding up when the size is odd.
        Thus, the original implementation only works if the size is sampled
        down to an even number.
        """
        if self.params.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.params.down_sampling_window,
                                           return_indices=False)
        elif self.params.down_sampling_method == 'avg':
            down_pool = torch.nn.AvgPool1d(self.params.down_sampling_window)
        elif self.params.down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            down_pool = nn.Conv1d(in_channels=self.params.enc_in,
                                  out_channels=self.params.enc_in,
                                  kernel_size=3, padding=padding,
                                  stride=self.params.down_sampling_window,
                                  padding_mode='circular',
                                  bias=False)
        else:
            return x_enc, x_mark_enc
            # B,T,C -> B,C,T
        x_enc = x_enc.permute(0, 2, 1)

        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc

        x_enc_sampling_list = []
        x_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        x_mark_sampling_list.append(x_mark_enc)

        for i in range(self.params.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)

            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

            if x_mark_enc_mark_ori is not None:
                x_mark_sampling_list.append(
                    x_mark_enc_mark_ori[:, ::self.params.down_sampling_window,
                    :])
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:,
                                      ::self.params.down_sampling_window, :]

        x_enc = x_enc_sampling_list
        if x_mark_enc_mark_ori is not None:
            x_mark_enc = x_mark_sampling_list
        else:
            x_mark_enc = x_mark_enc

        return x_enc, x_mark_enc

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_future_temporal_feature:
            if self.channel_independence:
                B, T, N = x_enc.size()
                x_mark_dec = x_mark_dec.repeat(N, 1, 1)
                self.x_mark_dec = self.enc_embedding(None, x_mark_dec)
            else:
                self.x_mark_dec = self.enc_embedding(None, x_mark_dec)

        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc,
                                                              x_mark_enc)
        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                    x_mark = x_mark.repeat(N, 1, 1)
                x_list.append(x)
                x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x_enc)), x_enc):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)

        # embedding
        enc_out_list = []
        x_list = self.pre_enc(x_list)
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_list[0])), x_list[0],
                                    x_mark_list):
                enc_out = self.enc_embedding(x, x_mark)  # [B,T,C]
                enc_out_list.append(enc_out)
        else:
            for i, x in zip(range(len(x_list[0])), x_list[0]):
                enc_out = self.enc_embedding(x, None)  # [B,T,C]
                enc_out_list.append(enc_out)

        # Past Decomposable Mixing as encoder for past
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        # Future Multi-predictor Mixing as decoder for future
        dec_out_list = self.future_multi_mixing(B, enc_out_list, x_list)

        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        return dec_out

    def future_multi_mixing(self, B, enc_out_list, x_list):
        dec_out_list = []
        if self.channel_independence:
            x_list = x_list[0]
            for i, enc_out in zip(range(len(x_list)), enc_out_list):
                dec_out = self.predict_layers[i](
                    enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1)  # align temporal dimension
                if self.use_future_temporal_feature:
                    dec_out = dec_out + self.x_mark_dec
                    dec_out = self.projection_layer(dec_out)
                else:
                    dec_out = self.projection_layer(dec_out)
                dec_out = dec_out.reshape(B, self.params.output_feature_dim,
                                          self.pred_len).permute(0, 2,
                                                                 1).contiguous()
                dec_out_list.append(dec_out)

        else:
            for i, enc_out, out_res in zip(range(len(x_list[0])), enc_out_list,
                                           x_list[1]):
                dec_out = self.predict_layers[i](
                    enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1)  # align temporal dimension
                dec_out = self.out_projection(dec_out, i, out_res)
                dec_out_list.append(dec_out)

        return dec_out_list

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if (self.task_name == 'long_term_forecast'
                or self.task_name == 'short_term_forecast'):
            dec_out_list = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out_list
        else:
            raise ValueError('Only forecast tasks implemented yet')
