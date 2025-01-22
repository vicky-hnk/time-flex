"""
The modules is a collection of utility functions that do not directly belong to the training and logging procedure
when running experiments.
"""
import matplotlib.pyplot as plt
import matplotlib.dates as mpld
import pandas as pd


class Attribs:
    """
    Holds the training and data attributes in a way that is compatible with
    an argument parser to avoid the necessity
    of code adaption when using either a config json file or parse the config
    from command line.
    """
    def __init__(self, data=None):
        """Updates object attributes with key-value pairs from a dict."""
        self.data = None
        self.data_source = None
        self.data_path = None
        self.root_path = None
        self.data_root_path = None
        self.num_workers = None
        self.target = None
        self.time_enc = None
        self.separate = None
        self.pred_len = None
        self.data_overlap = None
        self.freq = None
        self.seq_len = None
        self.features = None
        self.seq_overlap = None
        self.batch_size = None
        self.num_variates = None
        self.train_procedure = None
        self.data_split = None
        self.stat_split = None
        self.scale = None
        self.hidden_size = None
        self.num_layers = None
        self.shuffle = None
        self.num_workers = None
        self.learning_rate = None
        self.loss = None
        self.optimizer = None
        self.scheduler_patience = None
        self.scheduler_factor = None
        self.kernel_size = None
        self.window_size = None
        self.dropout = None
        self.attention_dropout = None
        self.num_filters = None
        self.skip_channels = None
        self.num_dilated_layers = None
        self.num_heads = None
        self.model_dim = None
        self.factor = None
        self.enc_in = None
        self.dec_in = None
        self.c_tide_out = None
        self.output_feature_dim = None
        self.num_encoder_layers = None
        self.num_decoder_layers = None
        self.ff_dim = None
        self.conv_dim = None
        self.expand = None
        self.transform = None
        self.output_attention = None
        self.use_norm = None
        self.activation = None
        self.stop_patience = None
        self.stop_delta = None
        self.use_revin = None
        self.down_sampling_layers = None
        self.down_sampling_window = None
        self.down_sampling_method = None
        self.use_future_temporal_feature = None
        self.task_name = None
        self.decomp_method = None
        self.scales = None
        self.top_k = None
        self.embed = None
        self.snap_size = None
        self.proj_dim = None
        self.hidden_dim = None
        self.hidden_layers = None
        self.period_len = None
        self.model_type = None
        self.cycle = None
        self.aug_type = None
        self.aug_rate = None
        self.sampling_rate = None
        self.rate = None
        self.wavelet = None
        self.n_imf = None
        self.level = None
        self.percentage = None
        self.n_imf = None
        self.random_seed = None

        if data:
            self.__dict__.update(data)
