""" Argument parser."""
import argparse


def parse_args():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Experiments on multivar. time series forecasting-data- '
                    'and training-related arguments.')

    parser.add_argument('--random_seed', type=int, default=3,
                        help='random seed')

    # Data Settings
    parser.add_argument('--data', type=str, default=None,
                        help='Name of dataset to use. Should be according to '
                             'datasets in the dataset dictionaries.')
    parser.add_argument('--root_path', type=str, default=None,
                        help='Root path of repository.')
    parser.add_argument('--data_source', type=str,
                        default="local_csv",
                        help='Defines where to load the data.',
                        choices=['local_csv'])
    parser.add_argument('--features', type=str, default='MS',
                        help='Multi-variate or uni-variate features.',
                        choices=['S', 'MS', 'M'])
    parser.add_argument('--time_enc', type=int, default=0,
                        help='0=No special encoding.')
    parser.add_argument('--data_split', nargs=3,
                        default=[0.7, 0.2, 0.1],
                        help='Percentage of training, validation and '
                             'testing data.')
    parser.add_argument('--stat_split', type=float, default=0.6,
                        help='Percentage of training data.')
    parser.add_argument('--scale', type=str, default="Standard",
                        help='Scaler type, False means no scaling.',
                        choices=['Standard', 'MinMax', 'False'])
    parser.add_argument('--seq_len', type=int, default=192,
                        help='Sequence length training.')
    parser.add_argument('--seq_overlap', type=int, default=0,
                        help='Seq. overlap of input and pred. for training.')
    parser.add_argument('--pred_len', type=int, default=92,
                        help='Length of predicted sequence.')
    parser.add_argument('--data_overlap', type=int, default=0,
                        help='Length of overlap of training & val. datasets.')

    # ----------
    # Training settings
    parser.add_argument('--train_procedure', type=str,
                        default="batch", choices=['seq2seq', 'batch'],
                        help='Defines whether data shall be split '
                             'sequentially or to shuffled batches.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float,
                        default=0.0001, help='Initial learning rate.')
    parser.add_argument('--epochs', type=int, default=25,
                        help='Number of training epochs.')
    parser.add_argument('--shuffle', type=bool, default=True,
                        help='Defines whether training batches shall be '
                             'shuffled, if not hard-coded else wise')
    parser.add_argument('--loss', type=str, default='MSE',
                        help='Training loss.',
                        choices=['MSE', 'MAE', 'Smooth'])
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='Optimizer choice.',
                        choices=['Adam', 'SGD', 'AdamW'])
    parser.add_argument('--scheduler_patience', type=int,
                        default=2,
                        help='Number of epochs without improvement.')
    parser.add_argument('--scheduler_factor', type=float,
                        default=0.1,
                        help='Reduction factor for learning rate.')
    parser.add_argument('--stop_patience', type=int, default=20,
                        help='Number of epochs wo improvement.')
    parser.add_argument('--stop_delta', type=float,
                        default=0.00001, help='Minimum change.')

    # ----------
    # Model settings
    parser.add_argument('--decomp_method', type=str,
                        default='moving_avg')
    parser.add_argument('--scales', type=int, nargs='+', default=[1],
                        help='list of kernel sizes for decomposition')
    parser.add_argument("--use_revin",
                        type=lambda x: x.lower() == 'true',
                        default=False,
                        help='Decides whether to use revin or not.')
    parser.add_argument('--kernel_size', type=int, default=2,
                        help='Kernel size for convolutional layers or '
                             'sliding mean.')
    parser.add_argument('--window_size', type=int, default=25,
                        help='Kernel size for moving average in '
                             'decomposition layer.')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Rate for dropout.')
    parser.add_argument('--hidden_size', type=int, default=100,
                        help='Size of hidden dimension.')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of stacked layers for networks where this'
                             ' is defined as tunable parameter.')
    parser.add_argument('--separate', action='store_true',
                        help='Defines whether each variate/variable shall be '
                             'processed separately.')
    # Parse arguments
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of subprocesses during data loading')
    parser.add_argument('--model_dim', type=int, default=512,
                        help='The dim of the model (input features).')
    # Transformer parameters
    parser.add_argument('--num_heads', type=int, default=None,
                        help='Num of separate heads in Transformer.')
    parser.add_argument('--factor', type=int, default=None,
                        help='Factor to choose top k periods, higher factor '
                             'results in a larger top_k (Autoformer).')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Top k factor.')
    parser.add_argument('--num_encoder_layers', type=int,
                        default=2,
                        help='Num of encoder layers (defaults to 2)')
    parser.add_argument('--num_decoder_layers', type=int,
                        default=1,
                        help='Num of decoder layers (defaults to 1)')
    parser.add_argument('--ff_dim', type=int, default=2048,
                        help='Num of feedforward dimension (defaults to 2048)')
    parser.add_argument('--conv_dim', type=int, default=4,
                        help='Width of convolution layers, Mamba (default 4)')
    parser.add_argument('--expand', type=int, default=2,
                        help='factor for inner dimension - Mamba (default 2)')
    parser.add_argument('--attention_dropout', type=float,
                        default=0.1,
                        help='Rate for dropout of attention layer.')
    parser.add_argument('--activation', type=str, default='gelu',
                        help='activation')
    parser.add_argument('--output_attention', type=bool,
                        default=False,
                        help='Whether to output attention in decoder')
    parser.add_argument('--use_norm', type=bool, default=True,
                        help='Whether to use normalization layer.')
    parser.add_argument('--num_filters', type=int, default=32,
                        help='Num of unique output features.')
    # dilated convolution
    parser.add_argument('--skip_channels', type=int, default=32,
                        help='Num of channels to be skipped in WaveNet model.')
    parser.add_argument('--num_dilated_layers', type=int, default=8,
                        help='Num of dilated layers in convolutional models.')

    parser.add_argument('--transform', type=str, default=None,
                        help='Flag for time series transformation.',
                        choices=['FFT', '2dFFT', 'DWT', 'None'])
    # TimeMixer
    parser.add_argument('--down_sampling_layers', type=int, default=0,
                        help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1,
                        help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default='avg',
                        help='down sampling method, only support avg, max, '
                             'conv')
    parser.add_argument('--use_future_temporal_feature', type=bool,
                        default=False,
                        help='whether to use future_temporal_feature.')
    parser.add_argument('--task_name', type=str,
                        default='long_term_forecast')
    # TiDE
    parser.add_argument('--c_tide_out', type=int, default=8,
                        help='Num of forecasting variables in TiDE.')
    # VCFormer
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding',
                        choices=['timeF', 'fixed', 'learned'])
    parser.add_argument('--snap_size', type=int, default=16,
                        help='snapshot size for Koopman Temporal Detector')
    parser.add_argument('--proj_dim', type=int, default=128,
                        help='projection dim of Koopman space')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='hidden dim of Koopman Enc/Dec')
    parser.add_argument('--hidden_layers', type=int, default=1,
                        help='hidden layers of Koopman Temporal Detector')
    # SparseTSF
    parser.add_argument('--period_len', type=int, default=24,
                        help='time period of pattern fpr SparseTSF')
    # CycleNet
    parser.add_argument('--model_type', type=str, default='linear',
                        choices=['linear', 'mlp'])
    parser.add_argument('--cycle', type=int, default=24,
                        help='Cycle length for CycleNet.')
    # WaveMask
    parser.add_argument('--aug_type', type=int, default=0,
                        choices=[0, 1, 2, 3, 4, 5],
                        help='Aug type for WaveMask: 0: None, 1: Freq-Mask, '
                             '2: Freq-Mix, 3: Wave-Mask, 4: Wave-Mix, '
                             '5: StAug')
    parser.add_argument('--aug_rate', type=float, default=0.5,
                        help='rate for FreqMask, FreqMix, and STAug')
    parser.add_argument('--sampling_rate', type=float, default=0.5,
                        help='sampling rate for WaveMask and WaveMix')
    parser.add_argument('--wavelet', type=str, default='db2',
                        help='wavelet form for DWT')
    parser.add_argument('--level', type=int, default=2,
                        help='level for DWT')
    parser.add_argument('--rates', type=str,
                        default="[0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]",
                        help='List of float rates as a string, e.g., '
                             '"[0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]"')
    parser.add_argument('--n_imf', type=int, default=500)

    # ----------
    return parser.parse_args()
