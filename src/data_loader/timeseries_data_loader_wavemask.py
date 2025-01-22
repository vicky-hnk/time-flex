"""
This module contains classes to provide Datasets for Wave-Augs/Mask models for
time series forecasting using emd augmentation.
"""

import os

import torch
import joblib
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from src.util.train_utils import set_seeds
from src.util.general_utils import Attribs
from src.data_loader.wavemask.emd_augmentation import emd_augment

set_seeds(0)


class DatasetTimeseriesWave(Dataset):
    """
    WaveMask data loader taken from:
    https://github.com/jafarbakhshaliyev/Wave-Augs/blob/main/dataset_loader/datasetloader.py
    """

    def __init__(self, df_raw: pd.DataFrame, target: str,
                 flag: str = 'train', size=None, data=str,
                 features: str = 'MS', scale: str = "False", time_enc: int = 0,
                 freq: str = 'h', log_encoding: bool = False, data_split=None,
                 n_imf: int = 500, percentage: int = 100, aug_type: int = 5):
        """
        Parameters:
            :param scale: uses scikit StandardScaler/MaxMinScaler
            :param time_enc: 0 (month, day, weekday, hour, minute,
            (second) seperated)
            :param features: S for uni-var.->uni-var.,
            M for multi-var.->multi-var., MS for multi-var.->uni-var.
        Remarks:
            y is dependent variable, x the independent (y=f(x))
        """

        if data_split is None:
            data_split = [0.7, 0.2, 0.1]
        self.df_raw = df_raw
        self.seq_len = size[0]
        self.seq_overlap = size[1]
        self.pred_len = size[2]
        self.data_overlap = size[3]

        self.train_portion = data_split[0]
        self.val_portion = data_split[1]
        self.test_portion = data_split[2]

        if flag not in ['train', 'val', 'test']:
            raise ValueError("Flag should be 'train', 'val', or 'test'")
        self.set_type = {'train': 0, 'val': 1, 'test': 2}[flag]
        if features not in ['S', 'M', 'MS']:
            raise ValueError("Features should be 'S', 'M', or 'MS'")
        self.features = features
        self.target = target
        self.time_enc = time_enc
        valid_freqs = ['h', 'min', 'sec', 'D', 'W']
        if freq not in valid_freqs:
            raise ValueError(
                f"Invalid frequency: {freq}. Frequency should be one of "
                f"{valid_freqs}")
        self.freq = freq
        self.n_imf = n_imf
        self.aug_type = aug_type
        self.percentage = percentage
        self.scale = scale
        self.data = data
        if self.scale == "Standard":
            self.scaler = StandardScaler()
        elif self.scale == "MinMax":
            self.scaler = MinMaxScaler()
        elif self.scale == "False":
            self.scaler = None
        else:
            self.scaler = None
        # use logarithmic encoding to smooth data
        self.log = log_encoding

        self.__read_data__()

    def __read_data__(self) -> None:
        total_length = len(self.df_raw)
        # define boundaries to split data to train, validation and test set
        train_end = int(self.train_portion * total_length)
        # creating an overlap of training and validation data of
        # length seq_len+data_overlap
        val_start = train_end - (self.seq_len + self.data_overlap)
        val_end = (val_start + int(self.val_portion * total_length) +
                   (self.seq_len + self.data_overlap))
        test_start = val_end
        test_end = test_start + int(self.test_portion * total_length)
        # set up sequence boundaries
        split_start = [0, val_start, test_start]
        split_end = [train_end, val_end, test_end]
        # define sequence borders according to flag
        border1, border2 = split_start[self.set_type], split_end[self.set_type]
        # select feature columns
        if self.features == 'M' or self.features == 'MS':
            cols_data = self.df_raw.columns[1:]
            df_data = self.df_raw[cols_data]
        elif self.features == 'S':
            df_data = self.df_raw[[self.target]]
        else:
            print("No valid feature provided - feature is set to S.")
            self.features = 'S'
            df_data = self.df_raw[[self.target]]
        # apply scaling if desired
        if self.scaler:
            # fit scaler on training data for consistency also among
            # test or validation set
            train_data = df_data[split_start[0]:split_end[0]]
            self.scaler.fit(train_data.values)
            # dump scaler for descaling afterwards
            scaler_path = os.path.join('..', '..', "key_files", "scaler",
                                       f"{self.scale}_{self.data}.joblib")
            os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
            joblib.dump(self.scaler, scaler_path)
            data = self.scaler.transform(df_data.values)
            print('Scaler successfully saved to {}'.format(scaler_path))
        else:
            data = df_data.values
        if self.log:
            data = np.log1p(data.values)

        if self.set_type == 0 and self.aug_type == 5:
            self.aug_data = emd_augment(
                data[border1:border2][-len(train_data):],
                self.seq_len + self.pred_len, n_IMF=self.n_imf)
        else:
            self.aug_data = np.zeros_like(data[border1:border2])

        if self.set_type == 0:
            self.data_x = data[border1:border2][-len(train_data):]
            self.data_y = data[border1:border2][-len(train_data):]
        else:
            self.data_x = data[border1:border2]
            self.data_y = data[border1:border2]

        if len(self.data_x) - self.seq_len - self.pred_len <= 1:
            raise ValueError(
                "Not enough data for sequence and prediction length. "
                "The length of input should be at least: ",
                self.seq_len + self.pred_len + 1,
                "The length is: ", len(self.data_x))

    def __getitem__(self, index):
        inp_begin = index
        inp_end = inp_begin + self.seq_len
        pred_begin = inp_end - self.seq_overlap
        pred_end = pred_begin + self.seq_overlap + self.pred_len
        # get input sequence and its marks
        seq_x = (self.data_x[inp_begin:inp_end])
        # get prediction sequence and its marks
        seq_y = (self.data_y[pred_begin:pred_end])
        if self.aug_type == 5:
            aug_data = self.aug_data[inp_begin]
        else:
            aug_data = np.array([])

        return seq_x, seq_y, aug_data, torch.tensor([])


    def __len__(self) -> int:
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


def provide_series_mask(params: Attribs, flag: str, data_dict: dict) \
        -> tuple[torch.utils.data.Dataset, torch.utils.data.DataLoader]:
    """
    Provides the time series data in the Pytorch-specific split to
    Dataset and Dataloader.
    """
    df_raw = pd.DataFrame()
    if params.data_source == 'local_csv':
        try:
            current_script_dir = os.path.dirname(os.path.abspath(__file__))
            csv_file = os.path.join(current_script_dir, "..",
                                    params.data_root_path, params.data_path)
            print(csv_file)
            df_raw = pd.read_csv(csv_file)
        except FileNotFoundError:
            print("CSV file was not found. Check file path and name.")
        except pd.errors.EmptyDataError:
            print("CSV file is empty. Provide a CSV file with data.")
        except pd.errors.ParserError:
            print("Error parsing CSV file. Check the file's structure.")
        except Exception as e:
            print(f"An unknown error {e} occurred.")
    data_class = data_dict[params.data]
    # Shuffle settings (no shuffling for seq2seq)
    set_shuffle = False if (params.train_procedure == 'seq2seq'
                            or flag != 'train') else True
    # no batch division of test data
    batch_size = 1 if flag == 'test' else params.batch_size
    # incomplete batches are only dropped in training and test phases
    drop_last = flag in {'train', 'test'}
    # create dataloader and dataset
    data_set = data_class(
        df_raw, flag=flag, size=[params.seq_len, params.seq_overlap,
                                 params.pred_len, params.data_overlap],
        data=params.data, features=params.features, target=params.target,
        time_enc=params.time_enc, freq=params.freq, scale=params.scale,
        data_split=params.data_split, n_imf=params.n_imf,
        percentage=params.percentage, aug_type=params.aug_type)
    print("Data loaded for: ", flag, "-task", "\n",
          "The length of the dataset is: ", len(data_set))
    data_loader = DataLoader(data_set, batch_size=batch_size,
                             shuffle=set_shuffle,
                             num_workers=params.num_workers,
                             drop_last=drop_last)
    return data_set, data_loader
