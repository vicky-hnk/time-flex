"""Utility functions for training and evaluating models."""
import torch
import random
import numpy as np


def set_seeds(seed=0):
    torch.manual_seed(seed)
    # torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True


class EarlyStopping:
    def __init__(self, patience=10, delta=0.00001, mode='min'):
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.mode == 'min':
            score = -val_loss
        else:
            score = val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


class Metrics:
    def __init__(self):
        pass

    @staticmethod
    def mse(val_pred, val_true):
        return ((val_pred - val_true) ** 2).mean()

    @staticmethod
    def mae(val_pred, val_true):
        return abs(val_pred - val_true).mean()

    @staticmethod
    def rmse(val_pred, val_true):
        return (((val_pred - val_true) ** 2) ** .5).mean()

    @staticmethod
    def calculate_metrics(values_pred, values_true,
                          metric_flags: list = ['mae', 'mse', 'rmse']):
        metrics_values = []
        for func_flag in metric_flags:
            func = getattr(Metrics, func_flag, None)
            if func and callable(func):
                result = func(values_pred, values_true)
                metrics_values.append(result)
        return metrics_values
