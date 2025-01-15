"""Module for evaluation of performance."""
import os
import torch
import json
import joblib
from datetime import datetime

import numpy as np
import pandas as pd

from src.util.train_utils import Metrics, set_seeds
from src.util import mlflow_utils

set_seeds(0)


class ModelEvaluator:
    """Class contains (static) methods to handle the model's evaluation"""

    @staticmethod
    def unscale(dataframe: pd.DataFrame, scaler_name: str) -> pd.DataFrame:
        """Loads a scaler file and returns an unscaled pandas Dataframe."""

        def calc_unscaled(data, mean, scale):
            return (data.values * scale + mean).flatten()

        current_dir = os.path.dirname(os.path.abspath(__file__))
        scaler = joblib.load(
            os.path.join(
                current_dir, '..', '..', 'key_files', 'scaler',
                f'{scaler_name}.joblib'))

        columns_to_unscale = [col for col in dataframe.columns if
                              col.startswith('Pred_') or col.startswith(
                                  'Real_')]
        for col in columns_to_unscale:
            scaler_mean, scaler_scale = scaler.mean_[-1], scaler.scale_[-1]
            dataframe[col] = calc_unscaled(dataframe[col], scaler_mean,
                                           scaler_scale)

        return dataframe

    @staticmethod
    def create_test_df(prediction: np.ndarray, ground_truth: np.ndarray,
                       marks: list, variates: int) -> pd.DataFrame:
        """
        Recreates datetime timestamps from encoded date values and returns a
        Dataframe with predicted and real values.
        """
        if prediction.shape != ground_truth.shape:
            raise ValueError(
                "Prediction and ground_truth arrays must have the "
                "same dimensions.")

        # create a combined dataframe
        column_names = [f'{prefix}_Val{i + 1}' for prefix in ('Pred', 'Real')
                        for i in range(variates)]
        combined_data = np.hstack((prediction, ground_truth))
        test_df = pd.DataFrame(combined_data, columns=column_names)

        # reverse the date encodings
        date_marks = pd.to_datetime([datetime(year=row[0], month=row[1],
                                              day=row[2], hour=row[4],
                                              minute=row[5])
                                     for array in marks[0] for row in array])
        test_df.insert(0, 'date', date_marks[:len(test_df)])
        return test_df

    @staticmethod
    def save_evaluation(predictions: list, truth: list):
        """
        Calculates and stores important metrics in json file. If necessary
        converts Pytorch Tensors to lists.
        Logged with MLFlow.
        """
        metrics_values = Metrics.calculate_metrics(predictions[0], truth[0])
        print(f"\n{'=' * 46}\nTest metrics {metrics_values} \n{'=' * 46}\n")
        results = {metric: value.item() for metric, value in
                   zip(['mae', 'mse', 'rmse'], metrics_values)}
        mlflow_utils.log_metrics(results)
