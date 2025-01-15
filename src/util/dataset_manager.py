"""
This file contains the parameters specific to each dataset. The dataset name
can be defined as argparse parameter within the bash file.
"""
import sys

from src.data_loader.timeseries_data_loader import DatasetTimeseries
from src.data_loader.timeseries_data_loader_cycle import DatasetTimeseriesCycle
from src.data_loader.timeseries_data_loader_wavemask import (
    DatasetTimeseriesWave)


considered_datasets = {'Beutenberg21': DatasetTimeseries,
                       'Beutenberg22': DatasetTimeseries,
                       'Beutenberg21_22': DatasetTimeseries,
                       'Etth_test': DatasetTimeseries,
                       'Etth1': DatasetTimeseries, 'Etth2': DatasetTimeseries,
                       'Ettm1': DatasetTimeseries, 'Ettm2': DatasetTimeseries,
                       'traffic': DatasetTimeseries,
                       'illness': DatasetTimeseries,
                       'exchange_rate': DatasetTimeseries,
                       'electricity': DatasetTimeseries,
                       'weather': DatasetTimeseries,
                       'se_kernel': DatasetTimeseries,
                       'periodic_kernel': DatasetTimeseries,
                       'locally_periodic_kernel': DatasetTimeseries,
                       'rational_quadratic_kernel': DatasetTimeseries,
                       'linear_periodic_kernel': DatasetTimeseries,
                       'combined_kernel': DatasetTimeseries,
                       'season_kernel': DatasetTimeseries,
                       'trend_kernel': DatasetTimeseries}

cycle_datasets = {'Beutenberg21': DatasetTimeseriesCycle,
                  'Beutenberg22': DatasetTimeseriesCycle,
                  'Beutenberg21_22': DatasetTimeseriesCycle,
                  'Etth_test': DatasetTimeseriesCycle,
                  'Etth1': DatasetTimeseriesCycle,
                  'Etth2': DatasetTimeseriesCycle,
                  'Ettm1': DatasetTimeseriesCycle,
                  'Ettm2': DatasetTimeseriesCycle,
                  'traffic': DatasetTimeseriesCycle,
                  'illness': DatasetTimeseriesCycle,
                  'exchange_rate': DatasetTimeseriesCycle,
                  'electricity': DatasetTimeseriesCycle,
                  'weather': DatasetTimeseriesCycle,
                  'se_kernel': DatasetTimeseriesCycle,
                  'periodic_kernel': DatasetTimeseriesCycle,
                  'locally_periodic_kernel': DatasetTimeseriesCycle,
                  'rational_quadratic_kernel': DatasetTimeseriesCycle,
                  'linear_periodic_kernel': DatasetTimeseriesCycle,
                  'combined_kernel': DatasetTimeseriesCycle,
                  'season_kernel': DatasetTimeseriesCycle,
                  'trend_kernel': DatasetTimeseriesCycle}

mask_datasets = {'Beutenberg21': DatasetTimeseriesWave,
                 'Beutenberg22': DatasetTimeseriesWave,
                 'Beutenberg21_22': DatasetTimeseriesWave,
                 'Etth_test': DatasetTimeseriesWave,
                 'Etth1': DatasetTimeseriesWave,
                 'Etth2': DatasetTimeseriesWave,
                 'Ettm1': DatasetTimeseriesWave,
                 'Ettm2': DatasetTimeseriesWave,
                 'traffic': DatasetTimeseriesWave,
                 'illness': DatasetTimeseriesWave,
                 'exchange_rate': DatasetTimeseriesWave,
                 'electricity': DatasetTimeseriesWave,
                 'weather': DatasetTimeseriesWave,
                 'se_kernel': DatasetTimeseriesWave,
                 'periodic_kernel': DatasetTimeseriesWave,
                 'locally_periodic_kernel': DatasetTimeseriesWave,
                 'rational_quadratic_kernel': DatasetTimeseriesWave,
                 'linear_periodic_kernel': DatasetTimeseriesWave,
                 'combined_kernel': DatasetTimeseriesWave,
                 'season_kernel': DatasetTimeseriesWave,
                 'trend_kernel': DatasetTimeseriesWave}

dataset_params = {
    # open source data sets
    "traffic": {"freq": "h", "data_root_path": "data",
                "data_path": "traffic/traffic.csv",
                "target": "OT", "num_variates": 862, "c_out": 862,
                "enc_in": 862, "dec_in": 862, "output_feature_dim": 862},
    "Etth1": {"freq": "h", "data_root_path": "data", "c_out": 7,
              "data_path": "ETT/ETTh1.csv", "target": "OT", "num_variates": 7,
              "enc_in": 7, "dec_in": 7, "output_feature_dim": 7},
    "Etth2": {"freq": "h", "data_root_path": "data", "c_out": 7,
              "data_path": "ETT/ETTh2.csv", "target": "OT", "num_variates": 7,
              "enc_in": 7, "dec_in": 7, "output_feature_dim": 7},
    "Ettm1": {"freq": "min", "data_root_path": "data", "c_out": 7,
              "data_path": "ETT/ETTm1.csv", "target": "OT", "num_variates": 7,
              "enc_in": 7, "dec_in": 7, "output_feature_dim": 7},
    "Ettm2": {"freq": "min", "data_root_path": "data", "c_out": 7,
              "data_path": "ETT/ETTm2.csv", "target": "OT", "num_variates": 7,
              "enc_in": 7, "dec_in": 7, "output_feature_dim": 7},
    "Beutenberg21_22": {"freq": "h", "data_root_path": "data",
                        "data_path": "weather/mpi_roof_2021_22.csv",
                        "target": "T (degC)", "num_variates": 21, "c_out": 21,
                        "enc_in": 21, "dec_in": 21, "output_feature_dim": 21},
    "Beutenberg21": {"freq": "h", "data_root_path": "data",
                     "data_path": "weather/mpi_roof_2021.csv",
                     "target": "T (degC)", "num_variates": 21, "c_out": 21,
                     "enc_in": 21, "dec_in": 21, "output_feature_dim": 21},
    "Beutenberg22": {"freq": "h", "data_root_path": "data",
                     "data_path": "weather/mpi_roof_2022.csv",
                     "target": "T (degC)", "num_variates": 21, "c_out": 21,
                     "enc_in": 21, "dec_in": 21, "output_feature_dim": 21},
    "Beutenberg23": {"freq": "h", "data_root_path": "data",
                     "data_path": "weather/mpi_roof_2023.csv",
                     "target": "T (degC)", "num_variates": 21, "c_out": 21,
                     "enc_in": 21, "dec_in": 21, "output_feature_dim": 21},
    "weather": {"freq": "h", "data_root_path": "data",
                "data_path": "weather/weather.csv",
                "target": "T (degC)", "num_variates": 21, "c_out": 21,
                "enc_in": 21, "dec_in": 21, "output_feature_dim": 21},
    "electricity": {"freq": "h", "data_root_path": "data",
                    "data_path": "electricity/electricity.csv",
                    "target": "OT", "num_variates": 321, "c_out": 321,
                    "enc_in": 321, "dec_in": 321, "output_feature_dim": 321},
    "exchange_rate": {"freq": "D", "data_root_path": "data",
                      "data_path": "exchange_rate/exchange_rate.csv",
                      "target": "OT", "num_variates": 8, "c_out": 8,
                      "enc_in": 8, "dec_in": 8, "output_feature_dim": 8},
    "illness": {"freq": "W", "data_root_path": "data",
                "data_path": "illness/national_illness.csv",
                "target": "OT", "num_variates": 7, "c_out": 7,
                "enc_in": 7, "dec_in": 7, "output_feature_dim": 7},
    # sampled datasets
    "se_kernel": {"freq": "h", "data_root_path": "data",
                  "data_path": "sampled_data/se_kernel.csv",
                  "target": "value4", "num_variates": 4, "c_out": 4,
                  "enc_in": 4, "dec_in": 4, "output_feature_dim": 4},
    "periodic_kernel": {"freq": "h", "data_root_path": "data",
                        "data_path": "sampled_data/periodic_kernel.csv",
                        "target": "value4", "num_variates": 4, "c_out": 4,
                        "enc_in": 4, "dec_in": 4, "output_feature_dim": 4},
    "locally_periodic_kernel": {
        "freq": "h", "data_root_path": "data",
        "data_path": "sampled_data/locally_periodic_kernel.csv",
        "target": "value4", "num_variates": 4, "c_out": 4, "enc_in": 4,
        "dec_in": 4, "output_feature_dim": 4},
    "rational_quadratic_kernel": {
        "freq": "h", "data_root_path": "data",
        "data_path": "sampled_data/rational_quadratic_kernel.csv",
        "target": "value4", "num_variates": 4, "c_out": 4,
        "enc_in": 4, "dec_in": 4, "output_feature_dim": 4},
    "linear_periodic_kernel": {
        "freq": "h", "data_root_path": "data",
        "data_path": "sampled_data/linear_periodic_kernel.csv",
        "target": "value4", "num_variates": 4, "c_out": 4,
        "enc_in": 4, "dec_in": 4, "output_feature_dim": 4},
    "combined_kernel": {"freq": "h", "data_root_path": "data",
                        "data_path": "sampled_data/combined_kernel.csv",
                        "target": "value4", "c_out": 4,
                        "num_variates": 4, "enc_in": 4, "dec_in": 4,
                        "output_feature_dim": 4},
    "season_kernel": {"freq": "h", "data_root_path": "data",
                      "data_path": "sampled_data/season_kernel.csv",
                      "target": "value4", "c_out": 4,
                      "num_variates": 4, "enc_in": 4, "dec_in": 4,
                      "output_feature_dim": 4},
    "trend_kernel": {"freq": "h", "data_root_path": "data",
                     "data_path": "sampled_data/trend_kernel.csv",
                     "target": "value4", "c_out": 4,
                     "num_variates": 4, "enc_in": 4, "dec_in": 4,
                     "output_feature_dim": 4},
    # Add parameters for new datasets here.
}


def merge_params(cmd_args):
    """
    Merges params from command line with the one set in the dictionary above.
    :param cmd_args: command line arguments, should be vars not
    argparse Namespace!
    """
    dataset_name = cmd_args['data']
    if dataset_name not in dataset_params:
        print(f"Error: Dataset '{dataset_name}' not recognized.")
        sys.exit(1)
    dataset_specific_params = dataset_params[dataset_name]
    merged_params = {**dataset_specific_params, **cmd_args}
    return merged_params
