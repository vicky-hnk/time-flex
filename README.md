# TimeFlex

This repository contains code and resources related to the scientific paper titled 
"Tailored Architectures for Time Series Forecasting:
Evaluating Deep Learning Models on Gaussian
Process-Generated Data". It includes the scripts, datasets, and environment setup necessary to 
reproduce the experiments and results presented in the paper.

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Datasets and Data Manager](#datasets-and-data-manager)
3. [Environment Setup](#environment-setup)
4. [Running the Scripts](#running-the-scripts)
   - [run_all.sh](#run_allsh)
   - [run_bench.sh](#run_benchsh)
   - [run_sampled.sh](#run_sampledsh)
5. [Model Overview](#model-overview)
6. [Results](#results)

## Repository Structure

The repository is organized as follows:
* **key_files** is a file storage like the applied scaler
* **scripts** contains the shell scripts to run all experiments, i.e. execute the training
* **src** contains the main code base:
  * **data**: all datasets
  * **data_loader**: general time series dataloaders, and adapted ones for _CycleNet_ and _WaveMask_
  * **evaluation**: class for experiment evaluation
  * **experiments**: exec_entry.py is the entrypoint to execute training, executor.py contains the training class, main.py contains the main run function 

## Datasets and Data Manager

This project uses the following datasets (which can be found in src/data):

- **Benchmarking**: ETT, Traffic, Weather, Solar, Electricity, Exchange Rate.
- **GP-Timeset**: SE, Combined, Periodic, Locally Periodic, Rational Quadratic.

You can place your datasets in the `src/data/` folder and put the data parameters in util/dataset_manager.py. 
The `src/dataloader/timeseries_data_loader.py` script is responsible for loading and preprocessing these datasets. 
You can modify it according to your needs.

## Environment Setup

To run the experiments, you will need to set up a Python environment with the required dependencies. Follow these steps to set up your environment:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/time-flex.git
    cd time-flex
    ```

2. Create a conda environment:
    ```bash
    conda env create -f environment.yml
    conda activate torch
    ```
   
3. Make sure you have all necessary system dependencies (e.g., CUDA for GPU support) if applicable. To check CUDA availability run:
    ```
   python3 src/util/check_cuda.py
    ```

## Running the Scripts

### `run_all.sh`

This script runs all the experiments in sequence. It is designed to automate the entire process from data preprocessing to model training and evaluation. To run it, simply execute:

```bash
bash scripts/run_all.sh