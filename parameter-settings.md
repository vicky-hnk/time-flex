# Model Parameter Settings for Datasets

## Table of Contents
1. [Model Parameters among all datasets](#model-parameters-among-all-datasets)
2. [GP-TimeSet Model Parameters](#gp-timeset-model-parameters)
3. [Traffic Dataset](#traffic-dataset)
4. [Etth1, Etth2 Datasets](#etth1-etth2-datasets)
5. [Ettm1 Dataset](#ettm1-dataset)
6. [Ettm2 Dataset](#ettm2-dataset)
7. [Weather Dataset](#weather-dataset)
8. [Electricity Dataset](#electricity-dataset)
9. [Exchange Rate Dataset](#exchange-rate-dataset)
10. [Illness Dataset](#illness-dataset)
11. [WaveMask Settings](#wavemask-settings)

## Model Parameters among all datasets

| Parameter              | Auto-former | Cross-former | Cycle Net | DLinear/ NLinear | Fed-former | iTrans/ PatchTST | PatchTST | TiDE | TimeFlex | TimeMixer | VC-former | 
|------------------------|-------------|--------------|-----------|------------------|------------|------------------|----------|------|----------|-----------|-----------|
| `window_size`          | 25          | 25           | -         | 25               | 25         | 25               | -        | -    | 25       | 25        | -         |   
| `kernel_size`          | -           | -            | -         | -                | -          | -                | -        | -    | 10       | -         | -         |          
| `num_dilated_layers`   | -           | -            | -         | -                | -          | -                | -        | -    | 6        | -         | -         |      
| `dropout`              | 0.1         | 0.1          | -         | -                | 0.1        | 0.1              | 0.2      | 0.3  | 0.1      | -         | 0.1       |  
| `attention_dropout`    | 0.1         | 0.1          | -         | -                | -          | 0.1              | -        | -    | -        | -         | -         | 
| `num_heads`            | 8           | 2            | -         | -                | 8          | 8                | 8        | -    | -        | -         | 8         | 
| `ff_dim`               | 2048        | 2048         | -         | -                | 2048       | 128              | -        | 256  | -        | -         | 1024      |
| `factor`               | 3           | 3            | -         | -                | 3          | 3                | -        | -    | -        | -         | -         |
| `num_encoder_layers`   | 2           | 2            | -         | -                | 2          | 2                | 2        | 2    | -        | -         | 3         | 
| `num_decoder_layers`   | 1           | 1            | -         | -                | 1          | 1                | 1        | 2    | -        | -         | 1         | 
| `seq_overlap`          | 48          | 48           | 48        | 0                | 48         | 48               | 48       | 48   | -        | -         | 48        |
| `down_sampling_layers` | -           | -            | -         | -                | -          | -                | -        | -    | -        | 3         | -         |
| `down_sampling_window` | -           | -            | -         | -                | -          | -                | -        | -    | -        | 2         | -         |
| `snap_size`            | -           | -            | -         | -                | -          | -                | -        | -    | -        | -         | 16        |
| `proj_dim`             | -           | -            | -         | -                | -          | -                | -        | -    | -        | -         | 128       |

## GP-TimeSet Model Parameters

| Parameter    | Auto-former | Cross-former | Cycle Net | DLinear/ NLinear/ TimeFlex/ TimeMixer/ WaveMask/ WaveNet | Fed-former | iTrans/ PatchTST | TimeFlex | TimeMixer | VCformer | 
|--------------|-------------|--------------|-----------|----------------------------------------------------------|------------|------------------|----------|-----------|----------|
| `batch_size` | 6           | 6            | 6         | 6                                                        | 6          | 6                | 6        | 6         | 6        | 
| `epochs`     | 25          | 25           | 25        | 25                                                       | 25         | 25               | 25       | 25        | 25       | 
| `model_dim`  | 512         | 512          | 256       | -                                                        | 512        | 256              | 256      | -         | 256      |    
| `cycle`      | -           | -            | 24        | -                                                        | -          | -                | -        | -         | -        |

## Traffic Dataset

| Parameter    | Auto-former | Cross-former | Cycle Net | DLinear/ NLinear/ TimeFlex/ TimeMixer/ WaveMask/ WaveNet | Fed-former | iTrans/ PatchTST/ TiDE | VCformer | 
|--------------|-------------|--------------|-----------|----------------------------------------------------------|------------|------------------------|----------|
| `batch_size` | 8           | 4            | 8         | 8                                                        | 8          | 8                      | 8        | 
| `epochs`     | 20          | 20           | 20        | 20                                                       | 20         | 20                     | 20       |
| `model_dim`  | 32          | 32           | 256       | -                                                        | 512        | 32                     | 256      |
| `cycle`      | -           | -            | 168       | -                                                        | -          | -                      | -        |

## Etth1, Etth2 Datasets

| Parameter    | Auto-former | Cross-former | Cycle Net | DLinear/ NLinear/ TimeFlex/ TimeMixer/ WaveMask/ WaveNet | Fed-former | iTrans/ PatchTST/ TiDE | VCformer |
|--------------|-------------|--------------|-----------|----------------------------------------------------------|------------|------------------------|----------|
| `batch_size` | 128         | 64           | 128       | 128                                                      | 128        | 128                    | 128      |
| `epochs`     | 10          | 10           | 10        | 10                                                       | 10         | 10                     | 10       |
| `model_dim`  | 16          | 16           | 256       | -                                                        | 512        | 16                     | 256      |
| `cycle`      | -           | -            | 24        | -                                                        | -          | -                      | -        |

## Ettm1 Dataset

| Parameter    | Auto-former | Cross-former | Cycle Net | DLinear/ NLinear/ TimeFlex/ TimeMixer/ WaveMask/ WaveNet | Fed-former | iTrans/ PatchTST/ TiDE | VCformer |
|--------------|-------------|--------------|-----------|----------------------------------------------------------|------------|------------------------|----------|
| `batch_size` | 128         | 64           | 128       | 128                                                      | 128        | 128                    | 128      |
| `epochs`     | 10          | 10           | 10        | 10                                                       | 10         | 10                     | 10       |
| `model_dim`  | 16          | 16           | 256       | -                                                        | 512        | 16                     | 256      |
| `cycle`      | -           | -            | 96        | -                                                        | -          | -                      | -        |

## Ettm2 Dataset

| Parameter    | Auto-former | Cross-former | Cycle Net | DLinear/ NLinear/ TimeFlex/ TimeMixer/ WaveMask/ WaveNet | Fed-former | iTrans/ PatchTST/ TiDE | VCformer |
|--------------|-------------|--------------|-----------|----------------------------------------------------------|------------|------------------------|----------|
| `batch_size` | 128         | 128          | 128       | 128                                                      | 128        | 128                    | 128      |
| `epochs`     | 10          | 10           | 10        | 10                                                       | 10         | 10                     | 10       |
| `model_dim`  | 32          | 32           | 256       | -                                                        | 512        | 16                     | 256      |
| `cycle`      | -           | -            | 96        | -                                                        | -          | -                      | -        |

## Weather Dataset

| Parameter    | Auto-former | Cross-former | Cycle Net | DLinear/ NLinear/ TimeFlex/ TimeMixer/ WaveMask/ WaveNet | Fed-former | iTrans/ PatchTST/ TiDE | VCformer |
|--------------|-------------|--------------|-----------|----------------------------------------------------------|------------|------------------------|----------|
| `batch_size` | 16          | 16           | 16        | 16                                                       | 16         | 16                     | 16       |
| `epochs`     | 20          | 20           | 20        | 20                                                       | 20         | 20                     | 20       |
| `model_dim`  | 16          | 16           | 256       | -                                                        | 512        | 16                     | 256      |
| `cycle`      | -           | -            | 144       | -                                                        | -          | -                      | -        |

## Electricity Dataset

| Parameter    | Auto-former | Cross-former | Cycle Net | DLinear/ NLinear/ TimeFlex/ TimeMixer/ WaveMask/ WaveNet | Fed-former | iTrans/ PatchTST/ TiDE | VCformer | 
|--------------|-------------|--------------|-----------|----------------------------------------------------------|------------|------------------------|----------|
| `batch_size` | 32          | 16           | 32        | 32                                                       | 32         | 32                     | 32       |
| `epochs`     | 20          | 20           | 20        | 20                                                       | 20         | 20                     | 20       |
| `model_dim`  | 16          | 16           | 256       | -                                                        | 512        | 16                     | 256      |
| `cycle`      | -           | -            | 168       | -                                                        | -          | -                      | -        |

## Exchange Rate Dataset

| Parameter    | Auto-former | Cross-former | Cycle Net | DLinear/ NLinear/ TimeFlex/ TimeMixer/ WaveMask/ WaveNet | Fed-former | iTrans/ PatchTST/ TiDE | VCformer |
|--------------|-------------|--------------|-----------|----------------------------------------------------------|------------|------------------------|----------|
| `batch_size` | 32          | 32           | 32        | 32                                                       | 32         | 32                     | 32       |
| `epochs`     | 20          | 20           | 20        | 20                                                       | 20         | 20                     | 20       |
| `model_dim`  | 16          | 16           | 256       | -                                                        | 512        | 16                     | 256      |
| `cycle`      | -           | -            | 7         | -                                                        | -          | -                      | -        |

## Illness Dataset

| Parameter    | Auto-former | Cross-former | Cycle Net | DLinear/ NLinear/ TimeFlex/ TimeMixer/ WaveMask/ WaveNet | Fed-former | iTrans/ PatchTST/ TiDE | VCformer | 
|--------------|-------------|--------------|-----------|----------------------------------------------------------|------------|------------------------|----------|
| `batch_size` | 32          | 32           | 32        | 32                                                       | 32         | 32                     | 32       |
| `epochs`     | 20          | 20           | 20        | 20                                                       | 20         | 20                     | 20       |   
| `model_dim`  | 16          | 16           | 256       | -                                                        | 512        | 16                     | 256      |   
| `cycle`      | -           | -            | 52        | -                                                        | -          | -                      | -        | 


## WaveMask Settings
| Parameter       | Value                               |
|-----------------|-------------------------------------|
| `aug_types`     | 1 2 3 4 5                           |
| `aug_rate`      | 0.5                                 |
| `n_imf`         | 500                                 |
| `rates`         | [0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1] |
| `wavelet`       | db2                                 |
| `sampling_rate` | 0.5                                 |
| `level`         | 2                                   |
