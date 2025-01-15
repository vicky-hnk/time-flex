#!/bin/bash
# adjust with your path
export PYTHONPATH="$HOME/repos/forecastexperiments":$PYTHONPATH

# fixed values
seq_len=96
seq_overlap=0
data_overlap=0

# settings for training
shuffle=True
num_workers=0
init_learning_rate=0.001
scheduler_factor=0.1
scheduler_patience=2
stop_patience=10
stop_delta=0.00001
loss="MSE"
optimizer="Adam"

#settings for data preparation
data_source="local_csv"
features="M"
time_enc=0
scale="Standard"

# Determine the base directory of the current repository and target directory of the trained model
BASE_DIR=$(dirname "$(realpath "$0")")/../..

# List of datasets and prediction lengths
datasets=("traffic" "Etth1" "Etth2" "Ettm1" "Ettm2" "weather" "electricity" "exchange_rate")
pred_lengths=(96 192 336 720)

for dataset in "${datasets[@]}"; do
  case $dataset in
        "traffic")
            batch_size=8
            epochs=20
            ;;
        "Etth1" | "Etth2" | "Ettm1" | "Ettm2")
            batch_size=128
            epochs=10
            ;;
        "weather")
            batch_size=16
            epochs=20
            ;;
        "electricity")
            batch_size=32
            epochs=20
            ;;
        "exchange_rate")
            batch_size=32
            epochs=20
            ;;
        "illness")
            batch_size=32
            epochs=20
            ;;
        *)
            echo "Unknown dataset: $dataset"
            exit 1
            ;;
    esac
    for pred_len in "${pred_lengths[@]}"; do
      python3 "$BASE_DIR/src/experiments/exec_entry.py" "timemixer" \
      --data "$dataset" \
      --pred_len "$pred_len" \
      --seq_len "$seq_len" \
      --seq_overlap "$seq_overlap" \
      --data_overlap "$data_overlap" \
      --root_path "$BASE_DIR" \
      --shuffle "$shuffle" \
      --num_workers "$num_workers" \
      --learning_rate "$init_learning_rate" \
      --scheduler_patience "$scheduler_patience" \
      --scheduler_factor "$scheduler_factor" \
      --stop_patience "$stop_patience" \
      --stop_delta "$stop_delta" \
      --loss "$loss" \
      --optimizer "$optimizer" \
      --epochs "$epochs" \
      --data_source "$data_source" \
      --batch_size "$batch_size" \
      --features "$features" \
      --time_enc "$time_enc" \
      --scale "$scale" \
      --window_size 25 \
      --dropout 0.1 \
      --down_sampling_layers 3 \
      --down_sampling_window 2 \
      --down_sampling_method "avg" \
      --use_future_temporal_feature true \
      --task_name "long_term_forecast" \
      --separate \
      --decomp_method "moving_avg"
  done
done
