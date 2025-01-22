#!/bin/bash

# set a seed
seed=3

# fixed values
seq_len=96
seq_overlap=48 # label_len in original implementation
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
epochs=25

#settings for data preparation
data_source="local_csv"
features="M"
time_enc=0
scale="Standard"
batch_size=6

BASE_DIR=$(dirname "$(realpath "$0")")/../..

# List of datasets and prediction lengths
datasets=("periodic_kernel" "locally_periodic_kernel" "se_kernel" "combined_kernel" "rational_quadratic_kernel")
pred_lengths=(96 192 336 720)

for dataset in "${datasets[@]}"; do
      for pred_len in "${pred_lengths[@]}"; do
      python3 "$BASE_DIR/src/experiments/exec_entry.py" "fedformer" \
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
          --num_heads 8 \
          --model_dim 512 \
          --ff_dim 2048 \
          --factor 3 \
          --embed 'fixed' \
          --num_encoder_layers 2 \
          --num_decoder_layers 1 \
          --random_seed "$seed"
    done
done
