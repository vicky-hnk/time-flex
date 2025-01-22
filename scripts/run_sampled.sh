#!/bin/bash

export PYTHONPATH="$HOME/repos/time-flex":$PYTHONPATH
BASE_DIR=$(dirname "$(realpath "$0")")

# Autoformer
bash "$BASE_DIR/sampled_datasets/autoformer_sampled.sh"

# Crossformer
bash "$BASE_DIR/sampled_datasets/crossformer_sampled.sh"

# CycleNet
bash "$BASE_DIR/sampled_datasets/cycle_sampled.sh"

# DLinear
bash "$BASE_DIR/sampled_datasets/d_linear_sampled.sh"

# FEDformer
bash "$BASE_DIR/sampled_datasets/fedformer_sampled.sh"

# I-Transformer
bash "$BASE_DIR/sampled_datasets/i_transformer_sampled.sh"

# NLinear
bash "$BASE_DIR/sampled_datasets/n_linear_sampled.sh"

# PatchTST
bash "$BASE_DIR/sampled_datasets/patchtst_sampled.sh"

# RLinear
bash "$BASE_DIR/sampled_datasets/r_linear_sampled.sh"

# SparseTSF
bash "$BASE_DIR/sampled_datasets/sparse_tsf_sampled.sh"

# TiDE
bash "$BASE_DIR/sampled_datasets/tide_sampled.sh"

# TimeFlex
bash "$BASE_DIR/sampled_datasets/time_flex_sampled.sh"

# TimeFlexJPP
bash "$BASE_DIR/sampled_datasets/time_flex_jpp_sampled.sh"

# TimeMixer
bash "$BASE_DIR/sampled_datasets/timemixer_sampled.sh"

# VCFormer
bash "$BASE_DIR/sampled_datasets/vcformer_sampled.sh"

# WaveMask
bash "$BASE_DIR/sampled_datasets/wavemask_sampled.sh"

# WaveNet
bash "$BASE_DIR/sampled_datasets/wavenet_sampled.sh"

