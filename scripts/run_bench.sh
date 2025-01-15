#!/bin/bash

BASE_DIR=$(dirname "$(realpath "$0")")

# Autoformer
bash "$BASE_DIR/benchmarking_datasets/autoformer_bench.sh"

# Crossformer
bash "$BASE_DIR/benchmarking_datasets/crossformer_bench.sh"

# CycleNet
bash "$BASE_DIR/benchmarking_datasets/cycle_bench.sh"

# DLinear
bash "$BASE_DIR/benchmarking_datasets/d_linear_bench.sh"

# FEDformer
bash "$BASE_DIR/benchmarking_datasets/fedformer_bench.sh"

# I-Transformer
bash "$BASE_DIR/benchmarking_datasets/i_transformer_bench.sh"

# NLinear
bash "$BASE_DIR/benchmarking_datasets/n_linear_bench.sh"

# PatchTST
bash "$BASE_DIR/benchmarking_datasets/patchtst_bench.sh"

# RLinear
bash "$BASE_DIR/benchmarking_datasets/r_linear_bench.sh"

# SparseTSF
bash "$BASE_DIR/benchmarking_datasets/sparse_tsf_bench.sh"

# TiDE
bash "$BASE_DIR/benchmarking_datasets/tide_bench.sh"

# TimeFlex
bash "$BASE_DIR/benchmarking_datasets/time_flex_bench.sh"

# TimeFlexJPP
bash "$BASE_DIR/benchmarking_datasets/time_flex_jpp_bench.sh"

# TimeMixer
bash "$BASE_DIR/benchmarking_datasets/timemixer_bench.sh"

# VCFormer
bash "$BASE_DIR/benchmarking_datasets/vcformer_bench.sh"

# WaveMask
bash "$BASE_DIR/benchmarking_datasets/wavemask_bench.sh"

# WaveNet
bash "$BASE_DIR/benchmarking_datasets/wavenet_bench.sh"

