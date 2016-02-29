#!/bin/bash

model_dir=models/multiclass/
data_dir=data/

n_embed_dims=10
n_filters=3000
filter_width=6
n_fully_connected=2
n_residual_blocks=2
n_hidden=1000

bin/train.py $model_dir \
    $data_dir/train.h5 \
    $data_dir/validation.h5 \
    non_word_marked_chars \
    --target-name multiclass_correction_target \
    --n-embeddings 255 \
    --model-cfg n_embed_dims=$n_embed_dims n_filters=$n_filters filter_width=$filter_width n_fully_connected=$n_fully_connected n_residual_blocks=$n_residual_blocks n_hidden=$n_hidden patience=10 \
    --class-weight-exponent 3 \
    --verbose \
    --no-save
