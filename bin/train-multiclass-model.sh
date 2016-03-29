#!/bin/bash

model_dir=models/multiclass/
if [ $# -eq 0 ]
then
    echo "usage: train.py --mode MODE KEY=VALUE KEY=VALUE ... KEY=VALUE" >&2
    echo "       MODE can be 'transient', 'persistent', or 'persistent-background'" >&2
    echo "       KEY=VALUE pairs are model arguments that may also be set in model.json" >&2
    exit 1
fi

mode=$1
shift

if [ $# -eq 0 ]
then
    bin/train.py $model_dir --mode $mode
else
    bin/train.py $model_dir --mode $mode --model-cfg "$@"
fi

#bin/train.py $model_dir \
#    $data_dir/train.h5 \
#    $data_dir/validation.h5 \
#    non_word_marked_chars \
#    --target-name multiclass_correction_target \
#    --n-embeddings 255 \
#    --model-cfg n_embed_dims=$n_embed_dims n_filters=$n_filters filter_width=$filter_width n_fully_connected=$n_fully_connected n_residual_blocks=$n_residual_blocks n_hidden=$n_hidden patience=10 \
#    --class-weight-exponent 3 \
#    --verbose \
#    --no-save
