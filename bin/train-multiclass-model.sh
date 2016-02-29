#!/bin/bash

model_dir=models/keras/spelling/correction/isolated/multiclass/
data_dir=data/spelling/experimental/

experiment_name=$(echo $0 | sed -r 's/.*-(exp[0-9][0-9]-..*).sh/\1/')
experiment_dir=$model_dir/$experiment_name
mkdir -p $experiment_dir

n_embed_dims=10
n_filters=5000
filter_width=6
n_fully_connected=2
n_residual_blocks=5
n_hidden=500

#corpora="non-word-error-detection-experiment-04-random-negative-examples.h5 non-word-error-detection-experiment-04-generated-negative-examples.h5"
#corpora="non-word-error-detection-experiment-04-random-negative-examples.h5"
corpora="non-word-error-detection-experiment-04-generated-negative-examples.h5"

for corpus in $corpora
do
    model_dest=$experiment_dir/$(echo $corpus | sed -e 's,-,_,g' -e 's,.h5,,')
    if [ -d $model_dest ]
    then
        continue
    fi
    bin/train_keras_simple.py $model_dir \
        $data_dir/$corpus \
        $data_dir/$corpus \
        non_word_marked_chars \
        --target-name multiclass_correction_target \
        --model-dest $model_dest \
        --n-embeddings 255 \
        --model-cfg n_embed_dims=$n_embed_dims n_filters=$n_filters filter_width=$filter_width n_fully_connected=$n_fully_connected n_residual_blocks=$n_residual_blocks n_hidden=$n_hidden patience=10 batch_normalization=true \
        --class-weight-exponent 3 \
        --verbose \
        --n-epochs 20 \
        --no-save
done 
        #--log
#| parallel --gnu -j 2
