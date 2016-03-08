#!/bin/bash

model_dir=models/binary/

if [ $# -ne 1 ]
then
    echo "usage: train.py --mode MODE KEY=VALUE KEY=VALUE ... KEY=VALUE" >&2
    echo "       MODE can be 'transient', 'persistent', or 'persistent-background'" >&2
    echo "       KEY=VALUE pairs are model arguments that may also be set in model.json" >&2
    exit 1
fi

mode=$1
shift

patience=2

script_name=$(echo $0 | sed -e 's,.*bin/,,' -e 's,.sh,,' -e 's,-,_,g')
model_path=$model_dir/$script_name/
mkdir -p $model_path

for pool_merge_mode in cos dot mul 
do
    for n_embed_dims in 2 10 20
    do
        n_embed_dims_fmt=$(printf "%02g" $n_embed_dims)
        for n_filters in 10 100 200
        do
            n_filters_fmt=$(printf "%03g" $n_filters)
            for filter_width in 5 10
            do
                filter_width_fmt=$(printf "%02g" $filter_width)
                for n_hidden in "" 10 100
                do
                    if [ "$n_hidden" == "" ]
                    then
                        nhid_fmt="000"
                    else
                        nhid_fmt=$(printf "%03g" $n_hidden)
                    fi
                    fully_connected="[$n_hidden]"

                    model_dest="$model_path/${pool_merge_mode}_${n_embed_dims_fmt}_${n_filters_fmt}_${filter_width_fmt}_${nhid_fmt}"

                    echo -e "$model_dest \t $pool_merge_mode \t $n_embed_dims \t $n_filters \t $filter_width \t $fully_connected"
                done
            done
        done
    done
done | parallel --gnu --jobs 8 --colsep '\t' --verbose bin/train.py $model_dir --mode $mode --model-dest {1} --model-cfg patience=$patience pool_merge_mode={2} n_embed_dims={3} n_filters={4} filter_width={5} fully_connected='{6}'
