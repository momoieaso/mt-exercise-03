#! /bin/bash

scripts=$(dirname "$0")
base=$(realpath $scripts/..)

models=$base/models
data=$base/data
tools=$base/tools

mkdir -p $models
mkdir -p $models/logs  

num_threads=4
device=""

SECONDS=0

dropout_values=(0 0.1 0.3 0.6 0.8)

for dropout in "${dropout_values[@]}"
do
    echo "Training with dropout $dropout"
    logfile=$models/logs/perplexity_dropout_${dropout}.log  

    (cd $tools/pytorch-examples/word_language_model &&
        CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python main.py --data $data/arnim \
            --epochs 40 \
            --log-interval 98 \
            --emsize 250 --nhid 250 --dropout $dropout --tied \
            --save $models/model_dropout_$dropout.pt \
            --log-file $logfile
    )
done

echo "time taken:"
echo "$SECONDS seconds"
