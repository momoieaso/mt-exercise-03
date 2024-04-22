#! /bin/bash

scripts=$(dirname "$0")
base=$(realpath $scripts/..)

models=$base/models
data=$base/data
tools=$base/tools
samples=$base/samples

mkdir -p $samples

num_threads=4
device=""

(cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python generate.py \
        --data $data/arnim \
        --words 100 \
        --checkpoint $models/model.pt \
        --outf $samples/sample.txt
)


(cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python generate.py \
        --data $data/arnim \
        --words 100 \
        --checkpoint $models/model_dropout_0.3.pt \
        --outf $samples/sample_lowest_perplexity.txt
)


(cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python generate.py \
        --data $data/arnim \
        --words 100 \
        --checkpoint $models/model_dropout_0.pt \
        --outf $samples/sample_highest_perplexity.txt
)