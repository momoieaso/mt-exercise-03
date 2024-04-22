#! /bin/bash

scripts=$(dirname "$0")
base=$scripts/..

data=$base/data

mkdir -p $data

tools=$base/tools

# link default training data for easier access

mkdir -p $data/wikitext-2

for corpus in train valid test; do
    absolute_path=$(realpath $tools/pytorch-examples/word_language_model/data/wikitext-2/$corpus.txt)
    ln -snf $absolute_path $data/wikitext-2/$corpus.txt
done

# download a different interesting data set!

mkdir -p $data/arnim

mkdir -p $data/arnim/raw

wget https://www.gutenberg.org/cache/epub/16389/pg16389.txt
mv pg16389.txt $data/arnim/raw/april.txt

# preprocess slightly

cat $data/arnim/raw/april.txt | python $base/scripts/preprocess_raw.py > $data/arnim/raw/april.cleaned.txt

# tokenize, fix vocabulary upper bound

cat $data/arnim/raw/april.cleaned.txt | python $base/scripts/preprocess.py --vocab-size 5000 --tokenize --lang "en" --sent-tokenize > \
    $data/arnim/raw/april.preprocessed.txt

# split into train, valid and test

head -n 440 $data/arnim/raw/april.preprocessed.txt | tail -n 400 > $data/arnim/valid.txt
head -n 840 $data/arnim/raw/april.preprocessed.txt | tail -n 400 > $data/arnim/test.txt
tail -n 3075 $data/arnim/raw/april.preprocessed.txt | head -n 2955 > $data/arnim/train.txt
