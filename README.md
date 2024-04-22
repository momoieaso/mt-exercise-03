# MT Exercise 3: Pytorch RNN Language Models

This repo shows how to train neural language models using [Pytorch example code](https://github.com/pytorch/examples/tree/master/word_language_model). Thanks to Emma van den Bold, the original author of these scripts. 

# Requirements

- This only works on a Unix-like system, with bash.
- Python 3 must be installed on your system, i.e. the command `python3` must be available
- Make sure virtualenv is installed on your system. To install, e.g.

    `pip install virtualenv`

# Steps

Clone this repository in the desired place:

    git clone https://github.com/momoieaso/mt-exercise-03.git
    cd mt-exercise-03

Create a new virtualenv that uses Python 3. Please make sure to run this command outside of any virtual Python environment:

    ./scripts/make_virtualenv.sh

**Important**: Then activate the env by executing the `source` command that is output by the shell script above.

Download and install required software:

    ./scripts/install_packages.sh

Download and preprocess data:

    ./scripts/download_data.sh

Before training a model, update the main.py to adapt to the new data set, i.e. replace /tools/pytorch-examples/word_language_model/main.py (the old one) with the /scripts/main.py (the new one): 

    cp ./scripts/main.py ./tools/pytorch-examples/word_language_model/main.py

Train a model:

    ./scripts/train.sh

The training process can be interrupted at any time, and the best checkpoint will always be saved.

Generate (sample) some text from a trained model with:

    ./scripts/generate.sh

Create tables for the three different perplexities and line plots for the training and validation perplexity to visualize the results:

    cd scripts
    python3 plot.py


# Scripts Modifications

## `download_data.sh`

**Changes Made:**
1. The dataset URL has been updated to reflect the new dataset source we have identified.
2. In the data preprocessing section, we have updated all relevant paths and filenames to align with our new dataset. 

## `main.py`

**Additions:**
- Imported the package of logging, added a flag to save the perplexities as a log-file and introduced initialization of log configuration to help in debugging and recording operations. 
- Noted down the log files after every print of the perplexities. 

**Reminder:**
- Before training a model, update the main.py to adapt to the new data set, i.e. replace /tools/pytorch-examples/word_language_model/main.py (the old one) with the /scripts/main.py (the new one). 

## `generate.sh`

**Changes Made:**
The script uses two different models for text generation:
1. model_dropout_0.3.pt to generate a text file named sample_lowest_perplexity.txt.
2. model_dropout_0.pt to generate a text file named sample_highest_perplexity.txt.

## `train.sh`

**Updates:**
1. Added a command to create a directory for log files.
2. Defined five different dropout values (0, 0.1, 0.3, 0.6, and 0.8) and set up a loop to train a model for each value, logging the perplexity to the log files.
3. Updated the embedding size and the number of hidden units to 250.
4. Changed the log-interval to 98. 

## `plot.py`

**Description:**
This script analyzes and visualizes the perplexity of a language model across various dropout values during the training, validation, and testing phases. It parses the model training log files, extracts perplexity information, and generates data tables and line plots to compare the model's performance at different stages of training.

