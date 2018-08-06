# Opinion Lab

The repository contains codebase for building state-of-the-art deep learning techniques for opinion mining.

Currenly following models have been created and tested:

* Models for IEST @ WASSA-2018 [FNN, CNN, LSTM, CNN-LSTM, LSTM-CNN ++]

## Prerequisites
1. Related resources for featureizers (only if you are using them) (ex: lexicons, word-embedding models)
2. Some python knowledge

## Setup the Lab
1. Install [`textkit-learn`](https://github.com/ysenarath/textkit-learn)
2. Configure path to resources in [`oplab/config.py`](https://github.com/ysenarath/opinion-lab/blob/master/oplab/config.py)
3. Happy Experimenting!

## How to Start
* To train a model its simple as calling
      `python oplab train -c [config_file] -m [model_path]`

* To evaluate just 
      `python oplab evaluate -m [model_path] -e [evaluation_metrics]  -o [predictions_file]`

## Licensing
* Code is under [Apache License 2.0](https://github.com/ysenarath/opinion-lab/blob/master/LICENSE) license.
* Model (releases) are under [Creative Commons Attribution - Non Commercial 2.0 Generic](https://creativecommons.org/licenses/by-nc/2.0/uk/legalcode) license.

