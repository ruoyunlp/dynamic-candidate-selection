## Improving Unsupervised Zero-shot Intent Classification with Dynamic Candidate Selection

#### About

This open-source code repository contains the code for the paper "Improving Unsupervised Zero-shot Intent Classification with Dynamic Candidate Selection" paper. If you find this useful, please cite our paper!


#### Quick Start Guide
`git clone` the project prior to setting up data and models. All path variables in the code are unset and should be set before running code.

#### Data
We provide preprocessed datasets used in our experiments for purposes of reproducibility in `data/preprocessed`. Original files for ATIS can be found [here](https://github.com/howl-anderson/ATIS_dataset/blob/master/README.en-US.md), SNIPS-NLU [here](https://github.com/sonos/nlu-benchmark), CLINC150 [here](https://github.com/clinc/oos-eval) and MASSIVE [here](https://github.com/alexa/massive).

#### Python Environment

This codebase is developed in a Python 3.8.10 environment. Dependencies can be installed via

```
pip install -r requirements.txt
```

#### Repository Structure

`analysis` contains jupyter notebooks that were used to perform analysis and generate some of the graphs/figures seen in the paper.

`data` contains the data used to conduct our experiments. Please note the preprocessed compressed data for the CLINC150 dataset was too large to upload to GitHub but can be generated using the code provided.

`postprocessing` contains code for cleaning and mapping model outputs to intents.