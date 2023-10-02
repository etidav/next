# Variational quantization for state space models

Authors: Etienne David, Jean Bellot and Sylvain Le Corff

Paper link: 

### Abstract
> Forecasting tasks using large datasets gathering thousands of heterogeneous time series is a crucial statistical problem in numerous sectors. The main challenge is to model a rich variety of time series, leverage any available external signals and provide sharp predictions with statistical guarantees. In this work, we propose a new forecasting model that combines discrete state space hidden Markov models with recent neural network architectures and training procedures inspired by vector quantized variational autoencoders. We introduce a variational discrete posterior distribution of the latent states given the observations and a two-stage training procedure to alternatively train the parameters of the latent states and of the emission distributions. By learning a collection of emission laws and temporarily activating them depending on the hidden process dynamics, the proposed method allows to explore large datasets and leverage available external signals. We assess the performance of the proposed method using several datasets and show that it outperforms other state-of-the-art solutions.

## Code Organisation

This repository provides a code base to reproduce results presented in [paper](). The repository is organized as follow:

 - [model/](model/): Directory gathering the code of the model and training process
 - [run/](run/): Directory gathering scripts to train the model.
 - [exp/](exp/): Directory gathering script with commande lines to reproduce results of the paper.
 - [data/](data/): Empty directory where to put datasets on which the model will be train (to download datasets used in the paper: [Fashion dataset](https://github.com/etidav/HERMES),[Reference datasets](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy).
 - [docker/](docker/): directory gathering the code to build a docker so as to recover results of the paper.  

## Reproduce benchmark results

First, you should build, run and enter into the docker. In the main folder, run
```bash
make build run enter
```
Then, open one of the file of the dir [exp/](exp/) to find commande lines to reproduce the different results.
