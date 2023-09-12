# NEXT: Probabilistic forecasting relying on quantized hidden states to deeper explore time series diversity.

Authors: Etienne David, Jean Bellot and Sylvain Le Corff

Paper link: 

### Abstract
> 

## Code Organisation

This repository provides the code of the NEXT model and a simple code base to reproduce results presented in this [paper](). The repository is organized as follow:

 - [model/](model/): Directory gathering the code of the NEXT model.
 - [run/](run/): Directory gathering the different scripts to reproduce results of the paper.
 - [data/](data/): Directory where the dataset used in the paper have to be stored.
 - [docker/](docker/): directory gathering the code to build a docker and recover the exact result of the paper.  

## Reproduce benchmark results

First, if you want to reproduce results on the fashion dataset, store the fashion dataset on the directory [data/](data/). The fashion dataset is available [here](https://github.com/etidav/HERMES/blob/main/data/f1_main.csv). If you want to reproduce results on the 6 benchmarks dataset, store them in the directory [data/](data/). They are available in the following google drive directory provided with the Informer paper [here](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy).

Secondly, build, run and enter into the NEXT docker. In the main folder, run
```bash
make build run enter
```

To reproduce the result on the fashion dataset:
- [run/fashion_dataset/run_NEXT_model.py](run/fashion_dataset/run_NEXT_model.py)
run
```bash
python run/fashion_dataset/run_NEXT_model.py --help # display the default parameters and their description
```
To reproduce the result on the reference datasets:
- [run/reference_dataset/run_NEXT_model.py](run/reference_dataset/run_NEXT_model.py)
run
```bash
python run/reference_dataset/run_NEXT_model.py --help # display the default parameters and their description
```

## paper results
