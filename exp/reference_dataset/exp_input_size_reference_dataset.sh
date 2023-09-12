#!/bin/sh

# ETTm2
python3 /next/run/reference_dataset/run_gridsearch_NEXT_model.py --main_folder /next/result/ettm2_gridsearch --dataset_path /next/data/reference_dataset/ETT-small/ETTm2.csv --season 96 --horizon 96 --nb_max_epoch 50 --gpu_number 0 --seed 0

# ETTm1
python3 /next/run/reference_dataset/run_gridsearch_NEXT_model.py --main_folder /next/result/ettm1_gridsearch --dataset_path /next/data/reference_dataset/ETT-small/ETTm1.csv --season 96 --horizon 96 --nb_max_epoch 50 --gpu_number 0 --seed 0

# ETTh2
python3 /next/run/reference_dataset/run_gridsearch_NEXT_model.py --main_folder /next/result/etth2_gridsearch --dataset_path /next/data/reference_dataset/ETT-small/ETTh2.csv --season 168 --horizon 96 --nb_max_epoch 50 --gpu_number 0 --seed 0

# ETTh1
python3 /next/run/reference_dataset/run_gridsearch_NEXT_model.py --main_folder /next/result/etth1_gridsearch --dataset_path /next/data/reference_dataset/ETT-small/ETTh1.csv --season 168 --horizon 96 --nb_max_epoch 50 --gpu_number 0 --seed 0

# ILI
python3 /next/run/reference_dataset/run_gridsearch_NEXT_model.py --main_folder /next/result/ili_gridsearch --dataset_path /next/data/reference_dataset/illness/national_illness.csv --season 52 --horizon 24 --nb_max_epoch 50 --gpu_number 0 --seed 0

# weather
python3 /next/run/reference_dataset/run_gridsearch_NEXT_model.py --main_folder /next/result/weather_gridsearch --dataset_path /next/data/reference_dataset/weather/weather.csv --season 144 --horizon 96 --nb_max_epoch 50 --gpu_number 0 --seed 0

# traffic
python3 /next/run/reference_dataset/run_gridsearch_NEXT_model.py --main_folder /next/result/traffic_gridsearch --dataset_path /next/data/reference_dataset/traffic/traffic.csv --season 168 --horizon 96 --nb_max_epoch 50 --gpu_number 0 --seed 0

# electricity
python3 /next/run/reference_dataset/run_gridsearch_NEXT_model.py --main_folder /next/result/electricity_gridsearch --dataset_path /next/data/reference_dataset/electricity/electricity.csv --season 168 --horizon 96 --nb_max_epoch 50 --gpu_number 0 --seed 0