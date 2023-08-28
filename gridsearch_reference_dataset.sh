#!/bin/sh
 
# weather
python3 /next/run/reference_dataset/run_gridsearch_NEXT_model.py --main_folder /next/result/weather_gridsearch --dataset_path /next/data/reference_dataset/weather/weather.csv --past_dependency_list 144 288 432 576 --season 144 --horizon 96 --nb_max_epoch 50 --nb_iteration_per_epoch 50 --learning_rate_list 0.005 --batch_size_list 2048 --gpu_number 0 --seed 0

# ILI
python3 /next/run/reference_dataset/run_gridsearch_NEXT_model.py --main_folder /next/result/ili_gridsearch --dataset_path /next/data/reference_dataset/illness/national_illness.csv --past_dependency_list 13 26 52 78 104 --season 52 --horizon 24 --nb_max_epoch 50 --nb_iteration_per_epoch 50 --learning_rate_list 0.005 --batch_size_list 2048 --gpu_number 0 --seed 0 --small_model

# ETTm2
python3 /next/run/reference_dataset/run_gridsearch_NEXT_model.py --main_folder /next/result/ettm2_gridsearch --dataset_path /next/data/reference_dataset/ETT-small/ETTm2.csv --past_dependency_list 96 192 288 384 480 --season 96 --horizon 96 --nb_max_epoch 50 --nb_iteration_per_epoch 50 --learning_rate_list 0.005 --batch_size_list 2048 --gpu_number 0 --seed 0 --small_model

# exchange_rate
python3 /next/run/reference_dataset/run_gridsearch_NEXT_model.py --main_folder /next/result/exchange_rate_gridsearch --dataset_path /next/data/reference_dataset/exchange_rate/exchange_rate.csv --past_dependency_list 7 28 56 84 182 --season 365 --horizon 92 --nb_max_epoch 50 --nb_iteration_per_epoch 50 --learning_rate_list 0.005 --batch_size_list 2048 --gpu_number 0 --seed 0 --small_model

# traffic
python3 /next/run/reference_dataset/run_gridsearch_NEXT_model.py --main_folder /next/result/traffic_gridsearch --dataset_path /next/data/reference_dataset/traffic/traffic.csv --past_dependency_list 168 336 504 672 --season 168 --horizon 96 --nb_max_epoch 50 --nb_iteration_per_epoch 50 --learning_rate_list 0.005 --batch_size_list 2048 --gpu_number 0 --seed 0

# electricity
python3 /next/run/reference_dataset/run_gridsearch_NEXT_model.py --main_folder /next/result/electricity_gridsearch --dataset_path /next/data/reference_dataset/electricity/electricity.csv --past_dependency_list 168 336 504 672 --season 168 --horizon 96 --nb_max_epoch 50 --nb_iteration_per_epoch 50 --learning_rate_list 0.005 --batch_size_list 2048 --gpu_number 0 --seed 0
