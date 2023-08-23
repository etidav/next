#!/bin/sh
 
# weather
for horizon in 96 192 336 720; do python3 /next/run/reference_dataset/run_NEXT_model.py --main_folder /next/result/weather_horizon$horizon --dataset_path /next/data/reference_dataset/weather/weather.csv --past_dependency 336 --season 144 --horizon horizon --nb_max_epoch 2 --nb_iteration_per_epoch 100 --learning_rate 0.005 --batch_size 2048 --gpu_number 1 --seed 0; done

# ILI
for horizon in 96 192 336 720; do python3 /next/run/reference_dataset/run_NEXT_model.py --main_folder /next/result/ili_horizon$horizon --dataset_path /next/data/reference_dataset/illness/national_illness.csv --past_dependency 104 --season 52 --horizon 24 --nb_max_epoch 2 --nb_iteration_per_epoch 100 --learning_rate 0.005 --batch_size 2048 --gpu_number 1 --seed 0

# ETTm2
for horizon in 96 192 336 720; do python3 /next/run/reference_dataset/run_NEXT_model.py --main_folder /next/result/ettm2_horizon$horizon --dataset_path /next/data/reference_dataset/ETT-small/ETTm2.csv --past_dependency 336 --season 96 --horizon horizon --nb_max_epoch 2 --nb_iteration_per_epoch 100 --learning_rate 0.005 --batch_size 2048 --gpu_number 1 --seed 0

# exchange_rate
for horizon in 96 192 336 720; do python3 /next/run/reference_dataset/run_NEXT_model.py --main_folder /next/result/exchange_rate_horizon$horizon --dataset_path /next/data/reference_dataset/exchange_rate/exchange_rate.csv --past_dependency 336 --season 365 --horizon horizon --nb_max_epoch 2 --nb_iteration_per_epoch 100 --learning_rate 0.005 --batch_size 2048 --gpu_number 1 --seed 0

# traffic
for horizon in 96 192 336 720; do python3 /next/run/reference_dataset/run_NEXT_model.py --main_folder /next/result/traffic_horizon$horizon --dataset_path /next/data/reference_dataset/traffic/traffic.csv --past_dependency 336 --season 168 --horizon horizon --nb_max_epoch 2 --nb_iteration_per_epoch 100 --learning_rate 0.005 --batch_size 2048 --gpu_number 1 --seed 0

# electricity
for horizon in 96 192 336 720; do python3 /next/run/reference_dataset/run_NEXT_model.py --main_folder /next/result/electricity_horizon$horizon --dataset_path /next/data/reference_dataset/electricity/electricity.csv --past_dependency 336 --season 168 --horizon horizon --nb_max_epoch 2 --nb_iteration_per_epoch 100 --learning_rate 0.005 --batch_size 2048 --gpu_number 1 --seed 0