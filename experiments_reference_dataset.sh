#!/bin/sh
 
# weather
for horizon in 96 192 336 720; do python3 /next/run/reference_dataset/run_NEXT_model.py --main_folder /next/result/weather_horizon$horizon --dataset_path /next/data/reference_dataset/weather/weather.csv --past_dependency 288 --season 144 --horizon $horizon --nb_max_epoch 200 --nb_iteration_per_epoch 50 --learning_rate 0.0005 --batch_size 1024 --gpu_number 0 --seed 0; done

# ILI
for horizon in 24 36 48 60; do python3 /next/run/reference_dataset/run_NEXT_model.py --main_folder /next/result/ili_horizon$horizon --dataset_path /next/data/reference_dataset/illness/national_illness.csv --past_dependency 104 --season 52 --horizon $horizon --nb_max_epoch 200 --nb_iteration_per_epoch 50 --learning_rate 0.0005 --batch_size 1024 --gpu_number 0 --seed 0 --small_model; done

# ETTm2
for horizon in 96 192 336 720; do python3 /next/run/reference_dataset/run_NEXT_model.py --main_folder /next/result/ettm2_horizon$horizon --dataset_path /next/data/reference_dataset/ETT-small/ETTm2.csv --past_dependency 192 --season 96 --horizon $horizon --nb_max_epoch 200 --nb_iteration_per_epoch 50 --learning_rate 0.0005 --batch_size 1024 --gpu_number 0 --seed 0 --small_model; done

# exchange_rate
for horizon in 96 192 336 720; do python3 /next/run/reference_dataset/run_NEXT_model.py --main_folder /next/result/exchange_rate_horizon$horizon --dataset_path /next/data/reference_dataset/exchange_rate/exchange_rate.csv --past_dependency 7 --season 365 --horizon $horizon --nb_max_epoch 200 --nb_iteration_per_epoch 50 --learning_rate 0.0005 --batch_size 1024 --gpu_number 0 --seed 0 --small_model; done

# traffic
for horizon in 96 192 336 720; do python3 /next/run/reference_dataset/run_NEXT_model.py --main_folder /next/result/traffic_horizon$horizon --dataset_path /next/data/reference_dataset/traffic/traffic.csv --past_dependency 336 --season 168 --horizon $horizon --nb_max_epoch 200 --nb_iteration_per_epoch 50 --learning_rate 0.0005 --batch_size 1024 --gpu_number 0 --seed 0; done

# electricity
for horizon in 96 192 336 720; do python3 /next/run/reference_dataset/run_NEXT_model.py --main_folder /next/result/electricity_horizon$horizon --dataset_path /next/data/reference_dataset/electricity/electricity.csv --past_dependency 336 --season 168 --horizon $horizon --nb_max_epoch 200 --nb_iteration_per_epoch 50 --learning_rate 0.0005 --batch_size 1024 --gpu_number 0 --seed 0; done