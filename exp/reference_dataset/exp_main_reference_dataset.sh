#!/bin/sh
 
# ETTm2
for horizon in 96 192 336 720; do python3 /next/run/reference_dataset/run_NEXT_model.py --main_folder /next/result/ettm2_horizon$horizon --dataset_path /next/data/reference_dataset/ETT-small/ETTm2.csv --past_dependency 480 --season 96 --horizon $horizon --nb_max_epoch 50 --gpu_number 0 --seed 0 --nb_hidden_state 3; done

# ETTm1
for horizon in 96 192 336 720; do python3 /next/run/reference_dataset/run_NEXT_model.py --main_folder /next/result/ettm1_horizon$horizon --dataset_path /next/data/reference_dataset/ETT-small/ETTm1.csv --past_dependency 288 --season 96 --horizon $horizon --nb_max_epoch 50 --gpu_number 0 --seed 0 --nb_hidden_state 3; done

# ETTh2
for horizon in 96 192 336 720; do python3 /next/run/reference_dataset/run_NEXT_model.py --main_folder /next/result/etth2_horizon$horizon --dataset_path /next/data/reference_dataset/ETT-small/ETTh2.csv --past_dependency 672 --season 168 --horizon $horizon --nb_max_epoch 50 --gpu_number 0 --seed 0 --nb_hidden_state 3; done

# ETTh1
for horizon in 96 192 336 720; do python3 /next/run/reference_dataset/run_NEXT_model.py --main_folder /next/result/etth1_horizon$horizon --dataset_path /next/data/reference_dataset/ETT-small/ETTh1.csv --past_dependency 84 --season 168 --horizon $horizon --nb_max_epoch 50 --gpu_number 0 --seed 0 --nb_hidden_state 3; done

# ILI
for horizon in 24 36 48 60; do python3 /next/run/reference_dataset/run_NEXT_model.py --main_folder /next/result/ili_horizon$horizon --dataset_path /next/data/reference_dataset/illness/national_illness.csv --past_dependency 104 --season 52 --horizon $horizon --nb_max_epoch 50 --gpu_number 0 --seed 0 --nb_hidden_state 3; done

# weather
for horizon in 96 192 336 720; do python3 /next/run/reference_dataset/run_NEXT_model.py --main_folder /next/result/weather_horizon$horizon --dataset_path /next/data/reference_dataset/weather/weather.csv --past_dependency 720 --season 144 --horizon $horizon --nb_max_epoch 50 --gpu_number 0 --seed 0 --nb_hidden_state 3; done

# traffic
for horizon in 96 192 336 720; do python3 /next/run/reference_dataset/run_NEXT_model.py --main_folder /next/result/traffic_horizon$horizon --dataset_path /next/data/reference_dataset/traffic/traffic.csv --past_dependency 672 --season 168 --horizon $horizon --nb_max_epoch 50 --gpu_number 0 --seed 0 --nb_hidden_state 3; done

# electricity
for horizon in 96 192 336 720; do python3 /next/run/reference_dataset/run_NEXT_model.py --main_folder /next/result/electricity_horizon$horizon --dataset_path /next/data/reference_dataset/electricity/electricity.csv --past_dependency 840 --season 168 --horizon $horizon --nb_max_epoch 50 --gpu_number 0 --seed 0 --nb_hidden_state 3; done