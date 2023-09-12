#!/bin/sh

# ETTh2
for hidden_state in 1 2 3 4; do python3 /next/run/reference_dataset/run_NEXT_model.py --main_folder /next/result/etth2_hidden_state_${hidden_state} --dataset_path /next/data/reference_dataset/ETT-small/ETTh2.csv --past_dependency 480 --season 144 --horizon 96 --nb_max_epoch 50 --gpu_number 0 --seed 0 --nb_hidden_state $hidden_state; done

# weather
for hidden_state in 1 2 3 4; do python3 /next/run/reference_dataset/run_NEXT_model.py --main_folder /next/result/weather_hidden_state_${hidden_state} --dataset_path /next/data/reference_dataset/weather/weather.csv --past_dependency 720 --season 144 --horizon 96 --nb_max_epoch 50 --gpu_number 0 --seed 0 --nb_hidden_state $hidden_state; done

# traffic
for hidden_state in 1 2 3 4; do python3 /next/run/reference_dataset/run_NEXT_model.py --main_folder /next/result/traffic_hidden_state_${hidden_state} --dataset_path /next/data/reference_dataset/traffic/traffic.csv --past_dependency 672 --season 168 --horizon 96 --nb_max_epoch 50 --gpu_number 0 --seed 0 --nb_hidden_state $hidden_state; done