#!/bin/sh
 
python3 /next/run/fashion_dataset/run_NEXT_model.py --main_folder /next/result/fashion_dataset_no_ext_signal --nb_max_epoch 1 --nb_iteration_per_epoch 100 --learning_rate 0.005 --batch_size 2048 --gpu_number 1 --seed 0 --nb_hidden_state 2 --main_signal_dataset_path /next/data/fashion_dataset/f1_main.csv --ext_signal_dataset_path /next/data/fashion_dataset/f1_fashion_forward.csv --no_ext_signal
python3 /next/run/fashion_dataset/run_NEXT_model.py --main_folder /next/result/fashion_dataset_with_ext_signal --nb_max_epoch 1 --nb_iteration_per_epoch 100 --learning_rate 0.005 --batch_size 2048 --gpu_number 1 --seed 0 --nb_hidden_state 2 --main_signal_dataset_path /next/data/fashion_dataset/f1_main.csv --ext_signal_dataset_path /next/data/fashion_dataset/f1_fashion_forward.csv
