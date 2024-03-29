#!/bin/sh
 
for i in {1..10}; do python3 /next/run/fashion_dataset/run_NEXT_model.py --main_folder /next/result/fashion_dataset_no_ext_signal_seed$i --nb_max_epoch 300 --nb_iteration_per_epoch 100 --learning_rate 0.0005 --batch_size 2048 --gpu_number 0 --seed $i --nb_hidden_state 2 --main_signal_dataset_path /next/data/fashion_dataset/f1_main.csv --ext_signal_dataset_path /next/data/fashion_dataset/f1_fashion_forward.csv --no_ext_signal; done

for i in {1..10}; do python3 /next/run/fashion_dataset/run_NEXT_model.py --main_folder /next/result/fashion_dataset_with_ext_signal_seed$i --nb_max_epoch 300 --nb_iteration_per_epoch 100 --learning_rate 0.0005 --batch_size 2048 --gpu_number 0 --seed $i --nb_hidden_state 2 --main_signal_dataset_path /next/data/fashion_dataset/f1_main.csv --ext_signal_dataset_path /next/data/fashion_dataset/f1_fashion_forward.csv; done
