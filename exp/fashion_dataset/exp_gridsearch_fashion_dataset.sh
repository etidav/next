#!/bin/sh
 
for lr in 0.005 0.0005 0.00005; do for batch_size in 64 256 1024 2048; do python3 /next/run/fashion_dataset/run_NEXT_model.py --main_folder /next/result/fashion_dataset_no_ext_signal_lr${lr}_batch_size${batch_size} --nb_max_epoch 200 --nb_iteration_per_epoch 100 --learning_rate $lr --batch_size $batch_size --gpu_number 0 --seed 0 --nb_hidden_state 2 --main_signal_dataset_path /next/data/fashion_dataset/f1_main.csv --ext_signal_dataset_path /next/data/fashion_dataset/f1_fashion_forward.csv --no_ext_signal; done; done