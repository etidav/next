#!/bin/sh

# ETTm2
for preprocess_name in minmaxscaler standardscaler ; do python3 /next/run/reference_dataset/run_NEXT_model.py --main_folder /next/result/ettm2_$preprocess_name --dataset_path /next/data/reference_dataset/ETT-small/ETTm2.csv --past_dependency 480 --season 96 --horizon 96 --nb_max_epoch 50 --gpu_number 0 --seed 0 --preprocess_name $preprocess_name ; done

# ETTm1
for preprocess_name in minmaxscaler standardscaler ; do python3 /next/run/reference_dataset/run_NEXT_model.py --main_folder /next/result/ettm1_$preprocess_name --dataset_path /next/data/reference_dataset/ETT-small/ETTm1.csv --past_dependency 288 --season 96 --horizon 96 --nb_max_epoch 50 --gpu_number 0 --seed 0 --preprocess_name $preprocess_name ; done

# ETTh2
for preprocess_name in minmaxscaler standardscaler ; do python3 /next/run/reference_dataset/run_NEXT_model.py --main_folder /next/result/etth2_$preprocess_name --dataset_path /next/data/reference_dataset/ETT-small/ETTh2.csv --past_dependency 672 --season 168 --horizon 96 --nb_max_epoch 50 --gpu_number 0 --seed 0 --preprocess_name $preprocess_name ; done

# ETTh1
for preprocess_name in minmaxscaler standardscaler ; do python3 /next/run/reference_dataset/run_NEXT_model.py --main_folder /next/result/etth1_$preprocess_name --dataset_path /next/data/reference_dataset/ETT-small/ETTh1.csv --past_dependency 84 --season 168 --horizon 96 --nb_max_epoch 50 --gpu_number 0 --seed 0 --preprocess_name $preprocess_name ; done