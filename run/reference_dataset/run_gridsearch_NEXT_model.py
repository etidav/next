import argparse
import os
import sys
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import tensorflow as tf
from utils.utils import write_json
from utils.reference_dataset_accuracy_metrics import (
    compute_reference_dataset_accuracy_metrics,
)
from model.NEXT_model import next_model_reference_dataset


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="train federate HMM")
    parser.add_argument(
        "--main_folder", type=str, help="where to store the model files", required=True
    )
    parser.add_argument(
        "--dataset_path", type=str, help="path to a dataset", required=True
    )
    parser.add_argument("--nb_hidden_state", type=int, help="", default=3)
    parser.add_argument("--season", type=int, help="", required=True)
    parser.add_argument("--horizon", type=int, help="", required=True)
    parser.add_argument("--nb_max_epoch", type=int, help="", default=50)
    parser.add_argument("--seed", type=int, help="", default=1)
    parser.add_argument("--gpu_number", type=int, help="", default=0)

    args = parser.parse_args()

    gpu_number = args.gpu_number
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus):
        tf.config.set_visible_devices(gpus[gpu_number], "GPU")
        tf.config.set_logical_device_configuration(
            gpus[gpu_number], [tf.config.LogicalDeviceConfiguration(memory_limit=10000)]
        )
    seed = args.seed
    np.random.seed(seed)
    tf.random.set_seed(seed)

    main_folder = args.main_folder
    dataset_path = args.dataset_path
    season = args.season
    horizon = args.horizon
    nb_hidden_state = args.nb_hidden_state
    nb_max_epoch = args.nb_max_epoch

    if main_folder[-1] != "/":
        main_folder += "/"
    if not os.path.exists(main_folder):
        os.makedirs(main_folder)

    config_dict = vars(args)
    write_json(config_dict, os.path.join(main_folder, "config.json"))

    dataset = pd.read_csv(dataset_path, index_col=0)
    dataset.index = pd.to_datetime(dataset.index)
    if os.path.basename(dataset_path) in [
        "ETTm1.csv",
        "ETTm2.csv",
        "ETTh1.csv",
        "ETTh2.csv",
    ]:
        dataset = dataset.loc[:"2018-02-20"]
        train_size = int(dataset.shape[0] * 0.6)
    else:
        train_size = int(dataset.shape[0] * 0.7)
    train_and_eval_size = int(dataset.shape[0] * 0.8)
    all_y_data = (dataset - dataset[:train_size].mean(axis=0)) / dataset[
        :train_size
    ].std(axis=0)
    eval_size = int(dataset.shape[0] * 0.1)
    preprocess_name = "minmaxscaler"

    y_train = all_y_data[:train_and_eval_size]
    if os.path.basename(dataset_path) == "weather.csv":
        y_train = y_train.replace(y_train.min(), np.nan)
        y_train = y_train.fillna(y_train.mean())
    w_train = pd.DataFrame(np.zeros_like(y_train))

    best_model_eval = np.inf
    best_training_config = {}
    for past_dependency in [int(i * season) for i in [0.5,1,2,3,4,5]]:
        model_folder = os.path.join(main_folder, f"past_dependency_{past_dependency}",)
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        model = next_model_reference_dataset(
            nb_hidden_states=nb_hidden_state,
            past_dependency=past_dependency,
            season=season,
            horizon=horizon,
            preprocess_name=preprocess_name,
        )

        print("Start Training")
        model.fit(
            y_signal=y_train,
            w_signal=w_train,
            nb_max_epoch=nb_max_epoch,
            nb_iteration_per_epoch=50,
            optimizer_name="adam",
            learning_rate=0.00005,
            batch_size=2048,
            eval_size=eval_size,
            model_folder=model_folder,
        )
        model_accuracy = compute_reference_dataset_accuracy_metrics(
            all_y_data, model, horizon, True
        )
        write_json(model_accuracy, os.path.join(model_folder, "final_accuracy.json"))
        if model_accuracy["eval"]["mse"] < best_model_eval:
            best_model_eval = model_accuracy["eval"]["mse"]
            best_training_config = {
                "past_dependency": past_dependency,
            }

    write_json(best_training_config, os.path.join(main_folder, "best_config.json"))
