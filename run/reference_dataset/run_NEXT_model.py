import argparse
import os
import sys
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import tensorflow as tf
from utils.utils import write_json
from model.NEXT_model import next_model_reference_dataset, next_model_reference_dataset_small


def compute_accuracy_metric(dataset, model, horizon, preprocess_input):

    result = {}
    result["mae"] = []
    result["mse"] = []
    nb_test_window = int(dataset.shape[0] * 0.2) - horizon + 1
    for i in tqdm(range(nb_test_window)):
        y_signal = dataset[: -horizon - i]
        w_signal = pd.DataFrame(np.zeros_like(y_signal))
        model_prediction = model.predict(
            y_signal=y_signal,
            w_signal=w_signal,
            nb_simulation=1,
            preprocess_input=preprocess_input
        )
        model_prediction = model_prediction['y_pred_mean'].T

        if i:
            result["mae"].append(
                np.mean(np.abs(dataset[-horizon - i : -i] - model_prediction))
            )
            result["mse"].append(
                np.mean(np.square(dataset[-horizon - i : -i] - model_prediction))
            )
        else:
            result["mae"].append(
                np.mean(np.abs(dataset[-horizon:] - model_prediction))
            )
            result["mse"].append(
                np.mean(np.square(dataset[-horizon:] - model_prediction))
            )
            
    for metric in result:
        result[metric] = np.round(np.mean(result[metric]), 3)

    return result


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="train federate HMM")
    parser.add_argument(
        "--main_folder", type=str, help="where to store the model files", required=True
    )
    parser.add_argument(
        "--dataset_path", type=str, help="path to a dataset", required=True
    )
    parser.add_argument("--past_dependency", type=int, help="", required=True)
    parser.add_argument("--season", type=int, help="", required=True)
    parser.add_argument("--horizon", type=int, help="", required=True)
    parser.add_argument("--nb_max_epoch", type=int, help="", default=50)
    parser.add_argument("--nb_iteration_per_epoch", type=int, help="", default=50)
    parser.add_argument("--learning_rate", type=float, help="", default=0.05)
    parser.add_argument("--optimizer_name", type=str, help="", default="adam")
    parser.add_argument("--batch_size", type=int, help="", default=None)
    parser.add_argument("--seed", type=int, help="", default=1)
    parser.add_argument("--gpu_number", type=int, help="", default=0)
    parser.add_argument(
        "--small_model",
        action="store_true",
        help="add this argument to train a small version of the next model without lstm components",
    )

    args = parser.parse_args()

    gpu_number = args.gpu_number
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus):
        tf.config.set_visible_devices(gpus[gpu_number], "GPU")
        tf.config.set_logical_device_configuration(
            gpus[gpu_number], [tf.config.LogicalDeviceConfiguration(memory_limit=20000)]
        )
    seed = args.seed
    np.random.seed(seed)
    tf.random.set_seed(seed)

    main_folder = args.main_folder
    dataset_path = args.dataset_path
    past_dependency = args.past_dependency
    season = args.season
    horizon = args.horizon
    preprocess_input = False
    nb_max_epoch = args.nb_max_epoch
    nb_iteration_per_epoch = args.nb_iteration_per_epoch
    learning_rate = args.learning_rate
    optimizer_name = args.optimizer_name
    batch_size = args.batch_size
    small_model = args.small_model

    if main_folder[-1] != "/":
        main_folder += "/"
    if not os.path.exists(main_folder):
        os.makedirs(main_folder)

    config_dict = vars(args)
    write_json(config_dict, os.path.join(main_folder, "config.json"))

    dataset = pd.read_csv(dataset_path, index_col=0)
    dataset.index = pd.to_datetime(dataset.index)
    if os.path.basename(dataset_path) == 'ETTm2.csv':
        dataset = dataset[:57600]
        train_size = int(dataset.shape[0] * 0.6)
    else:
        train_size = int(dataset.shape[0] * 0.7)
    train_and_eval_size = int(dataset.shape[0] * 0.8)
    all_y_data = (dataset - dataset[:train_size].mean(axis=0)) / dataset[
        :train_size
    ].std(axis=0)
    
    y_train = all_y_data[:train_and_eval_size]
    w_train = pd.DataFrame(np.zeros_like(y_train))

    if small_model:
        model = next_model_reference_dataset_small(
            nb_hidden_states=2, past_dependency=past_dependency, season=season, horizon=horizon,
        )
    else:
        model = next_model_reference_dataset_small(
            nb_hidden_states=2, past_dependency=past_dependency, season=season, horizon=horizon,
        )

    print("Start Training")
    model.fit(
        y_signal=y_train,
        w_signal=w_train,
        nb_max_epoch=nb_max_epoch,
        nb_iteration_per_epoch=nb_iteration_per_epoch,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        batch_size=batch_size,
        model_folder=main_folder,
        preprocess_input=preprocess_input
    )

    print("Compute accuracy")
    model_accuracy = compute_accuracy_metric(all_y_data, model, horizon, preprocess_input)
    write_json(model_accuracy, os.path.join(main_folder, "final_accuracy.json"))
    print(model_accuracy)
