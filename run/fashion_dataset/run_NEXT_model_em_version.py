import argparse
import os
import numpy as np
import pandas as pd
import json
import tensorflow as tf
from utils.utils import write_json
from model.NEXT_model_em_version import (
    next_model_no_ext_signal,
    next_model_with_ext_signal,
)


def compute_accuracy_metric(y_train, y_test, y_pred):
    y_train = y_train.values.T
    y_test = y_test.values.T
    mase_denom = np.mean(np.abs(y_train[:, 52:] - y_train[:, :-52]), axis=1)
    mase_list = np.mean(np.abs(y_test - y_pred), axis=1) / mase_denom
    result = {"mase": np.round(np.mean(mase_list), 4)}
    return result


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="train federate HMM")
    parser.add_argument(
        "--main_folder", type=str, help="where to store the model files", required=True
    )
    parser.add_argument(
        "--main_signal_dataset_path",
        type=str,
        help="path to a dataset gathering the main signals",
        required=True,
    )
    parser.add_argument(
        "--ext_signal_dataset_path",
        type=str,
        help="path to a dataset gathering the external signals",
        required=True,
    )
    parser.add_argument("--nb_hidden_state", type=int, help="", default=2)
    parser.add_argument("--nb_max_epoch", type=int, help="", default=50)
    parser.add_argument("--nb_iteration_per_epoch", type=int, help="", default=50)
    parser.add_argument("--learning_rate", type=float, help="", default=0.05)
    parser.add_argument("--optimizer_name", type=str, help="", default="adam")
    parser.add_argument("--batch_size", type=int, help="", default=None)
    parser.add_argument("--seed", type=int, help="", default=1)
    parser.add_argument("--gpu_number", type=int, help="", default=0)
    parser.add_argument(
        "--no_ext_signal",
        action="store_true",
        help="add this argument to train a next model without the ext signal",
    )

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
    main_signal_dataset_path = args.main_signal_dataset_path
    ext_signal_dataset_path = args.ext_signal_dataset_path
    nb_hidden_state = args.nb_hidden_state
    nb_max_epoch = args.nb_max_epoch
    nb_iteration_per_epoch = args.nb_iteration_per_epoch
    learning_rate = args.learning_rate
    optimizer_name = args.optimizer_name
    batch_size = args.batch_size
    no_ext_signal = args.no_ext_signal

    if main_folder[-1] != "/":
        main_folder += "/"
    if not os.path.exists(main_folder):
        os.makedirs(main_folder)

    config_dict = vars(args)
    write_json(config_dict, os.path.join(main_folder, "config.json"))

    all_y_data = pd.read_csv(main_signal_dataset_path, index_col=0)
    all_w_data = pd.read_csv(ext_signal_dataset_path, index_col=0)

    y_train = all_y_data.iloc[:-52]
    w_train = all_w_data.iloc[:-52].rolling(8, min_periods=0, axis=0).mean()

    if no_ext_signal:
        model = next_model_no_ext_signal(
            nb_hidden_states=nb_hidden_state,
            past_dependency=104,
            season=52,
            horizon=52,
        )
    else:
        model = next_model_with_ext_signal(
            nb_hidden_states=nb_hidden_state,
            past_dependency=104,
            season=52,
            horizon=52,
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
    )

    print("Compute accuracy")
    model_prediction = model.predict(
        y_signal=y_train, w_signal=w_train, nb_simulation=1,
    )
    y_test = all_y_data.iloc[-52:]
    model_accuracy = compute_accuracy_metric(
        y_train, y_test, model_prediction["y_pred_mean"]
    )
    write_json(model_accuracy, os.path.join(main_folder, "final_accuracy.json"))
    print(model_accuracy)
