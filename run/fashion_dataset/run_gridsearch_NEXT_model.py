import argparse
import os
import sys
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import tensorflow as tf
from utils.utils import write_json
from model.NEXT_model import next_model_no_ext_signal, next_model_no_ext_signal_FC


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
    parser.add_argument("--nb_max_epoch", type=int, help="", default=50)
    parser.add_argument("--nb_iteration_per_epoch", type=int, help="", default=50)
    parser.add_argument("--learning_rate_list", nargs="+", type=float, help="", default=0.05)
    parser.add_argument("--optimizer_name", type=str, help="", default="adam")
    parser.add_argument("--batch_size_list", nargs="+", type=int, help="", default=None)
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
    main_signal_dataset_path = args.main_signal_dataset_path
    ext_signal_dataset_path = args.ext_signal_dataset_path
    nb_max_epoch = args.nb_max_epoch
    nb_iteration_per_epoch = args.nb_iteration_per_epoch
    learning_rate_list = args.learning_rate_list
    optimizer_name = args.optimizer_name
    batch_size_list = args.batch_size_list

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
    
    best_model_eval = np.inf
    best_training_config = {}
    for learning_rate in learning_rate_list:
        for batch_size in batch_size_list:
            model_folder = os.path.join(main_folder, f'lr_{learning_rate}_batch_size_{batch_size}')
            if not os.path.exists(model_folder):
                os.makedirs(model_folder)
            model = next_model_no_ext_signal(
                nb_hidden_states=2, past_dependency=104, season=52, horizon=52,
            )

            print("Start Training")
            model_eval = model.fit(
                y_signal=y_train,
                w_signal=w_train,
                nb_max_epoch=nb_max_epoch,
                nb_iteration_per_epoch=nb_iteration_per_epoch,
                optimizer_name=optimizer_name,
                learning_rate=learning_rate,
                batch_size=batch_size,
                model_folder=model_folder,
            )
            write_json(model_eval.numpy(), os.path.join(model_folder, "elbo_eval.json"))
            if model_eval < best_model_eval:
                best_model_eval = model_eval
                best_training_config = {'learning_rate':learning_rate, 'batch_size':batch_size}
    
    write_json(best_training_config, os.path.join(main_folder, "best_config.json"))