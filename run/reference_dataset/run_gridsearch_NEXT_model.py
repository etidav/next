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
        "--dataset_path", type=str, help="path to a dataset", required=True
    )
    parser.add_argument("--past_dependency_list", nargs="+", type=int, help="", required=True)
    parser.add_argument("--season", type=int, help="", required=True)
    parser.add_argument("--horizon", type=int, help="", required=True)
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
    dataset_path = args.dataset_path
    past_dependency_list = args.past_dependency_list
    season = args.season
    horizon = args.horizon
    preprocess_input = False
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
    
    best_model_eval = np.inf
    best_training_config = {}
    for past_dependency in past_dependency_list:
        for learning_rate in learning_rate_list:
            for batch_size in batch_size_list:
                model_folder = os.path.join(main_folder, f'past_dependency_{past_dependency}_lr_{learning_rate}_batch_size_{batch_size}')
                if not os.path.exists(model_folder):
                    os.makedirs(model_folder)
                model = next_model_no_ext_signal_FC(
                    nb_hidden_states=2, past_dependency=past_dependency, season=season, horizon=horizon,
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
                    preprocess_input=preprocess_input
                )
                if model_eval < best_model_eval:
                    best_model_eval = model_eval
                    best_training_config = {'past_dependency':past_dependency, 'learning_rate':learning_rate, 'batch_size':batch_size}
    
    write_json(best_training_config, os.path.join(main_folder, "best_config.json"))