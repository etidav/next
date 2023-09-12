import argparse
import os
import numpy as np
import pandas as pd
import json
from tqdm import tqdm


def compute_naive_accuracy_metric(dataset, horizon_window):

    dataset = dataset.values
    result = {}
    for horizon in tqdm(horizon_window):
        result[horizon] = {}
        result[horizon]["mae"] = []
        result[horizon]["mse"] = []
        nb_test_window = int(dataset.shape[0] * 0.2) - horizon + 1
        for i in range(nb_test_window):
            naive_pred = np.concatenate(
                [dataset[-horizon - 1 - i : -horizon - i] for j in range(horizon)],
                axis=0,
            )
            if i:
                result[horizon]["mae"].append(
                    np.mean(np.abs(dataset[-horizon - i : -i] - naive_pred))
                )
                result[horizon]["mse"].append(
                    np.mean(np.square(dataset[-horizon - i : -i] - naive_pred))
                )
            else:
                result[horizon]["mae"].append(
                    np.mean(np.abs(dataset[-horizon:] - naive_pred))
                )
                result[horizon]["mse"].append(
                    np.mean(np.square(dataset[-horizon:] - naive_pred))
                )

    for horizon in result:
        for metric in result[horizon]:
            result[horizon][metric] = np.round(np.mean(result[horizon][metric]), 3)

    return result


if __name__ == "__main__":

    dataset_reference = {}
    dataset_reference["ETTh1"] = "/next/data/reference_dataset/ETT-small/ETTh1.csv"
    dataset_reference["ETTh2"] = "/next/data/reference_dataset/ETT-small/ETTh2.csv"
    dataset_reference["ETTm1"] = "/next/data/reference_dataset/ETT-small/ETTm1.csv"
    dataset_reference["ETTm2"] = "/next/data/reference_dataset/ETT-small/ETTm2.csv"
    dataset_reference[
        "electricity"
    ] = "/next/data/reference_dataset/electricity/electricity.csv"
    dataset_reference[
        "exchange_rate"
    ] = "/next/data/reference_dataset/exchange_rate/exchange_rate.csv"
    dataset_reference[
        "illness"
    ] = "/next/data/reference_dataset/illness/national_illness.csv"
    dataset_reference["traffic"] = "/next/data/reference_dataset/traffic/traffic.csv"
    dataset_reference["weather"] = "/next/data/reference_dataset/weather/weather.csv"

    for dataset_name in dataset_reference:
        dataset_path = dataset_reference[dataset_name]
        dataset = pd.read_csv(dataset_path, index_col=0)
        dataset.index = pd.to_datetime(dataset.index)
        if dataset_name in ["ETTm1.csv", "ETTm2.csv", "ETTh1.csv", "ETTh2.csv"]:
            dataset = dataset.loc[:"2018-02-20"]
            train_size = int(dataset.shape[0] * 0.6)
        else:
            train_size = int(dataset.shape[0] * 0.7)
        dataset = (dataset - dataset[:train_size].mean(axis=0)) / dataset[
            :train_size
        ].std(axis=0)
        if dataset_name == "illness":
            horizon_window = [24, 36, 48, 60]
        else:
            horizon_window = [96, 192, 336, 720]
        dataset_naive_accuracy = compute_naive_accuracy_metric(dataset, horizon_window)
        print(dataset_name)
        for horizon in dataset_naive_accuracy:
            print("horizon " + str(horizon), dataset_naive_accuracy[horizon])
