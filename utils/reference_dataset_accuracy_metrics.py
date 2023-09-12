import numpy as np
import pandas as pd
from tqdm import tqdm


def compute_reference_dataset_accuracy_metrics(
    dataset, model, horizon, quick_version=False
):

    result = {}
    result["eval"] = {}
    result["test"] = {}
    for set_name in result:
        result[set_name]["mae"] = []
        result[set_name]["mse"] = []
    nb_test_window = int(dataset.shape[0] * 0.2) - horizon + 1
    nb_eval_window = int(dataset.shape[0] * 0.1) - horizon + 1

    if quick_version:
        time_step = 10
    else:
        time_step = 1
    for i in tqdm(range(0, nb_eval_window, time_step)):
        y_signal = dataset[: -horizon - i - nb_test_window]
        w_signal = pd.DataFrame(np.zeros_like(y_signal))
        model_prediction = model.predict(
            y_signal=y_signal, w_signal=w_signal, nb_simulation=100,
        )
        model_prediction = model_prediction["y_pred_mean"].T
        result["eval"]["mae"].append(
            np.mean(
                np.abs(
                    dataset[-horizon - i - nb_test_window : -i - nb_test_window]
                    - model_prediction
                )
            )
        )
        result["eval"]["mse"].append(
            np.mean(
                np.square(
                    dataset[-horizon - i - nb_test_window : -i - nb_test_window]
                    - model_prediction
                )
            )
        )

    for i in tqdm(range(0, nb_test_window, time_step)):
        y_signal = dataset[: -horizon - i]
        w_signal = pd.DataFrame(np.zeros_like(y_signal))
        model_prediction = model.predict(
            y_signal=y_signal, w_signal=w_signal, nb_simulation=100,
        )
        model_prediction = model_prediction["y_pred_mean"].T

        if i:
            result["test"]["mae"].append(
                np.mean(np.abs(dataset[-horizon - i : -i] - model_prediction))
            )
            result["test"]["mse"].append(
                np.mean(np.square(dataset[-horizon - i : -i] - model_prediction))
            )
        else:
            result["test"]["mae"].append(
                np.mean(np.abs(dataset[-horizon:] - model_prediction))
            )
            result["test"]["mse"].append(
                np.mean(np.square(dataset[-horizon:] - model_prediction))
            )

    for set_name in result:
        for metric in result[set_name]:
            result[set_name][metric] = np.round(np.mean(result[set_name][metric]), 3)

    return result
