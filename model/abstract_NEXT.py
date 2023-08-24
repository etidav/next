import numpy as np
import pandas as pd
import tensorflow as tf
import os
from abc import ABC, abstractmethod
from typing import List, Tuple, Type, Union
from tqdm import tqdm
from utils.utils import write_json, write_pickle


class NEXT(ABC, tf.keras.Model):
    """
    Abstract class for NEXT models.
    """

    def __init__(
        self,
        nb_hidden_states: int = 2,
        past_dependency: int = 1,
        season: int = 1,
        horizon: int = 52,
    ) -> None:
        """
        Instantiate a next model with gaussian emission laws and discrete hidden states.
        
        Arguments:
    
        - *nb_hidden_states*: number of hidden states of the Next model.
        
        - *past_dependency*: define the past dependency length.
        
        - *season*: define the seasonality length.
        
        - *horizon*: define the forecast horizon of the Next model.
        """
        super(NEXT, self).__init__()
        self.K = nb_hidden_states
        self.past_dependency = past_dependency
        self.season = season
        self.horizon = horizon

    @abstractmethod
    def define_param(self) -> None:
        """
        Define parameters of the Next model. Have to be define : 
            - self.hidden_state_prior_model_params: parameters linked to the transition model
            - self.emission_laws_params: parameters linked to the emission model
            - self.hidden_state_posterior_model_params: parameters linked to the variational model
        """
        pass

    @abstractmethod
    def get_param(self) -> None:
        """
        Return model's parameters.
        """
        pass

    @abstractmethod
    def assign_param(self) -> None:
        """
        Instantiate a model with existing parameters.
        """
        pass

    @abstractmethod
    def compute_emission_laws_parameters(
        self, y_past: tf.Tensor, w_past: tf.Tensor, time_index: tf.Tensor,
    ) -> tf.Tensor:
        """
        Compute the mean and std of the emission densities.
        
        Arguments:
        
        - *y_past*: a tf.Tensor(nb_time_series, past_dependency) containing the past values of the main signal.
        
        - *w_past*: a tf.Tensor(nb_time_series, past_dependency) containing the values of the external signal.
        
        - *time_index*: a tf.Tensor(nb_time_series, past_dependency) corresponding to an encoding of the time period of the input window.
  
        Returns:
        
        - *mu*: a tf.Tensor(nb_time_series,horizon) containing the mean of the emission densities.
        
        - *sigma*: a tf.Tensor(nb_time_series,horizon) containing the std of the emission densities.
        """
        pass

    @abstractmethod
    def compute_prior_probabilities(
        self, y_past: tf.Tensor, w_past: tf.Tensor, time_index: tf.Tensor
    ) -> tf.Tensor:
        """
        Compute the hidden states probabilities
        
         Arguments:
        
        - *y_past*: tf.Tensor(nb_time_series, past_dependency) containing the past values of the main signal.
        
        - *w_past*: tf.Tensor(nb_time_series, past_dependency) containing the values of the external signal.
        
        - *time_index*: tf.Tensor(nb_time_series, past_dependency) corresponding to an encoding of the time period of the input window.
         
        Returns:
        
        - *prior_probabilities*: a tf.Tensor(nb_time_series,horizon,K) containing the prior probabilities to be in a hidden state at each time step.
        """
        pass

    @abstractmethod
    def compute_posterior_probabilities(
        self, y: tf.Tensor, y_past: tf.Tensor, w_past: tf.Tensor, time_index: tf.Tensor
    ) -> tf.Tensor:
        """
        Compute the hidden states posterior probabilities
        
         Arguments:
        
        - *y*: a tf.Tensor(nb_time_series, horizon) containing the current and future values of the main signal.
        
        - *time_index*: a tf.Tensor(nb_time_series, past_dependency) corresponding to an encoding of the time period of the input window.
        
        - *w_past*: a tf.Tensor(nb_time_series, past_dependency) containing the values of the external signal.
            
        - *y_past*: a tf.Tensor(nb_time_series, past_dependency) containing the past values of the main signal.
         
        Returns:
        
        - *prosterior_probabilities*: a tf.Tensor(nb_time_series,horizon,K) containing the posterior probabilities to be in a hidden state at each time step.
        """

    def compute_emission_laws_trajectories(
        self,
        y_past: tf.Tensor,
        w_past: tf.Tensor,
        time_index: tf.Tensor,
        nb_trajectories: int = 1,
    ) -> tf.Tensor:
        """
        Compute parameters or a realisation of the emission densities.
        
        Arguments:
        
        - *y_past*: a tf.Tensor(nb_time_series, past_dependency) containing the past values of the main signal.
        
        - *w_past*: a tf.Tensor(nb_time_series, past_dependency) containing the values of the external signal.
        
        - *time_index*: a tf.Tensor(nb_time_series, past_dependency) corresponding to an encoding of the time period of the input window.
        
        - *nb_trajectories*:  number of trajectories to simulate.
         
        Returns:
        
        - *trajectories*: a tf.Tensor(nb_time_series,horizon,K,nb_trajectories) containing the trajectories of the emission densities.
        
        """
        mu, sigma = self.compute_emission_laws_parameters(
            y_past,w_past, time_index,
        )
        emission_laws_trajectories = tf.stack(
            [
                tf.random.normal(
                    shape=mu.shape, mean=mu, stddev=sigma, dtype=tf.float64
                )
                for i in range(nb_trajectories)
            ],
            axis=3,
        )

        return emission_laws_trajectories

    def compute_emission_laws_densities(
        self,
        y: tf.Tensor,
        y_past: tf.Tensor,
        w_past: tf.Tensor,
        time_index: tf.Tensor,
        apply_log: bool = False,
    ) -> tf.Tensor:
        """
        Compute the probability density of y regarding the emission densities.
        
        Arguments:
        
        - *y*: a tf.Tensor(nb_time_series, horizon) containing the current and future values of the main signal.
        
        - *y_past*: a tf.Tensor(nb_time_series, past_dependency) containing the past values of the main signal.
        
        - *w_past*: a tf.Tensor(nb_time_series, past_dependency) containing the values of the external signal.
       
        - *time_index*: a tf.Tensor(nb_time_series, past_dependency) corresponding to an encoding of the time period of the input window.
        
        - *apply_log*: boolean indicating if the probability or the log probability has to be return.
        
        Returns:
        
        - *probability_value*: a tf.Tensor(nb_time_series, horizon,K) containing the probability that a value has been realised by one of the emission densities.
        """
        mu, sigma = self.compute_emission_laws_parameters(y_past, w_past, time_index)

        if not apply_log:
            return (1 / tf.math.sqrt(2 * np.pi * tf.math.pow(sigma, 2))) * tf.math.exp(
                -((tf.expand_dims(y, axis=2) - mu) ** 2) / (2 * tf.math.pow(sigma, 2))
            )
        else:
            return -(1 / 2) * (
                tf.math.log(2 * np.pi * sigma ** 2)
                + ((tf.expand_dims(y, axis=2) - mu) ** 2) / tf.math.pow(sigma, 2)
            )

    @tf.function
    def compute_elbo(
        self,
        y: tf.Tensor,
        y_past: tf.Tensor,
        w_past: tf.Tensor,
        time_index: tf.Tensor,
        use_uniform_prior: bool,
        posterior_probabilities: tf.Tensor = None,
    ) -> tf.Tensor:
        """
        Compute the ELBO
        
        Arguments:
        
        - *y*: a tf.Tensor(nb_time_series, horizon) containing the current and future values of the main signal.
        
        - *y_past*: a tf.Tensor(nb_time_series, past_dependency) containing the past values of the main signal.
        
        - *w_past*: a tf.Tensor(nb_time_series, past_dependency) containing the values of the external signal.
        
        - *time_index*: a tf.Tensor(nb_time_series, past_dependency) corresponding to an encoding of the time period of the input window.
        
        - *use_uniform_prior*: if True, will use a uniform prior in place of the variational model for the emission laws model training.
        
        - *posterior_probabilities*: a tf.Tensor(nb_time_series, horizon, K) already containing the prosterior propobabilites to speed up the training.
         
        Returns:
        
        - *elbo*: a tf.Tensor(1) containing the ELBO loss
        """
        batch_size = tf.shape(y)[0]
        T = tf.shape(y)[1]
        if w_past is None:
            w_past = tf.zeros_like(y)
        if y_past is None:
            y_past = tf.zeros_like(y)

        if use_uniform_prior:
            posterior_probabilities = self.compute_posterior_probabilities(
                y, y_past, w_past, time_index
            )
            emission_laws_densities = self.compute_emission_laws_densities(
                y, y_past, w_past, time_index, apply_log=True
            )
            elbo_emission_laws = tf.math.reduce_sum(
                tf.math.multiply(posterior_probabilities, emission_laws_densities),
                axis=[1, 2],
            )
            elbo_entropy = tf.math.reduce_sum(
                tf.math.multiply(
                    posterior_probabilities, tf.math.log(posterior_probabilities)
                ),
                axis=[1, 2],
            )
            elbo = -tf.math.reduce_mean(elbo_emission_laws) + tf.math.reduce_mean(
                elbo_entropy
            )
        else:
            posterior_probabilities = self.compute_posterior_probabilities(
                y, y_past, w_past, time_index
            )
            prior_probabilities = self.compute_prior_probabilities(
                y_past, w_past, time_index
            )
            emission_laws_densities = self.compute_emission_laws_densities(
                y, y_past, w_past, time_index, apply_log=True
            )
            elbo_emission_laws = tf.math.reduce_sum(
                tf.math.multiply(posterior_probabilities, emission_laws_densities),
                axis=[1, 2],
            )
            elbo_entropy = tf.math.reduce_sum(
                tf.math.multiply(
                    posterior_probabilities, tf.math.log(posterior_probabilities)
                ),
                axis=[1, 2],
            )
            elbo_hidden_states = tf.math.reduce_sum(
                tf.math.multiply(
                    posterior_probabilities, tf.math.log(prior_probabilities)
                ),
                axis=[1, 2],
            )
            elbo = (
                -tf.math.reduce_mean(elbo_emission_laws)
                - tf.math.reduce_mean(elbo_hidden_states)
                + tf.math.reduce_mean(elbo_entropy)
            )
        return elbo

    @tf.function
    def compute_grad(
        self,
        y: tf.Tensor,
        y_past: tf.Tensor,
        w_past: tf.Tensor,
        time_index: tf.Tensor,
        use_uniform_prior: bool,
        posterior_probabilities: tf.Tensor = None,
    ) -> Tuple[tf.Tensor, List]:
        """
        Compute the tf.gradient link the the ELBO loss.
        
        Arguments:
        
        - *y*: a tf.Tensor(nb_time_series, horizon) containing the current and future values of the main signal.
        
        - *y_past*: a tf.Tensor(nb_time_series, past_dependency) containing the past values of the main signal.
        
        - *w_past*: a tf.Tensor(nb_time_series, past_dependency) containing the values of the external signal.
        
        - *time_index*: a tf.Tensor(nb_time_series, past_dependency) corresponding to an encoding of the time period of the input window.
        
        - *use_uniform_prior*: if True, will use a uniform prior in place of the variational model for the emission laws model training.
        
        - *posterior_probabilities*: a tf.Tensor(nb_time_series, horizon, K) already containing the prosterior propobabilites to speed up the training.
        
        Returns:
        
        - *grad*: a  containing the gradient of the models parameters
        """

        with tf.GradientTape() as tape:
            EM_loss = self.compute_elbo(
                y,
                y_past,
                w_past,
                time_index,
                use_uniform_prior,
                posterior_probabilities,
            )
            if use_uniform_prior:
                grad = tape.gradient(
                    EM_loss,
                    self.emission_laws_params
                    + self.hidden_state_posterior_model_params,
                )
            else:
                grad = tape.gradient(
                    EM_loss,
                    self.hidden_state_prior_model_params
                    + self.hidden_state_posterior_model_params,
                )

        return grad

    def fit(
        self,
        y_signal: pd.DataFrame,
        w_signal: pd.DataFrame,
        nb_max_epoch: int = 10,
        nb_iteration_per_epoch: int = 1,
        optimizer_name: str = "adam",
        learning_rate: float = 0.1,
        batch_size: int = None,
        model_folder: str = None,
        preprocess_input: bool = True,
    ) -> None:
        """
        Fit the proposed model.

        Arguments:

        - *y_signal*: a pd.DataFrame(nb_time_step, nb_time_series) containing the main signal.
        
        - *w_signal*: a pd.DataFrame(nb_time_step, nb_time_series) containing the external signal.
        
        - *nb_max_epoch*: how many max epoch will be run per training step.
        
        - *nb_iteration_per_epoch*: how many gradient step will be computed per epoch.
        
        - *optimizer_name*: what tf.optimizer will be used to fit the model.
        
        - *learning_rate*: what learning rate will be used to instantiate the optimizer.
        
        - *batch_size*: a int indicating how many sequences will be randomly sampled

        - *model_folder*: a folder where the parameters of the model will be saved.
        """

        y_signal = tf.constant(y_signal.values.T)
        w_signal = tf.constant(w_signal.values.T)

        original_param = self.get_param()
        start_t = [
            y_signal[0, : -self.horizon].shape[0] for i in range(y_signal.shape[0])
        ]
        for alternate_training_index in range(2):
            print(f"training phase {alternate_training_index+1}")
            stop_training = False
            wrong_initialisation = 0
            overfitting_count = 0
            early_stopping = 200
            alternate_training_learning_rate = learning_rate
            if alternate_training_index == 0:
                use_uniform_prior = True
            else:
                use_uniform_prior = False
            eval_metric = self.compute_eval_metric(
                y_signal, w_signal, start_t, use_uniform_prior=use_uniform_prior,preprocess_input=preprocess_input,
            )
            print("eval metric", eval_metric)
            alternate_training_param = self.get_param()
            exec_param = []
            exec_param.append(self.get_param())
            for epoch in range(nb_max_epoch):
                (y, y_past, w_past, time_index,) = self.sample_and_normalize_dataset(
                    y_signal, w_signal, batch_size, preprocess_input
                )
                optimizer = self.build_optimizer(
                    optimizer_name=optimizer_name,
                    learning_rate=alternate_training_learning_rate,
                )
                if alternate_training_index == 1:
                    posterior_probabilities = self.compute_posterior_probabilities(
                        y, y_past, w_past, time_index
                    )
                else:
                    posterior_probabilities = None
                _elbo = self.compute_elbo(
                    y,
                    y_past,
                    w_past,
                    time_index,
                    use_uniform_prior,
                    posterior_probabilities,
                )
                elbo_diff = []
                updated_param = []
                for j in tf.range(nb_iteration_per_epoch):
                    grad = self.compute_grad(
                        y,
                        y_past,
                        w_past,
                        time_index,
                        use_uniform_prior,
                        posterior_probabilities,
                    )
                    if alternate_training_index == 0:
                        optimizer.apply_gradients(
                            zip(
                                grad,
                                self.emission_laws_params
                                + self.hidden_state_posterior_model_params,
                            )
                        )
                    elif alternate_training_index == 1:
                        optimizer.apply_gradients(
                            zip(
                                grad,
                                self.hidden_state_prior_model_params
                                + self.hidden_state_posterior_model_params,
                            )
                        )
                    updated_param.append(self.get_param())
                    elbo_iter = self.compute_elbo(
                        y,
                        y_past,
                        w_past,
                        time_index,
                        use_uniform_prior,
                        posterior_probabilities,
                    )
                    elbo_diff.append(elbo_iter - _elbo)
                try:
                    best_iter = np.nanargmin(elbo_diff)
                except ValueError:
                    best_iter = 0
                elbo_diff_best = elbo_diff[best_iter]
                updated_param = updated_param[best_iter]
                print("epoch : ", epoch)
                print("elbo diff : ", elbo_diff_best)
                if elbo_diff_best >= 0:
                    print(
                        f"elbo diff > 0 --> optimizer learning rate reduced to {optimizer.lr/2}"
                    )
                    self.assign_param(*exec_param[-1])
                    if optimizer.lr < 10 ** (-10):
                        break
                    alternate_training_learning_rate = (
                        alternate_training_learning_rate / 2
                    )
                    overfitting_count += 1
                elif elbo_diff_best < 0:
                    self.assign_param(*updated_param)
                    epoch_eval_metric = self.compute_eval_metric(
                        y_signal,
                        w_signal,
                        start_t,
                        use_uniform_prior=use_uniform_prior,
                        preprocess_input=preprocess_input
                    )

                    if epoch_eval_metric < eval_metric:
                        print(f"Eval elbo: {epoch_eval_metric}. Save param.")
                        eval_metric = epoch_eval_metric
                        alternate_training_param = self.get_param()
                        overfitting_count = 0
                    else:
                        print(f"Eval elbo: {epoch_eval_metric}.")
                        overfitting_count += 1
                else:
                    wrong_initialisation = 1
                    print("wrong initialisation")
                    stop_training = True
                    break

                if overfitting_count > 0 and overfitting_count % 15 == 0:
                    print(
                        f"15 epochs without improvment --> optimizer learning rate reduced to {optimizer.lr/2}"
                    )
                    alternate_training_learning_rate = (
                        alternate_training_learning_rate / 2
                    )

                exec_param.append(self.get_param())

                if np.abs(elbo_diff_best) < 10 ** (-6):
                    print("elbo diff < 10 ** (-6) --> stop of the training algorithm")
                    stop_training = True
                    break
                if overfitting_count == early_stopping:
                    print(
                        "no improvment on the eval size epoch --> stop of the training algorithm"
                    )
                    stop_training = True
                    break

                if stop_training == True:
                    break

            if not wrong_initialisation:
                self.assign_param(*alternate_training_param)
                eval_metric = self.compute_eval_metric(
                    y_signal, w_signal, start_t, use_uniform_prior=use_uniform_prior,preprocess_input=preprocess_input
                )
                print(f"alternate_training elbo : {eval_metric}")
                if model_folder is not None:
                    exec_folder = os.path.join(
                        model_folder,
                        f"alternate_training_index{alternate_training_index}",
                    )
                    if not os.path.exists(exec_folder):
                        os.makedirs(exec_folder)
                    write_pickle(
                        self.get_param(),
                        os.path.join(
                            exec_folder,
                            f"loop{alternate_training_index}_final_param.pkl",
                        ),
                    )

        if not wrong_initialisation:
            if model_folder is not None:
                write_pickle(
                    self.get_param(), os.path.join(model_folder, "final_param.pkl")
                )

    def sample_and_normalize_dataset(self, y_signal, w_signal, batch_size, preprocess_input=True):
        """
        draw a sample of sequences from the dataset and normalize them
        
        Arguments:
        
        - *y_signal*: a tf.Tensor(nb_time_series, nb_time_step) containing the main signal.
        
        - *w_signal*: a tf.Tensor(nb_time_series, nb_time_step) containing the external signal.
        
        - *batch_size*: a int indicating how many sequences will be randomly sampled
         
        Returns:
        - *y_true*: a tf.Tensor(nb_time_series,horizon) containing the normalized values of the main signal that the model will learn to predict
        
        - *y_past*: a tf.Tensor(nb_time_series,past_dependency) containing the normalized past values of the main signal.
        
        - *w_past*: a tf.Tensor(nb_time_series,past_dependency) containing the normalized past values of the external signal.
        
        - *time_index_train*: a tf.Tensor(nb_time_series, past_dependency) corresponding to an encoding of the time period of the input window.
        """

        nb_sample_per_ts = y_signal.shape[1] - self.past_dependency - self.horizon + 1

        if batch_size <= y_signal.shape[0]:
            ts_index = tf.random.shuffle(tf.range(y_signal.shape[0]))[:batch_size]
        else:
            ts_index =  tf.random.uniform(
                shape=[batch_size,], minval=0, maxval=y_signal.shape[0], dtype=tf.int32
            )
        window_index = tf.random.uniform(
            shape=[batch_size,], minval=0, maxval=nb_sample_per_ts, dtype=tf.int32
        )
        sampled_y_signal = tf.concat(
            [
                y_signal[i : i + 1, j : j + self.past_dependency + self.horizon]
                for i, j in zip(ts_index, window_index)
            ],
            axis=0,
        )
        if preprocess_input:
            y_signal_mean = tf.math.reduce_mean(
                sampled_y_signal[:, : self.past_dependency], axis=1, keepdims=True
            )
            y_signal_std = tf.math.reduce_std(
                sampled_y_signal[:, : self.past_dependency], axis=1, keepdims=True
            )
            sampled_y_signal = tf.math.divide_no_nan(
                sampled_y_signal - y_signal_mean, y_signal_std
            )
        y_past = sampled_y_signal[:, : self.past_dependency]
        y_true = sampled_y_signal[:, self.past_dependency :]

        if w_signal is not None:
            sampled_w_signal = tf.concat(
                [
                    w_signal[i : i + 1, j : j + self.past_dependency]
                    for i, j in zip(ts_index, window_index)
                ],
                axis=0,
            )
            if preprocess_input:
                w_past = tf.math.divide_no_nan(
                    sampled_w_signal - y_signal_mean, y_signal_std
                )
            else:
                w_past = sampled_w_signal
        else:
            w_past = tf.zeros_like(y_past)

        time_signal = tf.concat(
            [
                tf.reshape(tf.range(y_signal.shape[1], dtype=tf.float64), (1, -1))
                for i in range(y_signal.shape[0])
            ],
            axis=0,
        )
        sampled_time_signal = tf.concat(
            [
                time_signal[i : i + 1, j : j + self.past_dependency,]
                for i, j in zip(ts_index, window_index)
            ],
            axis=0,
        )
        time_index = tf.math.divide(
            tf.math.floormod(sampled_time_signal, self.season), self.season
        )

        return y_true, y_past, w_past, time_index

    def build_optimizer(
        self, optimizer_name: str, learning_rate: int
    ) -> tf.keras.optimizers.Optimizer:
        """
        Instantiate and return a tf.keras.optimizer

        Arguments:

        - *optimizer_name*: name of the chosen optimizer.

        - *learning_rate*: learning rate of the optimizer.
        
        Returns:

        - *optimizer*: a tensorflow optimizer.
        """
        if optimizer_name.lower() == "adam":
            return tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name.lower() == "sgd":
            return tf.keras.optimizers.SGD(learning_rate=learning_rate)

    def compute_eval_metric(
        self, y_signal, w_signal, start_t, use_uniform_prior=False, preprocess_input=True
    ):
        """
        compute the ELBO on the eval step
        
        Arguments:
        
        - *y_signal*: a tf.Tensor(nb_time_series, nb_time_step) containing the main signal.
        
         - *w_signal*: a tf.Tensor(nb_time_series, nb_time_step) containing the external signal.
        
        - *start_t*: a list of integers representing the time step of the last input value of each sequences.
        
        - *use_uniform_prior*: if True, will use a uniform prior in place of the variational model for the emission laws model training.
         
        Returns:
        
        - *eval_loss*: a tf.Tensor(1) containing the ELBO loss
        """
        time_index = tf.concat(
            [
                tf.reshape(
                    tf.range(i - self.past_dependency, i, dtype=tf.float64), (1, -1)
                )
                for i in start_t
            ],
            axis=0,
        )
        time_index = tf.math.divide(
            tf.math.floormod(time_index, self.season), self.season
        )
        sampled_y_signal = y_signal[:, -self.horizon - self.past_dependency :]
        sampled_w_signal = w_signal[:, -self.horizon - self.past_dependency :]
        if preprocess_input:
            y_signal_mean = tf.math.reduce_mean(
                sampled_y_signal[:, : self.past_dependency], axis=1, keepdims=True
            )
            y_signal_std = tf.math.reduce_std(
                sampled_y_signal[:, : self.past_dependency], axis=1, keepdims=True
            )
            sampled_y_signal = tf.math.divide_no_nan(
                sampled_y_signal - y_signal_mean, y_signal_std
            )
            sampled_w_signal = tf.math.divide_no_nan(
                sampled_w_signal - y_signal_mean, y_signal_std
            )
        eval_loss = self.compute_elbo(
            sampled_y_signal[:, self.past_dependency :],
            sampled_y_signal[:, : self.past_dependency],
            sampled_w_signal[:, : self.past_dependency],
            time_index,
            use_uniform_prior,
            None,
        )
        return eval_loss

    def predict(
        self,
        y_signal: pd.DataFrame = None,
        w_signal: pd.DataFrame = None,
        nb_simulation: int = 100,
        preprocess_input: bool = True,
    ):
        """
        simulate a (X_t,Y_t) process of length T.
        
        Arguments:
        
        - *y_signal*: a pd.DataFrame(nb_time_step, nb_time_series) containing the main signal.
        
        - *w_signal*: a pd.DataFrame(nb_time_step, nb_time_series) containing the external signal.
         
        - *nb_simulation*: int indicating how many simulation has to be computed.
         
        Returns:
        
        - *all_final_prediction*: a tf.Tensor(nb_time_series, nb_simulation,horizon) containing all the final predictions of the model.
        
        - *all_hs_prediction*: a tf.Tensor(nb_time_series, nb_simulation,horizon,K) containing all the final predictions of the hidden states.
        
        - *all_final_prior_proba*: a tf.Tensor(nb_time_series, nb_simulation,horizon,K) containing all the final predictions of the hidden states probabilities.
        """

        y_past = tf.constant(y_signal.values.T)
        w_past = tf.constant(w_signal.values.T)
        start_t = [y_past.shape[1] for i in range(y_past.shape[0])]

        batch_size = y_past.shape[0]
        y_past = y_past[:, -self.past_dependency :]
        if w_past is not None:
            w_past = w_past[:, -self.past_dependency :]
        else:
            w_past = tf.zeros_like(y_past)
        if preprocess_input:
            y_past_mean = tf.math.reduce_mean(y_past, axis=1, keepdims=True)
            y_past_std = tf.math.reduce_std(y_past, axis=1, keepdims=True)
            y_past = tf.math.divide_no_nan(y_past - y_past_mean, y_past_std)
            w_past = tf.math.divide_no_nan(w_past - y_past_mean, y_past_std)

        time_index = tf.concat(
            [
                tf.reshape(
                    tf.range(i - self.past_dependency, i, dtype=tf.float64), (1, -1)
                )
                for i in start_t
            ],
            axis=0,
        )
        time_index = tf.math.divide(
            tf.math.floormod(time_index, self.season), self.season
        )

        emission_laws_mu, emission_laws_sigma = self.compute_emission_laws_parameters(
            y_past=y_past, w_past=w_past, time_index=time_index
        )
        if preprocess_input:
            emission_laws_mu = emission_laws_mu * tf.expand_dims(
                y_past_std, axis=2
            ) + tf.expand_dims(y_past_mean, axis=2)
        prior_probabilities = self.compute_prior_probabilities(
            w_past, y_past, time_index
        )
        final_prediction_renorm = tf.math.reduce_sum(prior_probabilities*emission_laws_mu, axis=2)
        
        hidden_state_simulation = tf.stack(
            [
                tf.random.categorical(
                    logits=tf.math.log(prior_probabilities[i, :, :]),
                    num_samples=nb_simulation,
                )
                for i in range(prior_probabilities.shape[0])
            ],
            axis=0,
        )
        hidden_state_simulation_one_hot = tf.one_hot(
            hidden_state_simulation, self.K, axis=2
        )
        hidden_state_simulation_one_hot = tf.cast(
            hidden_state_simulation_one_hot, tf.float64
        )
        hidden_state_simulation = tf.cast(hidden_state_simulation, tf.float64)

        emission_laws_simulations = self.compute_emission_laws_trajectories(
            y_past=y_past,
            w_past=w_past,
            time_index=time_index,
            nb_trajectories=nb_simulation,
        )
        if preprocess_input:
            y_past_std = y_past_std[:, :, tf.newaxis, tf.newaxis]
            y_past_mean = y_past_mean[:, :, tf.newaxis, tf.newaxis]
            emission_laws_simulations = (
                emission_laws_simulations * y_past_std + y_past_mean
            )

        final_simulation = tf.math.reduce_sum(
            emission_laws_simulations * hidden_state_simulation_one_hot, axis=2
        )

        result = {}
        result["y_pred_mean"] = final_prediction_renorm.numpy()
        result["y_emission_law_mean"] = emission_laws_mu.numpy()
        result["prior_probabilities"] = prior_probabilities.numpy()
        result["all_y_pred"] = final_simulation.numpy()
        result["all_y_emission_law"] = emission_laws_simulations.numpy()
        result["all_hidden_state_trajectories"] = hidden_state_simulation.numpy()

        return result
