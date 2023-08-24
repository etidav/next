import numpy as np
import tensorflow as tf
import sys
from typing import List, Tuple, Type, Union

from model.abstract_NEXT import NEXT


class next_model_no_ext_signal(NEXT):
    """
    Class defining a next model with gaussian emission laws and discrete hidden states.
    """

    def __init__(
        self,
        nb_hidden_states: int = 2,
        past_dependency: int = 1,
        season: int = 1,
        horizon: int = 52,
    ) -> None:
        """
        Instantiate a ARHMMES with gaussian emission laws and discrete hidden states.
        
        Arguments:
    
        - *nb_hidden_states*: number of hidden states of the HMM.
        
        - *past_dependency*: define the past dependency length.
        
        - *season*: define the seasonality length.
        
        - *horizon*: define the forecast horizon of the HMM.
        """
        super().__init__(nb_hidden_states, past_dependency, season, horizon)
        self.define_param()

    def define_emission_model_no_ext_signal(self) -> tf.keras.Model:
        """
        Define emission law model who has only access to the main signal past
        """
        input_layer = tf.keras.Input((self.past_dependency, 2))
        lstm_layer0 = tf.keras.layers.LSTM(10, return_sequences=True)(input_layer)
        flatten_layer = tf.keras.layers.Flatten()(lstm_layer0)
        dense_layer3_mu = tf.keras.layers.Dense(self.horizon, dtype=tf.float64)(
            flatten_layer
        )
        dense_layer3_sigma = tf.keras.layers.Dense(self.horizon, dtype=tf.float64)(
            flatten_layer
        )
        activation_layer3 = (
            tf.keras.layers.Activation("relu", dtype=tf.float64)(dense_layer3_sigma)
            + 0.1
        )

        model = tf.keras.Model(
            inputs=input_layer, outputs=[dense_layer3_mu, activation_layer3]
        )

        return model

    def define_hidden_state_prior_model(self) -> tf.keras.Model:
        """
        Define hidden states prior model
        """
        input_layer0 = tf.keras.Input((self.past_dependency, 2))
        lstm_layer0 = tf.keras.layers.LSTM(10, return_sequences=True)(input_layer0)
        input_layer1 = tf.keras.Input((self.horizon, self.K))
        lstm_layer1 = tf.keras.layers.LSTM(10, return_sequences=True)(input_layer1)
        concat_layer = tf.keras.layers.concatenate([lstm_layer0, lstm_layer1], axis=1)
        flatten_layer = tf.keras.layers.Flatten()(concat_layer)
        dense_layer0 = tf.keras.layers.Dense(self.horizon * self.K, dtype=tf.float64)(
            flatten_layer
        )
        reshape_layer0 = tf.keras.layers.Reshape((self.horizon, self.K))(dense_layer0)
        activation_layer = tf.keras.layers.Activation("softmax", dtype=tf.float64)(
            reshape_layer0
        )
        model = tf.keras.Model(
            inputs=[input_layer0, input_layer1], outputs=activation_layer
        )

        return model

    def define_hidden_state_posterior_model(self) -> tf.keras.Model:
        """
        Define hidden states posterior model
        """
        input_layer0 = tf.keras.Input((self.past_dependency, 2))
        lstm_layer0 = tf.keras.layers.LSTM(10, return_sequences=True)(input_layer0)
        input_layer1 = tf.keras.Input((self.horizon, 1))
        lstm_layer1 = tf.keras.layers.LSTM(10, return_sequences=True)(input_layer1)
        concat_layer = tf.keras.layers.concatenate([lstm_layer0, lstm_layer1], axis=1)
        flatten_layer = tf.keras.layers.Flatten()(concat_layer)
        dense_layer0 = tf.keras.layers.Dense(self.horizon * self.K, dtype=tf.float64)(
            flatten_layer
        )
        reshape_layer0 = tf.keras.layers.Reshape((self.horizon, self.K))(dense_layer0)
        activation_layer = tf.keras.layers.Activation("softmax", dtype=tf.float64)(
            reshape_layer0
        )
        model = tf.keras.Model(
            inputs=[input_layer0, input_layer1], outputs=activation_layer
        )

        return model

    def define_param(self) -> None:
        """
        Define parameters of the HMM model.
        """
        self.emission_models = [
            self.define_emission_model_no_ext_signal() for i in range(self.K)
        ]
        self.hidden_state_prior_model = self.define_hidden_state_prior_model()
        self.hidden_state_posterior_model = self.define_hidden_state_posterior_model()

        self.hidden_state_prior_model_params = (
            self.hidden_state_prior_model.trainable_variables
        )
        self.emission_laws_params = self.emission_models.trainable_variables
        self.hidden_state_posterior_model_params = (
            self.hidden_state_posterior_model.trainable_variables
        )

    def get_param(self) -> Tuple:
        """
        Return model's parameters.
        
        Returns:
        
        - *param*: a Tuple containing the model parameters
        """
        return (
            [model.get_weights() for model in self.emission_models],
            self.hidden_state_prior_model.get_weights(),
            self.hidden_state_posterior_model.get_weights(),
        )

    def assign_param(
        self,
        emission_models_weights: List,
        hidden_state_prior_model_weights: tf.Tensor,
        hidden_state_posterior_model_weights: tf.Tensor,
    ) -> None:
        """
        Instantiate a model with existing parameters.
        
        Arguments:
        
        - *emission_models_weights*: a list containing the emission laws models' parameters
        
        - *hidden_state_prior_model_weights*: a tf.Tensor containing the hidden states model's parameters
        
        - *hidden_state_posterior_model_weights*: a tf.Tensor containing the variational model's parameters
        """
        for k in range(self.K):
            self.emission_models[k].set_weights(emission_models_weights[k])
        self.hidden_state_prior_model.set_weights(hidden_state_prior_model_weights)
        self.hidden_state_posterior_model.set_weights(
            hidden_state_posterior_model_weights
        )

    def compute_emission_laws_parameters(
        self, y_past: tf.Tensor, w_past: tf.Tensor, time_index: tf.Tensor
    ) -> tf.Tensor:
        """
        Compute the mean and std of the emission densities.
        
         Arguments:
         
         - *y_past*: a tf.Tensor(nb_time_series, past_dependency) containing the past values of the main signal.
         
         - *w*: a tf.Tensor(nb_time_series, past_dependency) containing the values of the external signal.
        
        - *time_index*: a tf.Tensor(nb_time_series, past_dependency) corresponding to an encoding of the time period of the input window.
         
        Returns:
        
        - *mu*: a tf.Tensor(nb_time_series,horizon) containing the mean of the emission densities.
        
        - *sigma*: a tf.Tensor(nb_time_series,horizon) containing the std of the emission densities.
        """
        past_input_no_ext_signal = tf.stack([y_past, time_index], axis=2)
        model_pred = [self.emission_models[i](past_input_no_ext_signal) for i in range(self.K)]
        mu = tf.stack([pred[0] for pred in model_pred], axis=2)
        sigma = tf.stack([pred[1] for pred in model_pred], axis=2)
        return mu, sigma

    def compute_prior_probabilities(
        self, y_past: tf.Tensor, w_past: tf.Tensor, time_index: tf.Tensor,
    ) -> tf.Tensor:
        """
        Compute the hidden states prior probabilities
        
        Arguments:
         
        - *y_past*: tf.Tensor(nb_time_series, past_dependency) containing the past values of the main signal.
        
        - *w*: tf.Tensor(nb_time_series, past_dependency) containing the values of the external signal.
        
        - *time_index*: tf.Tensor(nb_time_series, past_dependency) corresponding to an encoding of the time period of the input window.
         
        Returns:
        
        - *transition_matrix_proba*: a tf.Tensor(nb_time_series,horizon,K) containing the probabilities to be in a hidden state at each time step.
        """
        past_input = tf.stack([y_past, time_index], axis=2)
        mu, sigma = self.compute_emission_laws_parameters(y_past, w_past, time_index)
        future_input = mu
        prior_probabilities = self.hidden_state_prior_model([past_input, future_input])
        return prior_probabilities

    def compute_posterior_probabilities(
        self, y: tf.Tensor, y_past: tf.Tensor, w_past: tf.Tensor, time_index: tf.Tensor
    ) -> tf.Tensor:
        """
        Compute the hidden states posterior probabilities
        
         Arguments:
        
        - *y*: a tf.Tensor(nb_time_series, horizon) containing the current and future values of the main signal.
        
        - *y_past*: a tf.Tensor(nb_time_series, past_dependency) containing the past values of the main signal.
        
        - *w*: a tf.Tensor(nb_time_series, past_dependency) containing the values of the external signal.
        
        - *time_index*: a tf.Tensor(nb_time_series, past_dependency) corresponding to an encoding of the time period of the input window.
         
        Returns:
        
        - *transition_matrix_proba*: a tf.Tensor(nb_time_series,horizon,K) containing the probabilities to be in a hidden state at each time step.
        """
        past_input = tf.stack([y_past, time_index], axis=2)
        future_input = tf.expand_dims(y, axis=2)
        prosterior_probabilities = self.hidden_state_posterior_model(
            [past_input, future_input]
        )

        return prosterior_probabilities
    
class next_model_with_ext_signal(NEXT):
    """
    Class defining a next model with gaussian emission laws and discrete hidden states.
    """

    def __init__(
        self,
        nb_hidden_states: int = 2,
        past_dependency: int = 1,
        season: int = 1,
        horizon: int = 52,
    ) -> None:
        """
        Instantiate a ARHMMES with gaussian emission laws and discrete hidden states.
        
        Arguments:
    
        - *nb_hidden_states*: number of hidden states of the HMM.
        
        - *past_dependency*: define the past dependency length.
        
        - *season*: define the seasonality length.
        
        - *horizon*: define the forecast horizon of the HMM.
        """
        super().__init__(nb_hidden_states, past_dependency, season, horizon)
        self.define_param()

    def define_emission_model_no_ext_signal(self) -> tf.keras.Model:
        """
        Define emission law model who has only access to the main signal past
        """
        input_layer = tf.keras.Input((self.past_dependency, 2))
        lstm_layer0 = tf.keras.layers.LSTM(10, return_sequences=True)(input_layer)
        flatten_layer = tf.keras.layers.Flatten()(lstm_layer0)
        dense_layer3_mu = tf.keras.layers.Dense(self.horizon, dtype=tf.float64)(
            flatten_layer
        )
        dense_layer3_sigma = tf.keras.layers.Dense(self.horizon, dtype=tf.float64)(
            flatten_layer
        )
        activation_layer3 = (
            tf.keras.layers.Activation("relu", dtype=tf.float64)(dense_layer3_sigma)
            + 0.1
        )

        model = tf.keras.Model(
            inputs=input_layer, outputs=[dense_layer3_mu, activation_layer3]
        )

        return model

    def define_emission_model_with_ext_signal(self) -> tf.keras.Model:
        """
        Define emission law model who has only access to the main signal past
        """
        input_layer = tf.keras.Input((self.past_dependency, 3))
        lstm_layer0 = tf.keras.layers.LSTM(10, return_sequences=True)(input_layer)
        flatten_layer = tf.keras.layers.Flatten()(lstm_layer0)
        dense_layer3_mu = tf.keras.layers.Dense(self.horizon, dtype=tf.float64)(
            flatten_layer
        )
        dense_layer3_sigma = tf.keras.layers.Dense(self.horizon, dtype=tf.float64)(
            flatten_layer
        )
        activation_layer3 = (
            tf.keras.layers.Activation("relu", dtype=tf.float64)(dense_layer3_sigma)
            + 0.1
        )

        model = tf.keras.Model(
            inputs=input_layer, outputs=[dense_layer3_mu, activation_layer3]
        )

        return model

    def define_hidden_state_prior_model(self) -> tf.keras.Model:
        """
        Define hidden states prior model
        """
        input_layer0 = tf.keras.Input((self.past_dependency, 3))
        lstm_layer0 = tf.keras.layers.LSTM(10, return_sequences=True)(input_layer0)
        input_layer1 = tf.keras.Input((self.horizon, self.K))
        lstm_layer1 = tf.keras.layers.LSTM(10, return_sequences=True)(input_layer1)
        concat_layer = tf.keras.layers.concatenate([lstm_layer0, lstm_layer1], axis=1)
        flatten_layer = tf.keras.layers.Flatten()(concat_layer)
        dense_layer0 = tf.keras.layers.Dense(self.horizon * self.K, dtype=tf.float64)(
            flatten_layer
        )
        reshape_layer0 = tf.keras.layers.Reshape((self.horizon, self.K))(dense_layer0)
        activation_layer = tf.keras.layers.Activation("softmax", dtype=tf.float64)(
            reshape_layer0
        )
        model = tf.keras.Model(
            inputs=[input_layer0, input_layer1], outputs=activation_layer
        )

        return model

    def define_hidden_state_posterior_model(self) -> tf.keras.Model:
        """
        Define hidden states posterior model
        """
        input_layer0 = tf.keras.Input((self.past_dependency, 3))
        lstm_layer0 = tf.keras.layers.LSTM(10, return_sequences=True)(input_layer0)
        input_layer1 = tf.keras.Input((self.horizon, 1))
        lstm_layer1 = tf.keras.layers.LSTM(10, return_sequences=True)(input_layer1)
        concat_layer = tf.keras.layers.concatenate([lstm_layer0, lstm_layer1], axis=1)
        flatten_layer = tf.keras.layers.Flatten()(concat_layer)
        dense_layer0 = tf.keras.layers.Dense(self.horizon * self.K, dtype=tf.float64)(
            flatten_layer
        )
        reshape_layer0 = tf.keras.layers.Reshape((self.horizon, self.K))(dense_layer0)
        activation_layer = tf.keras.layers.Activation("softmax", dtype=tf.float64)(
            reshape_layer0
        )
        model = tf.keras.Model(
            inputs=[input_layer0, input_layer1], outputs=activation_layer
        )

        return model

    def define_param(self) -> None:
        """
        Define parameters of the HMM model.
        """
        self.emission_models = [
            self.define_emission_model_no_ext_signal() for i in range(int(self.K / 2))
        ] + [
            self.define_emission_model_with_ext_signal() for i in range(int(self.K / 2),self.K)
        ]
        self.hidden_state_prior_model = self.define_hidden_state_prior_model()
        self.hidden_state_posterior_model = self.define_hidden_state_posterior_model()

        self.hidden_state_prior_model_params = (
            self.hidden_state_prior_model.trainable_variables
        )
        self.emission_laws_params = self.emission_models.trainable_variables
        self.hidden_state_posterior_model_params = (
            self.hidden_state_posterior_model.trainable_variables
        )

    def get_param(self) -> Tuple:
        """
        Return model's parameters.
        
        Returns:
        
        - *param*: a Tuple containing the model parameters
        """
        return (
            [model.get_weights() for model in self.emission_models],
            self.hidden_state_prior_model.get_weights(),
            self.hidden_state_posterior_model.get_weights(),
        )

    def assign_param(
        self,
        emission_models_weights: List,
        hidden_state_prior_model_weights: tf.Tensor,
        hidden_state_posterior_model_weights: tf.Tensor,
    ) -> None:
        """
        Instantiate a model with existing parameters.
        
        Arguments:
        
        - *emission_models_weights*: a list containing the emission laws models' parameters
        
        - *hidden_state_prior_model_weights*: a tf.Tensor containing the hidden states model's parameters
        
        - *hidden_state_posterior_model_weights*: a tf.Tensor containing the variational model's parameters
        """
        for k in range(self.K):
            self.emission_models[k].set_weights(emission_models_weights[k])
        self.hidden_state_prior_model.set_weights(hidden_state_prior_model_weights)
        self.hidden_state_posterior_model.set_weights(
            hidden_state_posterior_model_weights
        )

    def compute_emission_laws_parameters(
        self, y_past: tf.Tensor, w_past: tf.Tensor, time_index: tf.Tensor
    ) -> tf.Tensor:
        """
        Compute the mean and std of the emission densities.
        
         Arguments:
         
         - *y_past*: a tf.Tensor(nb_time_series, past_dependency) containing the past values of the main signal.
         
         - *w*: a tf.Tensor(nb_time_series, past_dependency) containing the values of the external signal.
        
        - *time_index*: a tf.Tensor(nb_time_series, past_dependency) corresponding to an encoding of the time period of the input window.
         
        Returns:
        
        - *mu*: a tf.Tensor(nb_time_series,horizon) containing the mean of the emission densities.
        
        - *sigma*: a tf.Tensor(nb_time_series,horizon) containing the std of the emission densities.
        """
        past_input_no_ext_signal = tf.stack([y_past, time_index], axis=2)
        past_input_with_ext_signal = tf.stack([y_past, w_past, time_index], axis=2)
        model_pred = [
            self.emission_models[i](past_input_no_ext_signal) for i in range(int(self.K / 2))
        ] + [
            self.emission_models[i](past_input_with_ext_signal) for i in range(int(self.K / 2), self.K)
        ]
        mu = tf.stack([pred[0] for pred in model_pred], axis=2)
        sigma = tf.stack([pred[1] for pred in model_pred], axis=2)
        return mu, sigma

    def compute_prior_probabilities(
        self, y_past: tf.Tensor, w_past: tf.Tensor, time_index: tf.Tensor,
    ) -> tf.Tensor:
        """
        Compute the hidden states prior probabilities
        
        Arguments:
         
        - *y_past*: tf.Tensor(nb_time_series, past_dependency) containing the past values of the main signal.
        
        - *w*: tf.Tensor(nb_time_series, past_dependency) containing the values of the external signal.
        
        - *time_index*: tf.Tensor(nb_time_series, past_dependency) corresponding to an encoding of the time period of the input window.
         
        Returns:
        
        - *transition_matrix_proba*: a tf.Tensor(nb_time_series,horizon,K) containing the probabilities to be in a hidden state at each time step.
        """
        past_input = tf.stack([y_past, w_past, time_index], axis=2)
        mu, sigma = self.compute_emission_laws_parameters(y_past, w_past, time_index)
        future_input = mu
        prior_probabilities = self.hidden_state_prior_model([past_input, future_input])
        return prior_probabilities

    def compute_posterior_probabilities(
        self, y: tf.Tensor, y_past: tf.Tensor, w_past: tf.Tensor, time_index: tf.Tensor
    ) -> tf.Tensor:
        """
        Compute the hidden states posterior probabilities
        
         Arguments:
        
        - *y*: a tf.Tensor(nb_time_series, horizon) containing the current and future values of the main signal.
        
        - *y_past*: a tf.Tensor(nb_time_series, past_dependency) containing the past values of the main signal.
        
        - *w*: a tf.Tensor(nb_time_series, past_dependency) containing the values of the external signal.
        
        - *time_index*: a tf.Tensor(nb_time_series, past_dependency) corresponding to an encoding of the time period of the input window.
         
        Returns:
        
        - *transition_matrix_proba*: a tf.Tensor(nb_time_series,horizon,K) containing the probabilities to be in a hidden state at each time step.
        """
        past_input = tf.stack([y_past, w_past, time_index], axis=2)
        future_input = tf.expand_dims(y, axis=2)
        prosterior_probabilities = self.hidden_state_posterior_model(
            [past_input, future_input]
        )

        return prosterior_probabilities
    
class next_model_no_ext_signal_FC(NEXT):
    """
    Class defining a next model with gaussian emission laws and discrete hidden states.
    """

    def __init__(
        self,
        nb_hidden_states: int = 2,
        past_dependency: int = 1,
        season: int = 1,
        horizon: int = 52,
    ) -> None:
        """
        Instantiate a ARHMMES with gaussian emission laws and discrete hidden states.
        
        Arguments:
    
        - *nb_hidden_states*: number of hidden states of the HMM.
        
        - *past_dependency*: define the past dependency length.
        
        - *season*: define the seasonality length.
        
        - *horizon*: define the forecast horizon of the HMM.
        """
        super().__init__(nb_hidden_states, past_dependency, season, horizon)
        self.define_param()

    def define_emission_model_nows(self) -> tf.keras.Model:
        """
        Define emission law model who has only access to the main signal past
        """
        input_layer = tf.keras.Input((self.past_dependency, 2))
        flatten_layer = tf.keras.layers.Flatten()(input_layer)
        dense_layer3_mu = tf.keras.layers.Dense(self.horizon, dtype=tf.float64)(
            flatten_layer
        )
        dense_layer3_sigma = tf.keras.layers.Dense(self.horizon, dtype=tf.float64)(
            flatten_layer
        )
        activation_layer3 = (
            tf.keras.layers.Activation("relu", dtype=tf.float64)(dense_layer3_sigma)
            + 0.1
        )

        model = tf.keras.Model(
            inputs=input_layer, outputs=[dense_layer3_mu, activation_layer3]
        )

        return model

    def define_hidden_state_prior_model(self) -> tf.keras.Model:
        """
        Define hidden states prior model
        """
        input_layer0 = tf.keras.Input((self.past_dependency, 2))
        flatten_layer0 = tf.keras.layers.Flatten()(input_layer0)
        input_layer1 = tf.keras.Input((self.horizon, self.K))
        flatten_layer1 = tf.keras.layers.Flatten()(input_layer1)
        concat_layer = tf.keras.layers.concatenate([flatten_layer0, flatten_layer1], axis=1)
        dense_layer0 = tf.keras.layers.Dense(self.horizon * self.K, dtype=tf.float64)(
            concat_layer
        )
        reshape_layer0 = tf.keras.layers.Reshape((self.horizon, self.K))(dense_layer0)
        activation_layer = tf.keras.layers.Activation("softmax", dtype=tf.float64)(
            reshape_layer0
        )
        model = tf.keras.Model(
            inputs=[input_layer0, input_layer1], outputs=activation_layer
        )

        return model

    def define_hidden_state_posterior_model(self) -> tf.keras.Model:
        """
        Define hidden states posterior model
        """
        input_layer0 = tf.keras.Input((self.past_dependency, 2))
        flatten_layer0 = tf.keras.layers.Flatten()(input_layer0)
        input_layer1 = tf.keras.Input((self.horizon, 1))
        flatten_layer1 = tf.keras.layers.Flatten()(input_layer1)
        concat_layer = tf.keras.layers.concatenate([flatten_layer0, flatten_layer1], axis=1)
        dense_layer0 = tf.keras.layers.Dense(self.horizon * self.K, dtype=tf.float64)(
            concat_layer
        )
        reshape_layer0 = tf.keras.layers.Reshape((self.horizon, self.K))(dense_layer0)
        activation_layer = tf.keras.layers.Activation("softmax", dtype=tf.float64)(
            reshape_layer0
        )
        model = tf.keras.Model(
            inputs=[input_layer0, input_layer1], outputs=activation_layer
        )

        return model

    def define_param(self) -> None:
        """
        Define parameters of the HMM model.
        """
        self.emission_models = [
            self.define_emission_model_nows() for i in range(self.K)
        ]
        self.hidden_state_prior_model = self.define_hidden_state_prior_model()
        self.hidden_state_posterior_model = self.define_hidden_state_posterior_model()

        self.hidden_state_prior_model_params = (
            self.hidden_state_prior_model.trainable_variables
        )
        self.emission_laws_params = self.emission_models.trainable_variables
        self.hidden_state_posterior_model_params = (
            self.hidden_state_posterior_model.trainable_variables
        )

    def get_param(self) -> Tuple:
        """
        Return model's parameters.
        
        Returns:
        
        - *param*: a Tuple containing the model parameters
        """
        return (
            [model.get_weights() for model in self.emission_models],
            self.hidden_state_prior_model.get_weights(),
            self.hidden_state_posterior_model.get_weights(),
        )

    def assign_param(
        self,
        emission_models_weights: List,
        hidden_state_prior_model_weights: tf.Tensor,
        hidden_state_posterior_model_weights: tf.Tensor,
    ) -> None:
        """
        Instantiate a model with existing parameters.
        
        Arguments:
        
        - *emission_models_weights*: a list containing the emission laws models' parameters
        
        - *hidden_state_prior_model_weights*: a tf.Tensor containing the hidden states model's parameters
        
        - *hidden_state_posterior_model_weights*: a tf.Tensor containing the variational model's parameters
        """
        for k in range(self.K):
            self.emission_models[k].set_weights(emission_models_weights[k])
        self.hidden_state_prior_model.set_weights(hidden_state_prior_model_weights)
        self.hidden_state_posterior_model.set_weights(
            hidden_state_posterior_model_weights
        )

    def compute_emission_laws_parameters(
        self, y_past: tf.Tensor, w_past: tf.Tensor, time_index: tf.Tensor
    ) -> tf.Tensor:
        """
        Compute the mean and std of the emission densities.
        
         Arguments:
         
         - *y_past*: a tf.Tensor(nb_time_series, past_dependency) containing the past values of the main signal.
         
         - *w*: a tf.Tensor(nb_time_series, past_dependency) containing the values of the external signal.
        
        - *time_index*: a tf.Tensor(nb_time_series, past_dependency) corresponding to an encoding of the time period of the input window.
         
        Returns:
        
        - *mu*: a tf.Tensor(nb_time_series,horizon) containing the mean of the emission densities.
        
        - *sigma*: a tf.Tensor(nb_time_series,horizon) containing the std of the emission densities.
        """
        past_input_nows = tf.stack([y_past, time_index], axis=2)
        model_pred = [self.emission_models[i](past_input_nows) for i in range(self.K)]
        mu = tf.stack([pred[0] for pred in model_pred], axis=2)
        sigma = tf.stack([pred[1] for pred in model_pred], axis=2)
        return mu, sigma

    def compute_prior_probabilities(
        self, y_past: tf.Tensor, w_past: tf.Tensor, time_index: tf.Tensor,
    ) -> tf.Tensor:
        """
        Compute the hidden states prior probabilities
        
        Arguments:
         
        - *y_past*: tf.Tensor(nb_time_series, past_dependency) containing the past values of the main signal.
        
        - *w*: tf.Tensor(nb_time_series, past_dependency) containing the values of the external signal.
        
        - *time_index*: tf.Tensor(nb_time_series, past_dependency) corresponding to an encoding of the time period of the input window.
         
        Returns:
        
        - *transition_matrix_proba*: a tf.Tensor(nb_time_series,horizon,K) containing the probabilities to be in a hidden state at each time step.
        """
        past_input = tf.stack([y_past, time_index], axis=2)
        mu, sigma = self.compute_emission_laws_parameters(y_past, w_past, time_index)
        future_input = mu
        prior_probabilities = self.hidden_state_prior_model([past_input, future_input])
        return prior_probabilities

    def compute_posterior_probabilities(
        self, y: tf.Tensor, y_past: tf.Tensor, w_past: tf.Tensor, time_index: tf.Tensor
    ) -> tf.Tensor:
        """
        Compute the hidden states posterior probabilities
        
         Arguments:
        
        - *y*: a tf.Tensor(nb_time_series, horizon) containing the current and future values of the main signal.
        
        - *y_past*: a tf.Tensor(nb_time_series, past_dependency) containing the past values of the main signal.
        
        - *w*: a tf.Tensor(nb_time_series, past_dependency) containing the values of the external signal.
        
        - *time_index*: a tf.Tensor(nb_time_series, past_dependency) corresponding to an encoding of the time period of the input window.
         
        Returns:
        
        - *transition_matrix_proba*: a tf.Tensor(nb_time_series,horizon,K) containing the probabilities to be in a hidden state at each time step.
        """
        past_input = tf.stack([y_past, time_index], axis=2)
        future_input = tf.expand_dims(y, axis=2)
        prosterior_probabilities = self.hidden_state_posterior_model(
            [past_input, future_input]
        )

        return prosterior_probabilities
