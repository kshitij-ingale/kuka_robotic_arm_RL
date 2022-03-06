""" Script to generate neural networks for policy network and state value function estimator """

import tensorflow as tf


class DenseBlock(tf.keras.layers.Layer):
    def __init__(self, units, norm, activation):
        """ Fully connected layer block consisting of dense -> (batch) norm -> activation

        Parameters
        ----------
        units : int
            Number of hidden units in fully connected layer
        norm : str
            Normalization layer to be used (batch/instance/None) 
        activation : function from tf.nn module
            Activation function to be used for output

        Raises
        ------
        NotImplementedError
            Instance norm requires tensorflow-addons as of now, so that is not 
            implemented yet
        SyntaxError
            If normalization is not specified from (batch, instance, None), error is raised
        """
        super().__init__()
        dense = tf.keras.layers.Dense(units)
        self.functions = [dense]
        if norm is not None:
            if norm == "batch":
                self.functions.append(tf.keras.layers.BatchNormalization())
            elif norm == "instance":
                raise NotImplementedError(
                    "Instance norm requires tensorflow-addons (to be added later)"
                )
            else:
                raise SyntaxError(f"{norm} normalization not found")
        self.functions.append(activation)

    def call(self, x):
        for function in self.functions:
            x = function(x)
        return x


class Net(tf.keras.Model):
    def __init__(self, state_dim, action_dim, net_params, name, duel):
        """ Creates tf.keras model by building neural network from 
        configuration specified in yml file

        Parameters
        ----------
        state_dim : int
            Dimension of state vector obtained from environment
        action_dim : int
            Number of possible (discrete) actions
        net_params : OD (dict)
            Model configuration specifed in yml file in the form of 
            object dictionary (inherits dict)
        name : str
            Name for the tf.keras model
        duel : bool
            Use dueling networks with separate branch for value and 
            advantage function
        """
        super().__init__(name=name)
        self.hidden_layers = []
        for num_hidden_unit in net_params.hidden_units:
            self.hidden_layers.append(
                DenseBlock(
                    num_hidden_unit,
                    norm=net_params.normalization,
                    activation=getattr(tf.nn, net_params.activation),
                )
            )
        self.duel = duel
        if self.duel:
            self.value = tf.keras.layers.Dense(1, )
            self.advantage = tf.keras.layers.Dense(action_dim)
        else:
            self.output_layer = tf.keras.layers.Dense(action_dim)

    def call(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        if self.duel:
            value_fn = self.value(x)
            advantage_fn = self.advantage(x)
            x = value_fn + (advantage_fn - tf.reduce_mean(advantage_fn))
        else:
            x = self.output_layer(x)
        return x

    def print_summary(self, state_dim):
        """ Prints keras model summary (only used for debug)

        Parameters
        ----------
        state_dim : int
            Dimension of state vector obtained from environment
        """
        x = tf.keras.Input(shape=(state_dim,))
        print(tf.keras.Model(inputs=[x], outputs=self.call(x)).summary())


def state_function_estimator_loss(PredictedQvalues, targets):
    """ Loss function for state value function estimator network

    Parameters
    ----------
    PredictedQvalues : tf.Tensor
        Value functions predicted by model
    targets : tf.Tensor
        Targets for model obtained by Monte Carlo or n-step returns

    Returns
    -------
    tf.Tensor
        MSE between estimated and target value functions
    """
    return tf.reduce_mean(tf.keras.losses.mean_squared_error(PredictedQvalues, targets))


@tf.function
def train_step(model, optimizer, states, targets):
    """ Training step for state value function estimator network

    Parameters
    ----------
    model : tf.keras.Model
        State value function estimator network model
    optimizer : tf.keras.optimizers
        Optimizer for state value function estimator network gradient update step
    states : tf.Tensor
        Batch of state vectors obtained from environment
    targets : tf.Tensor
        Targets for state value function estimator model obtained by Monte Carlo or
         n-step returns

    Returns
    -------
    tf.Tensor
        MSE between estimated and target value functions
    """
    with tf.GradientTape() as tape:
        PredictedQvalues = model(states)
        loss_value = state_function_estimator_loss(PredictedQvalues, targets)
    gradients = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss_value

def update_target_model(current_model, target_model, smoothing=1.0):
    """ Update variables of target model with values of current model
    For stable training, target QNet model is not updated for some steps so that target Q-values for QNet model
    are not changing

    Parameters
    ----------
    current_model : tf.keras.Model
        Current Model of Q-values prediction
    target_model : tf.keras.Model
        Target Model of Q-values prediction to be updated
    smoothing : float
        Smoothing parameter for soft update of values, default: 1.0
    """
    target_model_vars = target_model.trainable_variables
    current_model_vars = current_model.trainable_variables
    for target_model_var, current_model_var in zip(target_model_vars, current_model_vars):
        target_model_var.assign((smoothing * current_model_var) + ((1 - smoothing) * target_model_var))
