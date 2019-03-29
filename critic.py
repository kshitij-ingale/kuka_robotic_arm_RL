import numpy as np
import tensorflow as tf
import keras.backend as K

from keras.initializers import RandomUniform
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.layers import Input, Dense, concatenate, LSTM, Reshape, BatchNormalization, Lambda, Flatten


def custom_objective(y_true, y_pred):
    """
    mse + lbc
    """
    critic_target = y_true[0]
    actor_res = y_true[1]
    demo_actor_res = y_true[2]
    mse = K.mean(K.square(y_pred - critic_target), axis=-1)
    lbc = K.mean(K.square(actor_res - demo_actor_res), axis=-1)
    return mse - 0.01*lbc


class Critic:
    """ Critic for the DDPG Algorithm, Q-Value function approximator
    """

    def __init__(self, inp_dim, out_dim, lr, tau):
        # Dimensions and Hyperparams
        self.env_dim = inp_dim
        self.act_dim = out_dim
        self.tau, self.lr = tau, lr
        # Build models and target models
        self.model = self.network()
        self.target_model = self.network()
        self.model.compile(Adam(self.lr), loss=custom_objective)
        self.target_model.compile(Adam(self.lr), 'mse')
        # Function to compute Q-value gradients (Actor Optimization)
        self.action_grads = K.function([self.model.input[0], self.model.input[1]], K.gradients(self.model.output, [self.model.input[1]]))

    def network(self):
        """ Assemble Critic network to predict q-values
        """
        state = Input((self.env_dim, ))
        action = Input((self.act_dim,))
        x = Dense(256, activation='relu')(state)
        x = concatenate([x, action])
        x = Dense(128, activation='relu')(x)
        out = Dense(1, activation='linear', kernel_initializer=RandomUniform())(x)
        return Model([state, action], [out, out, out])

    def gradients(self, states, actions):
        """ Compute Q-value gradients w.r.t. states and policy-actions
        """
        return self.action_grads([states, actions])

    def target_predict(self, inp):
        """ Predict Q-Values using the target network
        """
        return self.target_model.predict(inp)

    def train_on_batch(self, states, actions, critic_target, actor_res, demo_actor_res):
        """ Train the critic network on batch of sampled experience
        """
        return self.model.train_on_batch([states, actions], [critic_target, actor_res, demo_actor_res])

    def transfer_weights(self):
        """ Transfer model weights to target model with a factor of Tau
        """
        W, target_W = self.model.get_weights(), self.target_model.get_weights()
        for i in range(len(W)):
            target_W[i] = self.tau * W[i] + (1 - self.tau)* target_W[i]
        self.target_model.set_weights(target_W)