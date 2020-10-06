import numpy as np
import tensorflow as tf
import keras.backend as K

from keras.initializers import RandomUniform
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.layers import Input, Dense, concatenate

from numpy.linalg import norm

class Curiosity:

    def __init__(self, e_dim, a_dim, lr=0.0005):
        self.env_dim = e_dim
        self.act_dim = a_dim
        self.lr = lr
        
        self.model = self.network()
        self.model.compile(Adam(self.lr), 'mse')

# Curiosity exploration parameters
        self.max_explore = 1
        self.beta_explore = 0.2
        self.C_explore = 0.01
        
    def network(self):
        state = Input((self.env_dim, ))
        action = Input((self.act_dim,))
        x = Dense(128, activation='relu')(state)
        x = concatenate([x, action])
        x = Dense(256, activation='relu')(x)
        out = Dense(self.env_dim, activation='linear', kernel_initializer=RandomUniform())(x)
        return Model([state, action], out)


    def predict(self, states, actions):

        return self.model.predict([np.array(states),np.array(actions)])

    def train_on_batch(self, states, actions, target):
        return self.model.train_on_batch([states, actions], target)

    def get_curious_reward(self,old_state,action,new_state,iteration):
        old_state = np.reshape(old_state, (1, -1))
        action = np.reshape(action, (1, -1))        
        curio_state = self.predict(old_state,action)
        err_explore = (norm(curio_state - new_state)**2)/self.max_explore
        if err_explore > self.max_explore:
            self.max_explore = err_explore
        return self.beta_explore*(err_explore/(iteration*self.C_explore))
        