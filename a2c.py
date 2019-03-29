import sys
import argparse
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import keras
import gym
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from reinforce import Reinforce
from KukaEnv_10703 import KukaVariedObjectEnv
from state_encoding import extend_feature_vec_observation


class Replay_Memory():
    """ The memory essentially stores transitions recorder from the agent
    taking actions in the environment. """

    def __init__(self, env, memory_size=50000, pos_ctr=0.0):
        """
        Args:
            env: openAI gym environment.
            memory_size: the total number of transitions to buffer.
            burn_in: the number of transitions to initialize the buffer.
        """
        self._buffer_size = memory_size
        self._buffer_ptr = 0
        self._buffer = []
        self._pos_ctr = pos_ctr

    def pos_factor(self):
        return  np.float32(self._pos_ctr) / (len(self._buffer) + 0.00001)

    def sample_batch(self, batch_size=32):
        """ Sample transitions in the buffer.

        Args:
            batch_size: the number of transitions to sample.

        Raise: AssertError if batch_size is larger than the total number of
        transitions stored.
        """
        assert batch_size < len(self._buffer), "Sample more than buffer has"
        samples_idx = np.random.choice(len(self._buffer), batch_size)
        samples = []
        for sample_idx in samples_idx:
            samples.append(self._buffer[sample_idx])
        return samples

    def append(self, transition, is_pos):
        """Append transition to the memory.

        Args:
            transition: a (state, action, reward, next_state, done) tuple.
        """
        if len(self._buffer) < self._buffer_size:
            self._buffer.append(transition)
            self._buffer_ptr += 1
        else:
            # FIFO update
            self._buffer_ptr = self._buffer_ptr % self._buffer_size
            self._buffer[self._buffer_ptr] = transition
            self._buffer_ptr += 1
        if is_pos:
            self._pos_ctr += 1


class A2C(Reinforce):
    # Implementation of N-step Advantage Actor Critic.
    # This class inherits the Reinforce class, so for example, you can reuse
    # generate_episode() here.

    def __init__(self, model, lr, critic_model, critic_lr, train_config, n=20,
    state_size=18, num_classes=9, replay_mem_size=1000, pos_balance_factor=0.5):
        self.n = n
        self.num_classes = num_classes
        # agent model
        self.model = model
        self.tf_input_states = tf.placeholder(dtype=tf.float32, shape=[None, state_size])
        features = self.tf_input_states
        for layer_idx in range(len(model)-1):
            with tf.variable_scope('layer%d'%layer_idx, reuse=False):
                features = slim.fully_connected(features, model[layer_idx])
        features = slim.fully_connected(features, model[-1], activation_fn=None)
        self.tf_action_probs = tf.nn.softmax(features)
        self.tf_executed_actions = tf.placeholder(dtype=tf.int32, shape=[None, 1])
        self.tf_returns_with_baseline = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        


        # tf_executed_probs = tf.batch_gather(self.tf_action_probs, self.tf_executed_actions)
        tf_executed_probs = tf.gather(self.tf_action_probs, self.tf_executed_actions)



        self.tf_L = tf.reduce_mean(self.tf_returns_with_baseline*tf.log(tf_executed_probs))
        self.global_step = tf.train.get_or_create_global_step()
        self.learning_rate = tf.train.exponential_decay(lr, self.global_step,
            train_config['decay_step'], train_config['decay_factor'],
            staircase=True)
        self.train_op = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate).minimize(-self.tf_L,
                global_step=self.global_step)
        # critic_model
        self.critic_model = critic_model
        self.tf_critic_input_states = tf.placeholder(dtype=tf.float32, shape=[None, state_size])
        features = self.tf_critic_input_states
        for layer_idx in range(len(critic_model)-1):
            with tf.variable_scope('Critic_layer%d'%layer_idx, reuse=False):
                features = slim.fully_connected(features, critic_model[layer_idx])
        features = slim.fully_connected(features, critic_model[-1],
                    activation_fn=None)
        self.tf_state_value = features
        self.tf_returns = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.tf_critic_L = tf.losses.mean_squared_error(self.tf_returns, self.tf_state_value)
        self.critic_global_step = tf.train.get_or_create_global_step()
        self.critic_learning_rate = tf.train.exponential_decay(critic_lr, self.critic_global_step,
            train_config['critic_decay_step'], train_config['critic_decay_factor'],
            staircase=True)
        self.train_critic_op = tf.train.AdamOptimizer(
                learning_rate=self.critic_learning_rate).minimize(self.tf_critic_L,
                global_step=self.critic_global_step)
        # initialization
        gpu_ops = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(gpu_options=gpu_ops)
        self.sess = tf.Session(config=config)
        self.sess.run(tf.initializers.global_variables())
        self.replay_buffer = Replay_Memory(replay_mem_size)

    def train(self, env, gamma=1.0, render=False):
        # Trains the model on a single episode using A2C.
        # TODO: Implement this method. It may be helpful to call the class
        #       method generate_episode() to generate training data.

        # Training data
        states, actions, rewards = self.generate_episode(env, render=render)
        rewards = np.array(rewards)
        # Compute N step return
        state_value = self.sess.run(self.tf_state_value,
            feed_dict={self.tf_critic_input_states: states})
        returns = np.copy(rewards)
        T = returns.shape[0]
        for i in range(1, min(self.n, T)):
            returns[:T-i] += rewards[i:]*(gamma**i)
        if T >= self.n:
            returns[:T-self.n] += state_value[self.n:]*(gamma**self.n)
        # train agent
        self.sess.run(self.train_op, feed_dict={self.tf_input_states:states,
            self.tf_executed_actions: actions,
            self.tf_returns_with_baseline: (returns-state_value)})
        # train critic
        self.sess.run(self.train_critic_op, feed_dict={
            self.tf_critic_input_states:states, self.tf_returns: returns})
        return



def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-config-path', dest='model_config_path',
                        type=str, default='LunarLander-v2-config.json',
                        help="Path to the actor model config file.")
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=100000, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=5e-4, help="The actor's learning rate.")
    parser.add_argument('--critic_lr', dest='critic_lr', type=float,
                        default=1e-4, help="The critic's learning rate.")
    parser.add_argument('--gamma', dest='gamma', type=float,
                        default=1e-4, help="The critic's learning rate.")
    parser.add_argument('--decay_step', dest='decay_step', type=float,
                        default=5e-4, help="The actor's learning rate.")
    parser.add_argument('--critic_decay_step', dest='critic_decay_step', type=float,
                        default=1e-4, help="The critic's learning rate.")
    parser.add_argument('--n', dest='n', type=int,
                        default=20, help="The value of N in N-step A2C.")
    parser.add_argument('--train_dir',dest='train_dir',type=str,default='./')
    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--render', dest='render',
                              action='store_true',
                              help="Whether to render the environment.")
    parser_group.add_argument('--no-render', dest='render',
                              action='store_false',
                              help="Whether to render the environment.")
    parser.set_defaults(render=False)

    return parser.parse_args()


def main(args):
    # Parse command-line arguments.
    args = parse_arguments()
    model_config_path = args.model_config_path
    num_episodes = args.num_episodes
    lr = args.lr
    critic_lr = args.critic_lr
    n = args.n
    render = args.render

    # Create the environment.
    item_path = '/home/kshitij/Desktop/10703project/items/'
    env = KukaVariedObjectEnv(item_path, renders=True,isDiscrete=True, maxSteps = 10000000)

    train_config = {'decay_step': args.decay_step, 'decay_factor': 0.5,
                    'critic_decay_step': args.critic_decay_step, 'critic_decay_factor': 0.5}

    a2c_model = A2C((512, 512, 512, 512, 7), lr, (512, 512, 512, 512, 1), critic_lr, train_config, n=n, state_size=18, num_classes=9,)

    reward_curve = []
    a2c_model.load_model('model/model-2000')    
    test_reward = []
    for j in range(100):
        print(j)
        test_reward.append(a2c_model.test(env, render=True))

    mean = np.mean(test_reward)
    std = np.std(test_reward)
    print((mean, std))
    

if __name__ == '__main__':
    main(sys.argv)
