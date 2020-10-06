import sys
import argparse
import numpy as np
import tensorflow as tf
import keras
import gym
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from state_encoding import extend_feature_vec_observation
from KukaEnv_10703 import KukaVariedObjectEnv
from state_encoding import extend_feature_vec_observation
class Reinforce(object):
    # Implementation of the policy gradient method REINFORCE.

    def __init__(self, model, lr, train_config, num_classes=9):
        self.num_classes = num_classes
        self.model = model
        self.tf_input_states = model.input
        self.tf_action_probs = model.output
        self.tf_executed_actions = tf.placeholder(dtype=tf.int32, shape=[None, 1])
        self.tf_returns = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        tf_executed_probs = tf.batch_gather(self.tf_action_probs, self.tf_executed_actions)
        self.tf_L = tf.reduce_mean(self.tf_returns*tf.log(tf_executed_probs))
        self.global_step = tf.train.get_or_create_global_step()
        self.learning_rate = tf.train.exponential_decay(lr, self.global_step,
            train_config['decay_step'], train_config['decay_factor'],
            staircase=True)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = tf.train.AdamOptimizer(
                    learning_rate=self.learning_rate).minimize(-self.tf_L,
                    global_step=self.global_step)
        gpu_ops = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(gpu_options=gpu_ops)
        self.sess = tf.Session(config=config)
        # initialization
        self.sess.run(tf.initializers.global_variables())

    def save_model(self, checkpoint_path, step):
        saver = tf.train.Saver()
        saver.save(self.sess, checkpoint_path, global_step=step)

    def load_model(self, checkpoint_path):
        print('Restore from checkpoint %s' % checkpoint_path)
        saver = tf.train.Saver()
        saver.restore(self.sess, checkpoint_path)

    def train(self, env, gamma=1.0, render=False):
        # Generate an episode
        states, actions, rewards = self.generate_episode(env, render=render)
        # Compute returns
        returns = np.copy(rewards)/100.0
        total_returns = np.sum(returns)
        for i in range(returns.shape[0]-2, -1, -1):
            returns[i] += gamma*returns[i+1]
        assert np.isclose(total_returns, returns[0]), '%f != %f' % (total_returns, returns)
        # Optimize policy
        _, L = self.sess.run([self.train_op, self.tf_L], feed_dict={self.tf_input_states: states,
            self.tf_executed_actions: actions, self.tf_returns: returns})
        return

    def test(self, env, gamma=1.0, render=False):
        # Generate an episode
        states, actions, rewards = self.generate_episode(env, render=render, test_mode=True)
        returns = np.copy(rewards)
        total_returns = np.sum(returns)
        for i in range(returns.shape[0]-2, -1, -1):
            returns[i] += gamma*returns[i+1]
        assert np.isclose(total_returns, returns[0]), '%f != %f' % (total_returns, returns)
        return returns[0]

    def generate_episode(self, env, render=False, test_mode=False):
        _ = env.reset()
        state = extend_feature_vec_observation(
            env.get_feature_vec_observation(),
            num_classes=self.num_classes)
        done = False
        states = []
        actions = []
        rewards = []
        while not done:
            if render:
                env.render()
            action_probs = self.sess.run(self.tf_action_probs, feed_dict={
                self.tf_input_states: [state]})
            excuted_action = np.random.choice(action_probs[0].shape[0],
                p=action_probs[0])
            # print action_probs[0]
            # print action_probs[0].shape[0]
            # print excuted_action
            _, reward, done, info = env.step(excuted_action)
            reward = reward*1.0
            if not test_mode:
                reward = reward*100.0 - 5
            next_state = extend_feature_vec_observation(
				env.get_feature_vec_observation(),
				num_classes=self.num_classes)
            # print next_state
            states.append(state)
            actions.append([excuted_action])
            rewards.append([reward])
            state = next_state
        return states, actions, rewards


def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-config-path', dest='model_config_path',
                        type=str, default='LunarLander-v2-config.json',
                        help="Path to the model config file.")
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=50000, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=5e-4, help="The learning rate.")

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
    render = args.render

    # Create the environment.
    env = gym.make('LunarLander-v2')

    # Load the policy model from file.
    with open(model_config_path, 'r') as f:
        model = keras.models.model_from_json(f.read())

    # TODO: Train the model using REINFORCE and plot the learning curve.
    train_config = {'decay_step': 10000, 'decay_factor': 0.5}
    reinforce_model = Reinforce(model, lr, train_config)

    reward_curve = []
    for train_step in range(num_episodes):
        reinforce_model.train(env, render=render)
        if train_step % 1000 == 0:
            test_reward = []
            for j in range(100):
                test_reward.append(reinforce_model.test(env, render=False))
            mean = np.mean(test_reward)
            std = np.std(test_reward)
            print((train_step, mean, std))
            reinforce_model.save_model('./reinforce/model', step=train_step)
            reward_curve.append((train_step, mean, std))
    with open('Q1_curve.json', 'w') as f:
        json.dump(reward_curve, f)
if __name__ == '__main__':
    main(sys.argv)
