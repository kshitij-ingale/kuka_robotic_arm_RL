from keras.utils import to_categorical
import numpy as np, sys
from keras import models
from keras import layers
from keras import optimizers
from collections import deque
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from KukaEnv_10703 import KukaVariedObjectEnv
from local import canonical_plot

T = 1000
n_object = 9


class QNetwork:
    # This class essentially defines the network architecture.
    # The network should take in state of the world as an input,
    # and output Q values of the actions available to the agent as the output.

    def __init__(self, nstate, naction, lr):
        # Define your network architecture here. It is also a good idea to define any training operations
        # and optimizers here, initialize your variables, or alternately compile your model here.
        model = models.Sequential()
        model.add(layers.Dense(24, input_dim=nstate, activation='relu'))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(naction))
        model.compile(loss='mse', optimizer=optimizers.Adam(lr=lr))

        self.model = model


class Replay_Memory:
    def __init__(self, memory_size=50000, burn_in=10000):
        self.prio_memo = deque(maxlen=memory_size)
        self.memo = deque(maxlen=memory_size)

    def sample_batch(self, batch_size=32):
        if len(self.prio_memo) > batch_size//4:
            sample = random.sample(self.memo, batch_size)
            sample.extend(random.sample(self.prio_memo, batch_size//4))
        else:
            sample = random.sample(self.memo, batch_size)
            sample.extend(self.prio_memo)
        return sample

    def append(self, transition):
        # Appends transition to the memory.
        self.memo.append(transition)

    def append_prio(self, transition):
        # Appends transition to the memory.
        self.prio_memo.append(transition)


class DQN_Agent:
    # In this class, we will implement functions to do the following.
    # (1) Create an instance of the Q Network class.
    # (2) Create a function that constructs a policy from the Q values predicted by the Q Network.
    #       (a) Epsilon Greedy Policy.
    # 		(b) Greedy Policy.
    # (3) Create a function to train the Q Network, by interacting with the environment.
    # (4) Create a function to test the Q Network's performance on the environment.
    # (5) Create a function for Experience Replay.

    def __init__(self, env, render=False):

        # Create an instance of the network itself, as well as the memory.
        # Here is also a good place to set environmental parameters,
        # as well as training parameters - number of episodes / iterations, etc.
        # self.env_name = env_name
        self.env = env
        # self.nstate = len(self.env.observation_space.high)
        self.nstate = len(get_state(env))
        self.naction = self.env.action_space.n

        self.replay = Replay_Memory()
        self.lr = 0.001
        self.gamma = 0.5

        self.epsilon = 0.99
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.0000045

        self.batch_size = 32
        self.episode = 50000
        if render:
            self.episode = 10000
        self.videostep = self.episode // 3

        self.model = QNetwork(self.nstate, self.naction, self.lr).model

    def epsilon_greedy_policy(self, q_values, ceps=None):
        # Creating epsilon greedy probabilities to sample from.
        eps = self.epsilon
        if ceps is not None:
            eps = ceps

        if np.random.rand() <= eps:
            return random.randrange(self.naction)
        return np.argmax(q_values[0]).item()

    def greedy_policy(self, q_values):
        # Creating greedy policy for test time.
        pass

    def plot(self, train_reward, test_reward):
        x = range(100, len(test_reward) * 100, 100)
        xi = [i for i in range(len(x))]
        plt.title("Avg total reward")
        plt.plot(train_reward)
        plt.plot(test_reward)
        plt.legend(['train', 'test'])
        plt.xticks(xi, x)
        plt.savefig('training.jpg')

    def run(self):
        train_reward = []
        test_reward = []
        max_reward = 0.0

        for ep in range(self.episode):
            # print('Epoch {} start'.format(ep))
            state = self.env.reset()
            state = np.array(get_state(env))
            state = state.reshape((1, -1))

            is_terminal = False
            t = 0
            total_reward = 0.0

            while t < 200 and not is_terminal:
                qvalue = self.model.predict(state)
                action = self.epsilon_greedy_policy(qvalue)
                next_state, reward, is_terminal, _ = self.env.step(action)
                next_state = np.array(get_state(env))
                next_state = next_state.reshape((1, -1))
                # reward = 100 if reward == 1 else -1
                if reward == 1:
                    reward = 1000
                    self.replay.append_prio((state, action, reward, next_state, is_terminal))
                else:
                    reward = -10
                    self.replay.append((state, action, reward, next_state, is_terminal))
                total_reward += 1 if reward == 1000 else 0
                state = next_state
                t += 1
            if total_reward > 0:
                print("Episode {}# Score: {}".format(ep, total_reward), t, self.epsilon)
            self.train()

            if (ep+1) % 100 == 0:
                train_reward.append(total_reward)
                treward = self.test()
                test_reward.append(treward)
                print('Epoch {} test reward {}'.format(ep, treward))
                print('test_reward', test_reward)
                canonical_plot.plot(prefix='dqn/', rewards=train_reward)

                if treward > max_reward:
                    max_reward = treward
                    self.model.save('model/checkpoint.h5')

    def train(self):
        if len(self.replay.memo) < self.batch_size:
            return
        sample_batch = self.replay.sample_batch(self.batch_size)
        for state, action, reward, next_state, done in sample_batch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target

            self.model.fit(state, target_f, verbose=False)
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

    def test(self, episode=20, model_file=None):
        model = self.model
        if model_file is not None:
            model = models.load_model(model_file)
        total_reward = []
        for e in range(episode):
            state = self.env.reset()
            state = np.array(get_state(env))
            state = state.reshape((1, -1))
            r = 0.0
            for t in range(200):
                qvalue = model.predict(state)
                action = self.epsilon_greedy_policy(qvalue, 0.05)
                nextstate, reward, is_terminal, debug_info = self.env.step(action)
                nextstate = np.array(get_state(env))
                nextstate = nextstate.reshape((1, -1))
                r += reward
                if is_terminal:
                    break
                else:
                    state = nextstate
            total_reward.append(r)
        print("test reward", np.mean(total_reward), np.std(total_reward))
        return np.mean(total_reward)

    def burn_in_memory(self):
        # Initialize your replay memory with a burn_in number of episodes / transitions.
        pass

    def load_test(self, env, model_file):
        # self.env = Monitor(env, 'video/', force=True)
        env.reset()
        treward = self.test(model_file=model_file, episode=100)
        env.close()


def get_state(env):
    state = env.get_feature_vec_observation()
    res = state[0:-1]
    res.extend(to_categorical(state[-1], n_object))
    return res


if __name__ == '__main__':
    # main(sys.argv)
    # for i in range(100):
    #     env.reset()
    #     done = False
    #     while not done:
    #         # Sample a random action.
    #         action = env.action_space.sample()
    #         action = np.random.randint(0,6,size=1)[0].item()
    #         # print(action)
    #         # Run a simulation step using the sampled action.
    #         new_state, reward, done, info = env.step(action)
    #         print('feature', get_state(env))
    #         if reward > 0:
    #             print(i)
    #         state = new_state
    item_dir = sys.argv[1]
    env = KukaVariedObjectEnv(item_dir, renders=True, isDiscrete=True)
    state = env.reset()
    done = False
    naction = env.action_space.n
    nstate = len(env.get_feature_vec_observation())
    print(naction, nstate)

    agent = DQN_Agent(env)
    # agent.run()
    agent.load_test(env, 'model/checkpoint.h5')
