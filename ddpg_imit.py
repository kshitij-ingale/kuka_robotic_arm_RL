import sys
import numpy as np

from tqdm import tqdm
from actor import Actor
from critic import Critic
from utils.stats import gather_stats
from utils.networks import tfSummary, OrnsteinUhlenbeckProcess
from collections import deque
from keras.utils import to_categorical
import random
from KukaEnv_10703 import KukaVariedObjectEnv
import pybullet as p

n_object = 9


class Replay:
    def __init__(self, memory_size=50000):
        self.memo = deque(maxlen=memory_size)
        self.demo_memo = deque(maxlen=memory_size)

    def sample_batch(self, batch_size):
        return random.sample(self.memo, batch_size), random.sample(self.demo_memo, batch_size)

    def append(self, transition):
        self.memo.append(transition)

    def append_demo(self, transition):
        self.demo_memo.append(transition)


def get_state(env):
    state = env.get_feature_vec_observation()
    res = state[0:-1]
    res.extend(to_categorical(state[-1], n_object))
    return res


class DDPG:
    """ Deep Deterministic Policy Gradient (DDPG) Helper Class
    """

    def __init__(self, act_dim, env_dim, act_range, buffer_size = 20000, gamma=0.99, lr=0.00005, tau=0.001):
        """ Initialization
        """
        # Environment and A2C parameters
        self.act_dim = act_dim
        self.act_range = act_range
        self.env_dim = env_dim
        self.gamma = gamma
        # Create actor and critic networks
        self.actor = Actor(self.env_dim, act_dim, act_range, 0.1 * lr, tau)
        self.demo_actor = Actor(self.env_dim, act_dim, act_range, 0.1 * lr, tau)
        self.critic = Critic(self.env_dim, act_dim, lr, tau)
        self.buffer = Replay()
        self.batch_size = 2000

    def policy_action(self, s):
        """ Use the actor to predict value
        """
        return self.actor.predict(s)[0]

    def bellman(self, rewards, q_values, dones):
        """ Use the Bellman Equation to compute the critic target
        """
        critic_target = np.asarray(q_values)

        # if dones:
        #     critic_target[0] = rewards
        # else:
        #     critic_target[0] = rewards + self.gamma * q_values

        for i in range(q_values.shape[0]):
            if dones:
                critic_target[i] = rewards[i]
            else:
                critic_target[i] = rewards[i] + self.gamma * q_values[i]
        return critic_target

    def memorize(self, state, action, reward, done, new_state):
        """ Store experience in memory buffer
        """
        self.buffer.append((state, action, reward, done, new_state))

    def sample_batch(self):
        return self.buffer.sample_batch(self.batch_size)

    def update_models(self, states, actions, critic_target, actor_res, demo_actor_res):
        """ Update actor and critic networks from sampled experience
        """
        # Train critic
        # print('critic_target', critic_target)
        self.critic.train_on_batch(states, actions, critic_target, actor_res, demo_actor_res)
        # Q-Value Gradients under Current Policy
        actions = self.actor.model.predict(states)
        grads = self.critic.gradients(states, actions)
        demo_actions = self.demo_actor.model.predict(states)
        demo_grads = self.critic.gradients(states, demo_actions)
        # Train actor
        self.actor.train(states, actions, np.array(grads).reshape((-1, self.act_dim)))
        self.demo_actor.train(states, demo_actions, np.array(demo_grads).reshape((-1, self.act_dim)))
        # Transfer weights to target networks at rate Tau
        self.actor.transfer_weights()
        self.demo_actor.transfer_weights()
        self.critic.transfer_weights()

    def train(self, env):
        results = []

        # First, gather experience
        tqdm_e = tqdm(range(50000), desc='Score', leave=True, unit=" episodes")
        success = []
        for e in tqdm_e:

            # Reset episode
            time, cumul_reward, done = 0, 0, False
            old_state = env.reset()
            actions, states, rewards = [], [], []
            noise = OrnsteinUhlenbeckProcess(size=self.act_dim)
            blockPos, blockOrn = p.getBasePositionAndOrientation(env.blockUid)
            experience = []

            while not done:
                # Actor picks an action (following the deterministic policy)
                old_state = get_state(env)
                a = self.policy_action(old_state)
                # Clip continuous values to be valid w.r.t. environment

                gripperState  = p.getLinkState(env._kuka.kukaUid, env._kuka.kukaGripperIndex)
                gripperPos = gripperState[0]
                gripperOrn = gripperState[1]

                a = np.clip(a+noise.generate(time), -self.act_range, self.act_range)
                # Retrieve new state, reward, and whether the state is terminal
                new_state, r, done, _ = env.step(a)
                new_state = get_state(env)


                gripperState  = p.getLinkState(env._kuka.kukaUid, env._kuka.kukaGripperIndex)
                next_gripperPos = gripperState[0]
                next_gripperOrn = gripperState[1]

                # Add outputs to memory buffer
                experience.append((old_state, a, r, done, new_state, gripperPos, gripperOrn, next_gripperPos, next_gripperOrn, blockPos, blockOrn))
                # self.memorize(old_state, a, r, done, new_state)

                # HER replay, sample a new goal
                blockPos, blockOrn = gripperPos, gripperOrn
                step_size = len(experience)
                her_experience = []
                for t in range(step_size):
                    old_state, action, reward, done, next_state, gripperPos, gripperOrn, next_gripperPos, next_gripperOrn, _, _ = np.copy(experience[t])
                    blockInGripperPosXYEulZ = env.get_block_in_gripper_pos(gripperPos, gripperOrn, blockPos, blockOrn)
                    old_state[6:9] = blockInGripperPosXYEulZ
                    next_blockInGripperPosXYEulZ = env.get_block_in_gripper_pos(next_gripperPos, next_gripperOrn, blockPos, blockOrn)
                    next_state[6:9] = next_blockInGripperPosXYEulZ
                    if t == step_size - 1:
                        reward = 0.5
                    her_experience.append((old_state, action, reward, done, next_state, gripperPos, gripperOrn, next_gripperPos, next_gripperOrn, blockPos, blockOrn))

                self.train_batch()

                # Update current state
                old_state = new_state

                if r > 0:
                    print('r', r)
                    success.append(e)
                    print(success)
                cumul_reward += r
                time += 1
            self.buffer.memo.extend(experience)
            self.buffer.demo_memo.extend(her_experience)

            # Gather stats every episode for plotting
            tqdm_e.set_description("Score: " + str(cumul_reward))
            tqdm_e.refresh()

        return results

    def train_batch(self):
        if len(self.buffer.memo) > self.batch_size and len(self.buffer.demo_memo) > self.batch_size:
            # Sample experience from buffer
            sample_batch, sample_demo_batch = self.sample_batch()
            states = []
            actions = []
            rewards = []
            dones = []
            new_states = []
            samples_size = len(sample_batch)

            for state, action, reward, done, new_state, _, _, _, _, _, _ in sample_batch:
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                new_states.append(new_state)

            new_states = np.reshape(np.array(new_states), (samples_size, -1))
            actor_res = self.actor.target_predict(new_states)
            demo_actor_res = self.demo_actor.target_predict(new_states)
            q_values = self.critic.target_predict([new_states, actor_res])[0]
            q_values = np.reshape(q_values, (samples_size, ))
            critic_targets = self.bellman(rewards, q_values, dones)
            states = np.array(states)
            actions = np.array(actions)
            self.update_models(states, actions, critic_targets, actor_res, demo_actor_res)

            # for state, action, reward, done, new_state, _, _, _, _, _, _ in sample_batch:
            # # for state, action, reward, done, new_state in sample_batch:
            #     # Predict target q-values using target networks
            #     new_state = np.reshape(new_state, (1, -1))
            #     q_value = self.critic.target_predict([new_state, self.actor.target_predict(new_state)])
            #     # Compute critic target
            #     q_value = np.reshape(q_value, (1, ))
            #     critic_target = self.bellman(reward, q_value, done)
            #     # Train both networks on sampled batch, update target networks
            #     state = np.reshape(state, (1, -1))
            #     action = np.reshape(action, (1, -1))
            #     self.update_models(state, action, critic_target)


if __name__ == '__main__':
    item_dir = '/home/kshitij/Desktop/RL/kuka_robotic_arm_RL/items'
    # item_dir = sys.argv[1]
    env = KukaVariedObjectEnv(item_dir, isDiscrete=False)

    state = env.reset()
    # reward = 0
    # while reward == 0:
    #     done = False
    #     env.reset()
    #     while not done:
    #         # Sample a random action.
    #         action = env.action_space.sample()
    #         # Run a simulation step using the sampled action.
    #         new_state, reward, done, info = env.step(action)
    #         print(reward)
    #         state = new_state
    #
    n_action = len(env.action_space.sample())
    n_state = len(get_state(env))
    # state, xpos, ypos, angle = env.reset()
    state = env.reset()
    # print(get_state(env))

    # no
    # 0.5 0.0001

    ddpg = DDPG(n_action, n_state, env.action_space.high)
    stats = ddpg.train(env)