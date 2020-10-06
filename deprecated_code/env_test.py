#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import random
from collections import deque
import os
import tensorflow as tf
import argparse

from matplotlib import pyplot as plt
from KukaEnv_10703 import KukaVariedObjectEnv
import pybullet as p
from keras.utils import to_categorical
from keras import models
from keras import layers
from keras import optimizers
from ddpg_her import DDPG

T = 1000
n_object = 9
lr = 1e-3


def get_policy(nstate, naction):
    model = models.Sequential()
    model.add(layers.Dense(24, input_dim=nstate, activation='relu'))
    model.add(layers.Dense(naction, activation='tanh'))
    optimizer = optimizers.Adam(lr=lr)
    model.compile(optimizer, loss='mse')
    return model


def get_state(env):
    state = env.get_feature_vec_observation()
    res = state[0:-1]
    res.extend(to_categorical(state[-1], n_object))
    return res


def run(env, episode=20):
    n_action = len(env.action_space.sample())
    n_state = len(get_state(env))
    policy = get_policy(n_state, n_action)

    done = False
    for _ in range(episode):
        env.reset()
        blockPos, blockOrn = p.getBasePositionAndOrientation(env.blockUid)
        t = 0
        experience = []
        while t < T and not done:
            state = get_state(env)
            state = np.reshape(state, (1, -1))
            action = policy.predict(state)
            action = np.squeeze(action)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            gripperState  = p.getLinkState(env._kuka.kukaUid, env._kuka.kukaGripperIndex)
            gripperPos = gripperState[0]
            gripperOrn = gripperState[1]
            _, reward, done, _ = env.step(action)
            if reward == 1:
                reward = 1000
            else:
                reward = -100
            next_state = get_state(env)
            gripperState  = p.getLinkState(env._kuka.kukaUid, env._kuka.kukaGripperIndex)
            next_gripperPos = gripperState[0]
            next_gripperOrn = gripperState[1]
            state = np.squeeze(state).tolist()
            experience.append((state, action, reward, next_state, gripperPos, gripperOrn, next_gripperPos, next_gripperOrn, blockPos, blockOrn))

        print('experience', blockPos, blockOrn)
        for transition in experience:
            print(transition[0])
        # use gripper to create a new goal
        blockPos, blockOrn = next_gripperPos, next_gripperOrn
        # blockPos, blockOrn = gripperPos, gripperOrn
        step_size = len(experience)
        sample_goal_exprience = []
        for t in range(step_size):
            state, action, reward, next_state, gripperPos, gripperOrn, next_gripperPos, next_gripperOrn, _, _  = np.copy(experience[t])
            blockInGripperPosXYEulZ = env.get_block_in_gripper_pos(gripperPos, gripperOrn, blockPos, blockOrn)
            state[6:9] = blockInGripperPosXYEulZ
            next_blockInGripperPosXYEulZ = env.get_block_in_gripper_pos(next_gripperPos, next_gripperOrn, blockPos, blockOrn)
            next_state[6:9] = next_blockInGripperPosXYEulZ

            if t == step_size - 1:
                reward = 500
            sample_goal_exprience.append((state, action, reward, next_state, gripperPos, gripperOrn, next_gripperPos, next_gripperOrn, blockPos, blockOrn))
        print('experience', blockPos, blockOrn)
        for transition in sample_goal_exprience:
            print(transition)


if __name__ == '__main__':
    item_dir = '/home/kshitij/Desktop/RL/kuka_robotic_arm_RL/items'
    env = KukaVariedObjectEnv(item_dir, renders=False, isDiscrete=False)

    state = env.reset()
    reward = 0
    while reward == 0:
        done = False
        while not done:
            # Sample a random action.
            action = env.action_space.sample()
            # Run a simulation step using the sampled action.
            new_state, reward, done, info = env.step(action)
            state = new_state