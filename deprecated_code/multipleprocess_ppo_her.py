"""This file implements PPO with clipped object function.
ref: https://arxiv.org/pdf/1707.06347.pdf
"""
from multiprocessing import Process, Queue
import time

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import json
import pybullet as p

from KukaEnv_10703 import KukaVariedObjectEnv
from state_encoding import extend_feature_vec_observation, separate_state_goal

class Memory(object):
    """ The memory essentially stores transitions recorder from the agent
    taking actions in the environment. """

    def __init__(self, memory_size=50000):
        """
        Args:
        memory_size: the max number of transitions to store.
        """
        self._max_buffer_size = memory_size
        self._buffer_ptr = 0
        self._buffer = []

    def sample_batch(self, batch_size=32):
        """ Sample transitions in the buffer.

        Args:
            batch_size: the number of transitions to sample.

        Raise: AssertError if batch_size is larger than the total number of
            transitions stored.
        """
        if batch_size > len(self._buffer):
            print('Warning: Sampling more than buffer size')
        shuffled_idx = np.random.permutation(len(self._buffer))
        buffer_len = len(self._buffer)
        samples = [self._buffer[shuffled_idx[i]] for i in range(min(batch_size, buffer_len))]
        return samples

    def append(self, transition):
        """Append transition to the memory.
        Args:
            transition: a (state, action, reward, next_state, done) tuple.
        """

        if len(self._buffer) < self._max_buffer_size:
            self._buffer.append(transition)
            self._buffer_ptr += 1
        else:
            self._buffer_ptr = self._buffer_ptr % self._max_buffer_size
            self._buffer[self._buffer_ptr] = transition
            self._buffer_ptr += 1

    def append_list(self, transitions):
        """ append a list of transitions. """
        for transition in transitions:
            self.append(transition)

    def clear(self):
        """Clear buffer"""
        self._buffer_ptr = 0
        self._buffer = []

def gaussion_fn(mean_tensor):
    """Convert a mean value to a gaussion distribution.
    Because std must be positive, we predict log_std instead"""
    with tf.variable_scope('gaussion_fn', reuse=False):
        log_std = tf.get_variable(shape=(1, mean_tensor.shape[1]), dtype=tf.float32,
            initializer=tf.zeros_initializer(), name='std')
    return [mean_tensor, log_std]

def gaussion_entropy(gaussion_distribution):
    """Compute the entropy bonous for gaussion distribution.
    Refer to: http://www.biopsychology.org/norwich/isp/chap8.pdf"""
    mean, log_std = gaussion_distribution
    return tf.reduce_mean(0.5*(tf.log(2.0*np.pi)+1+2.0*log_std))

def gaussion_prob(gaussion_distribution, action):
    """Compute the pdf of the gaussion distribution."""
    mean, log_std = gaussion_distribution
    std = tf.exp(log_std)
    prob = tf.exp(-0.5*(action - mean)**2 / std**2) / (2*np.pi*std**2)**0.5
    prob = tf.reduce_prod(prob, axis=1, keepdims=True)
    return prob

def np_gaussion_prob(np_gaussion_distribution, np_action):
    """Compute the pdf of the gaussion distribution."""
    mean, log_std = np_gaussion_distribution
    std = np.exp(log_std)
    prob = np.exp(-0.5*(np_action - mean)**2 / std**2) / (2*np.pi*std**2)**0.5
    prob = np.prod(prob, axis=1, keepdims=True)
    return prob

def policy_fn(state_size, goal_size, model_config):
    """Construct policy network based on model_config.

    Args:
        state_size: the size of the state. e.g. 18.
        model_config:
            a list of tuples, [(shared layer), ..., (policy layer, value layer)].
            e.g. [(64,), (64,), (64, 64)]
    returns: input_state_tensor, output_action_distribution, output_state_value
    """
    input_state_tensor = tf.placeholder(shape=(None, state_size), dtype=tf.float32)
    input_goal_tensor = tf.placeholder(shape=(None, goal_size), dtype=tf.float32)
    features = (tf.concat([input_state_tensor,input_goal_tensor], axis=1), )
    with slim.arg_scope([slim.fully_connected], weights_regularizer=slim.l1_regularizer(0.0000)):
        for layer_idx, layer_config in enumerate(model_config):
            with tf.variable_scope('layer_%d'%layer_idx, reuse=False):
                if layer_idx != (len(model_config)-1):
                    if len(layer_config) == 1:
                        assert len(features) == 1, "Merging features are not supported"
                        features = (slim.fully_connected(features[0], layer_config[0]),)
                    else:
                        assert len(layer_config) == 2, "Layer config can only have 1 or 2 value"
                        if len(features) == 1:
                            features = (slim.fully_connected(features[0], layer_config[0]),
                                slim.fully_connected(features[0], layer_config[1]))
                        else:
                            features = (slim.fully_connected(features[0], layer_config[0]),
                                slim.fully_connected(features[1], layer_config[1]))
                else:
                    assert len(layer_config) == 2, 'model must has two heads'
                    if len(features) == 1:
                        features = (slim.fully_connected(features[0], layer_config[0],
                            activation_fn=None,),
                            slim.fully_connected(features[0], layer_config[1],
                            activation_fn=None,))
                    else:
                        features = (slim.fully_connected(features[0], layer_config[0],
                            activation_fn=None,),
                            slim.fully_connected(features[1], layer_config[1],
                            activation_fn=None,))
        actor_feautre, critic_feature = features
        action_distribution = gaussion_fn(actor_feautre)
        state_value = critic_feature
        network_io = {
            'input_state_tensor': input_state_tensor,
            'input_goal_tensor': input_goal_tensor,
            'action_distribution': action_distribution,
            'state_value': state_value}
    return network_io

def clipped_objective_loss(action_distribution, epsilon=0.2):
    advantage = tf.placeholder(shape=(None, 1), dtype=tf.float32)
    action = tf.placeholder(shape=(None, None), dtype=tf.float32)
    old_prob = tf.placeholder(shape=(None, 1), dtype=tf.float32)
    prob = gaussion_prob(action_distribution, action)
    r = prob/old_prob
    r_clipped = tf.clip_by_value(r, 1 - epsilon, 1 + epsilon)
    loss = -tf.minimum(r*advantage, r_clipped*advantage)
    return old_prob, action, advantage, tf.reduce_mean(loss)

def value_loss(state_value):
    estimated_state_value = tf.placeholder(shape=(None, 1), dtype=tf.float32)
    loss = tf.losses.mean_squared_error(estimated_state_value, state_value,
        reduction=tf.losses.Reduction.MEAN)
    return estimated_state_value, loss

def TD0_estimation(samples, gamma, advantage_offset):
    advantage_list = []
    estimated_state_value_list = []
    for i in range(len(samples)-1):
        state, goal, np_action, reward, done, np_action_prob, np_state_value = samples[i]
        assert not done, ('impossible to finish at %d local_step' % i)
        _, _, _, _, _, _, np_state_value_next = samples[i+1]
        estimated_state_value = reward + gamma * np_state_value_next
        advantage = estimated_state_value - np_state_value + advantage_offset
        estimated_state_value_list.append(estimated_state_value)
        advantage_list.append(advantage)
    state, goal, np_action, reward, done, np_action_prob, np_state_value = samples[-1]
    assert done, ('need to increase Max step to gaurrentee finish')
    estimated_state_value = reward
    advantage = estimated_state_value - np_state_value + advantage_offset
    estimated_state_value_list.append(estimated_state_value)
    advantage_list.append(advantage)
    return advantage_list, estimated_state_value_list

def kuka_env_init(renders=False, num_classes=9):
    item_path = '/home/kshitij/Desktop/RL/kuka_robotic_arm_RL/items'
    env = KukaVariedObjectEnv(item_path, isDiscrete=False, renders=renders, maxSteps = 10000000)
    return env

def kuka_env_reset(env, num_classes=9):
    env.reset()
    init_state = extend_feature_vec_observation(
        env.get_feature_vec_observation(), num_classes=num_classes)
    return init_state

def kuka_vec_sample(env, action, num_classes=9):
    """Sample kuka environment and observe vector space.
    Args:
        env: openAI gym kuka environment.
        action: an action vector.
    returns: a tuple (state, next_state, reward, done)"""
    _, reward, done, info = env.step(action)
    reward = 100.0*reward - 1
    next_state = extend_feature_vec_observation(
        env.get_feature_vec_observation(), num_classes=num_classes)
    return (reward, next_state, done)

def kuka_env_reset_separated_goal(env, num_classes=9):
    env.reset()
    init_state = extend_feature_vec_observation(
        env.get_feature_vec_observation(), num_classes=num_classes)
    init_state, init_goal = separate_state_goal(init_state)
    return init_state, init_goal

def kuka_vec_sample_separated_goal(env, action, num_classes=9):
    """Sample kuka environment and observe vector space.
    Args:
        env: openAI gym kuka environment.
        action: an action vector.
    returns: a tuple (state, next_state, reward, done)"""
    _, reward, done, info = env.step(action)
    reward = 100.0*reward - 1
    next_state = extend_feature_vec_observation(
        env.get_feature_vec_observation(), num_classes=num_classes)
    next_state, next_goal = separate_state_goal(next_state)
    return (reward, next_state, next_goal, done)

def encode_block_state_with_gripper_state(gripperState, blockState):
    gripperPos = gripperState[0]
    gripperOrn = gripperState[1]
    blockPos = blockState[0]
    blockOrn = blockState[1]
    invGripperPos,invGripperOrn = p.invertTransform(gripperPos,gripperOrn)
    blockPosInGripper,blockOrnInGripper = p.multiplyTransforms(invGripperPos,invGripperOrn,blockPos,blockOrn)
    blockEulerInGripper = p.getEulerFromQuaternion(blockOrnInGripper)
    #we return the relative x,y position and euler angle of block in gripper space
    blockInGripperPosXYEulZ =[blockPosInGripper[0],blockPosInGripper[1],blockEulerInGripper[2]]
    return list(blockInGripperPosXYEulZ)

def get_gripper_state(env):
    gripperState  = p.getLinkState(env._kuka.kukaUid, env._kuka.kukaGripperIndex)
    return gripperState

def get_block_state(env):
    blockPos,blockOrn = p.getBasePositionAndOrientation(env.blockUid)
    return (blockPos, blockOrn)

def fake_goal_from_gripper_state(gripperState, finalGripperState):
    finalgripperPos = list(finalGripperState[0])
    finalgripperOrn = list(finalGripperState[1])
    # change Pos, I got those number from my crystal ball
    finalgripperPos[2] = -0.164 + 0.01*np.random.normal()
    fake_block_state = (finalgripperPos, finalgripperOrn)
    encoded_block_state = encode_block_state_with_gripper_state(gripperState, fake_block_state)
    return encoded_block_state

class PPO(object):
    def __init__(self, train_config, model_config, num_actors,
        gamma=1.0, lam=1.0, loss_weights=(1.0, 1.0, 0.001), state_size=6,
        goal_size=12, max_steps=40,
        env_init_fn=kuka_env_init, env_reset_fn=kuka_env_reset,
        env_sample_fn=kuka_vec_sample, memory_size=50000, alt_success_rate=0.01, advantage_offset=0.0):
        # store environment functions
        self.gamma = gamma
        self.lam = lam
        self.env_init_fn = env_init_fn
        self.env_reset_fn = env_reset_fn
        self.env_sample_fn = env_sample_fn
        self.max_steps = max_steps
        self.num_actors = num_actors
        self.model_config = model_config
        self.state_size = state_size
        self.goal_size = goal_size
        self.alt_success_rate = alt_success_rate
        self.advantage_offset = advantage_offset
        workers = ["localhost:"+str(i+3333) for i in range(num_actors)]
        print('Create works'+str(workers))
        self.cluster = tf.train.ClusterSpec({
            "worker": workers,
            "ps": ["localhost:3332"]
        })
        # setup queues to communicate with actors
        self.sample_queue = Queue()
        self.command_queue = Queue()
        self.ret_queue = Queue()
        self.sucess_queue = Queue()
        self.processes = []
        print('start actors')
        self.start_actors()
        print('setup two policy networks on parameter_server')
        with tf.device("/job:ps/task:0"):
            with tf.variable_scope('old'):
                network_io = policy_fn(self.state_size, self.goal_size, self.model_config)
                self.input_state_tensor_old = network_io['input_state_tensor']
                self.input_goal_tensor_old = network_io['input_goal_tensor']
                self.action_distribution_old = network_io['action_distribution']
                self.state_value_old = network_io['state_value']
            with tf.variable_scope('new') as s:
                network_io = policy_fn(self.state_size, self.goal_size, self.model_config)
                self.input_state_tensor_new = network_io['input_state_tensor']
                self.input_goal_tensor_new = network_io['input_goal_tensor']
                self.action_distribution_new = network_io['action_distribution']
                self.state_value_new = network_io['state_value']
            # setup network copy function
            Ws_new = tf.trainable_variables(scope='new')
            Ws_old = tf.trainable_variables(scope='old')
            self.update_w_old = [
                tf.assign(w_old, w_new) for w_old, w_new in zip(Ws_old, Ws_new)]
            # add loss for new network
            (self.input_old_prob,
            self.input_old_action,
            self.input_estimated_advantage,
            self.objective_loss) = clipped_objective_loss(self.action_distribution_new)
            # add value_loss for new network
            (self.input_estimated_state_value,
            self.state_value_loss) = value_loss(self.state_value_new)
            # add entropy bonous for newtwork
            self.entropy_loss = -1.0*gaussion_entropy(self.action_distribution_new)
            # sum up losses
            self.objective_loss = loss_weights[0]*self.objective_loss
            self.state_value_loss = loss_weights[1]*self.state_value_loss
            self.entropy_loss = loss_weights[2]*self.entropy_loss
            self.reg_loss = tf.reduce_sum(tf.losses.get_regularization_losses('new'))
            self.loss = self.objective_loss+self.state_value_loss+self.entropy_loss+self.reg_loss
            # train ops
            # setup training ops
            self.global_step = tf.train.get_or_create_global_step()
            self.learning_rate = tf.train.exponential_decay(train_config['initial_lr'],
                self.global_step, train_config['decay_step'],
                train_config['decay_factor'], staircase=True)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = tf.train.AdamOptimizer(
                    learning_rate=self.learning_rate).minimize(self.loss,
                    global_step=self.global_step)
        print('setup server Session')
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
        server = tf.train.Server(self.cluster,
                                 job_name="ps",
                                 task_index=0,
                                 config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess = tf.Session(target=server.target)
        print('initialization')
        self.sess.run(tf.global_variables_initializer())
        # setup buffer for samples
        self.memory = Memory(memory_size)

    def train(self, epoch_num, num_trials, batch_size):
        """Training iteration."""
        # sampling with old policy
        self.memory.clear()
        print('Memory clear %d' % len(self.memory._buffer))
        sucess_ctr = self.run_actors(num_trials)
        print('sucess rate %f' % (float(sucess_ctr)/num_trials))
        print('lr='+str(self.sess.run(self.learning_rate)))
        # train new policy
        for epoch_id in range(epoch_num):
            batch = self.memory.sample_batch(batch_size)
            (b_np_state, b_np_goal, b_np_action, b_np_action_prob, \
            b_np_advantage, b_np_estimated_state_value) = zip(*batch)
            b_np_state = np.vstack(b_np_state)
            b_np_goal = np.vstack(b_np_goal)
            b_np_action = np.vstack(b_np_action)
            b_np_action_prob = np.vstack(b_np_action_prob)
            b_np_advantage = np.vstack(b_np_advantage)
            b_np_estimated_state_value = np.vstack(b_np_estimated_state_value)
            results = self.sess.run([self.train_op, self.loss, self.entropy_loss,
                self.objective_loss, self.state_value_loss, self.reg_loss],
                feed_dict={
                self.input_state_tensor_new: b_np_state,
                self.input_goal_tensor_new: b_np_goal,
                self.input_old_prob: b_np_action_prob,
                self.input_old_action: b_np_action,
                self.input_estimated_advantage: b_np_advantage,
                self.input_estimated_state_value: b_np_estimated_state_value
            })
            if (epoch_id % (int(epoch_num/10.0))) == 0:
                print(('epoch_id %d:'% epoch_id)+str(results))
        # update old policy
        self.sess.run(self.update_w_old)
        return sucess_ctr

    def test_once(self, render=False):
        env = self.env_init_fn(render=render)
        # wait for command forever
        local_step = 0
        sucess = 0
        # print "Start Sampling Process  " + str(worker_n)
        init_state, init_goal = self.env_reset_fn(env)
        state = init_state
        goal = init_goal
        while(local_step < self.max_steps):
            np_action_mean, np_action_log_std, np_state_value = sess.run(
                self.action_distribution_old+[self.state_value_old],
                feed_dict={self.input_state_tensor_old: [state],
                           self.input_goal_tensor_old: [goal]})
            np_action = np_action_mean[0] + np.exp(np_action_log_std[0])*np.random.normal(size=(3))
            transition = self.env_sample_fn(env, np_action)
            (reward, next_state, next_goal, done) = transition
            if reward > 0:
                sucess = 1
            np_action_prob = np_gaussion_prob((np_action_mean, np_action_log_std), np_action)
            local_step += 1
            state = next_state
            goal = next_goal
            if done:
                break
        env.close()
        return sucess

    def save_model(self, checkpoint_path, step):
        saver = tf.train.Saver()
        saver.save(self.sess, checkpoint_path, global_step=step)

    def load_model(self, checkpoint_path):
        print('Restore from checkpoint %s' % checkpoint_path)
        saver = tf.train.Saver()
        saver.restore(self.sess, checkpoint_path)

    def start_actors(self):
        def worker(worker_n, sample_queue, command_queue, ret_queue, sucess_queue):
            with tf.device(tf.train.replica_device_setter(cluster=self.cluster)):
                # worker only need old network to act
                with tf.variable_scope('old'):
                    network_io = policy_fn(self.state_size, self.goal_size, self.model_config)
                    input_state_tensor_old = network_io['input_state_tensor']
                    input_goal_tensor_old = network_io['input_goal_tensor']
                    action_distribution_old = network_io['action_distribution']
                    state_value_old = network_io['state_value']
            gpu_options = tf.GPUOptions(allow_growth=True)
            server = tf.train.Server(self.cluster,
                                     job_name="worker",
                                     task_index=worker_n,
                                     config=tf.ConfigProto(gpu_options=gpu_options))
            with tf.Session(target=server.target) as sess:
                # wait for the connection
                print("Worker %d: waiting for cluster connection..." % worker_n)
                while sess.run(tf.report_uninitialized_variables()).size != 0:
                    print("Worker %d: waiting for variable initialization..." % worker_n)
                    time.sleep(1.0)
                print("Worker %d: variables initialized" % worker_n)
                env = self.env_init_fn()
                # wait for command forever
                while True:
                    command = command_queue.get(True)
                    if command=='sample':
                        local_step = 0
                        sucess = 0
                        samples = []
                        # print "Start Sampling Process  " + str(worker_n)
                        init_state, init_goal = self.env_reset_fn(env)
                        state = init_state
                        goal = init_goal
                        while(local_step < self.max_steps):
                            np_action_mean, np_action_log_std, np_state_value = sess.run(
                                action_distribution_old+[state_value_old],
                                feed_dict={input_state_tensor_old: [state],
                                           input_goal_tensor_old: [goal]})
                            np_action = np_action_mean[0] + np.exp(np_action_log_std[0])*np.random.normal(size=(3))
                            transition = self.env_sample_fn(env, np_action)
                            (reward, next_state, next_goal, done) = transition
                            if reward > 0:
                                sucess = 1
                            np_action_prob = np_gaussion_prob((np_action_mean, np_action_log_std), np_action)
                            sample = (state, goal, np_action, reward, done, np_action_prob, np_state_value)
                            samples.append(sample)
                            local_step += 1
                            state = next_state
                            goal = next_goal
                            if done:
                                break
                        # ==============================================================================
                        # All TD(0)
                        advantage_list, estimated_state_value_list = TD0_estimation(samples, self.gamma, self.advantage_offset)
                        # ===============================================================================
                        # samples with advantage and target state_value
                        full_samples = []
                        for i in range(len(samples)):
                            state, goal, np_action, reward, done, np_action_prob, np_state_value = samples[i]
                            full_samples.append((state, goal, np_action, np_action_prob, advantage_list[i], estimated_state_value_list[i]))
                        # ===============================================================================
                        # send results
                        for sample in full_samples:
                            sample_queue.put(sample, True)
                        ret_queue.put(len(full_samples), True)
                        sucess_queue.put(sucess, True)
                    else:
                        if command=='sample-alt':
                            local_step = 0
                            sucess = 0
                            samples = []
                            gripper_states = []
                            # print "Start Sampling Process  " + str(worker_n)
                            init_state, init_goal = self.env_reset_fn(env)
                            state = init_state
                            goal = init_goal
                            while(local_step < self.max_steps):
                                np_action_mean, np_action_log_std, np_state_value = sess.run(
                                    action_distribution_old+[state_value_old],
                                    feed_dict={input_state_tensor_old: [state],
                                               input_goal_tensor_old: [goal]})
                                np_action = np_action_mean[0] + np.exp(np_action_log_std[0])*np.random.normal(size=(3))
                                # save current gripper state
                                gripper_states.append(get_gripper_state(env))
                                transition = self.env_sample_fn(env, np_action)
                                (reward, next_state, next_goal, done) = transition
                                if reward > 0:
                                    sucess = 1
                                np_action_prob = np_gaussion_prob((np_action_mean, np_action_log_std), np_action)
                                sample = (state, goal, np_action, reward, done, np_action_prob, np_state_value)
                                samples.append(sample)
                                local_step += 1
                                state = next_state
                                goal = next_goal
                                if done:
                                    break
                            # ==============================================================================
                            # All TD(0)
                            advantage_list, estimated_state_value_list = TD0_estimation(samples, self.gamma, self.advantage_offset)
                            # ===============================================================================
                            # samples with advantage and target state_value
                            full_samples = []
                            for i in range(len(samples)):
                                state, goal, np_action, reward, done, np_action_prob, np_state_value = samples[i]
                                full_samples.append((state, goal, np_action, np_action_prob, advantage_list[i], estimated_state_value_list[i]))

                            # ==================================================
                            # Generate alternative goals for fail cases
                            if sucess < 0.5 and np.random.random() < self.alt_success_rate:
                                samples_alt = []
                                is_impossible = False
                                for i in range(len(samples)):
                                    state, goal, np_action, reward, done, np_action_prob, np_state_value = samples[i]
                                    alt_goal = fake_goal_from_gripper_state(gripper_states[i], gripper_states[-1])
                                    alt_goal = np.hstack([alt_goal, goal[3:]])
                                    alt_np_action_mean, alt_np_action_log_std, alt_np_state_value = sess.run(
                                        action_distribution_old+[state_value_old],
                                        feed_dict={input_state_tensor_old: [state],
                                                   input_goal_tensor_old: [alt_goal]})
                                    alt_np_action_prob = np_gaussion_prob((alt_np_action_mean, alt_np_action_log_std), np_action)
                                    if alt_np_action_prob < 1e-2:
                                        is_impossible = True
                                        break
                                    alt_reward = -1
                                    if i == (len(samples)-1):
                                        alt_reward = 99
                                    sample_alt = (state, alt_goal, np_action, alt_reward, done, alt_np_action_prob, alt_np_state_value)
                                    samples_alt.append(sample_alt)
                                # All TD(0)
                                if not is_impossible:
                                    alt_advantage_list, alt_estimated_state_value_list = TD0_estimation(samples_alt, self.gamma, self.advantage_offset)
                                # samples with advantage and target state_value
                                    for i in range(len(samples_alt)):
                                        alt_state, alt_goal, alt_np_action, alt_reward, alt_done, alt_np_action_prob, alt_np_state_value = samples_alt[i]
                                        full_samples.append((alt_state, alt_goal, alt_np_action, alt_np_action_prob, alt_advantage_list[i], alt_estimated_state_value_list[i]))
                            # ===============================================================================
                            # send results
                            for sample in full_samples:
                                sample_queue.put(sample, True)
                            ret_queue.put(len(full_samples), True)
                            sucess_queue.put(sucess, True)
                        else:
                            if command=='end':
                                break
            # clean up
            env.close()
            print('Sampling Actor %d ends' % worker_n)
            return 0
        # start workers
        for worker_n in range(self.num_actors):
            process = Process(target=worker, args=(worker_n, self.sample_queue, self.command_queue, self.ret_queue, self.sucess_queue))
            process.daemon = True
            process.start()
            self.processes.append(process)

    def finish_actors(self):
        for _ in range(self.num_actors):
            self.command_queue.put('end', True)
        for pi, p in enumerate(self.processes):
            p.terminate()
            p.join()
            print('processes %d closed' % pi)

    def run_actors(self, num_trials):
        sample_ctr = 0
        sucess_ctr = 0
        for _ in range(num_trials):
            self.command_queue.put('sample-alt', True)
        for i in range(num_trials):
            sample_ctr += self.ret_queue.get(True)
        for i in range(num_trials):
            sucess_ctr += self.sucess_queue.get(True)
        for i in range(sample_ctr):
            self.memory.append(self.sample_queue.get(True))
        print('%d trials add %d new transitions' % (num_trials, len(self.memory._buffer)))
        return sucess_ctr

    def __del__(self):
        self.finish_actors()
        self.sess.close()

model_config = [(1024,), (1024,), (1024,), (1024,), (1024, 1024), (3, 1)]
train_config = {'initial_lr': 0.0016, 'decay_step': 1000, 'decay_factor': 0.5}
ppo = PPO(train_config, model_config, gamma=0.5, num_actors=8,
    loss_weights=(10.0, 1.0, 1.0), state_size=6, goal_size=12, max_steps=40,
        env_init_fn=kuka_env_init,
        env_reset_fn=kuka_env_reset_separated_goal,
        env_sample_fn=kuka_vec_sample_separated_goal,
        memory_size=500000,
        advantage_offset=0.0,
        alt_success_rate=0.0
        )

epoch_num = 10
num_trials = 800
batch_size = 4096
num_train_step = 1000
train_dir = './ppo_her_000_ckpt_offset_0_faster/'
reward_curve = []
for train_step in range(num_train_step):
    start_time = time.time()
    print('train_step %d'%train_step)
    sucess_ctr = ppo.train(epoch_num, num_trials, batch_size)
    if(train_step%10==0):
        print('save at step %d' % train_step)
        ppo.save_model(train_dir+'model', step=train_step)
    print('@train_step: %d, sucess %d out of %d trials'
        % (train_step, sucess_ctr, num_trials))
    reward_curve.append((train_step, num_trials, sucess_ctr))
    with open(train_dir+'train_curve.json', 'w') as f:
        json.dump(reward_curve, f)
    print('time cost %f'%(time.time() - start_time))
