""" Script implementing DQN for Gym environment """
import argparse, gym, os, time, logging, random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

plt.style.use("dark_background")
from algorithms.DQN.src.network import Net, train_step, update_target_model
from collections import deque
from tqdm import tqdm

from algorithms.utils import parse_config
from algorithms.env import KukaEnv
np.random.seed(42)


class DQN:
    def __init__(self, env, parameters, logger=None):
        """ DQN agent instance

        Parameters
        ----------
        env : gym environment instance
            Environment to be used for training or inference
        parameters : argparse.namespace
            CLI arguments provided as input through argparse
        logger : logging.Logger, optional
            logging instance to be used for writing logs to 
            file and stdout, by default None

        Raises
        ------
        ValueError
            If continuous actions environment is provided
        """
        # Gym environment parameters
        self.env_name = parameters.environment_name
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        if isinstance(self.env.action_space, gym.spaces.discrete.Discrete):
            self.num_actions = self.env.action_space.n
        else:
            raise ValueError("Continuous actions not supported")
        self.render_decision = parameters.render_decision

        if logger is not None:
            self.logger = logger
            self.print_fn = self.logger.info
        else:
            self.print_fn = print
        self.QNet = Net(
            self.state_dim,
            self.num_actions,
            NetworkParameters.QNet,
            "QNet",
            duel=TrainingParameters.duel,
        )
        self.test_only = parameters.test_decision
        if not self.test_only:
            # Training parameters
            self.train_episodes = parameters.num_episodes
            self.test_episodes = TrainingParameters.test_episodes
            self.test_frequency = TrainingParameters.test_frequency
            self.render_frequency = TrainingParameters.render_frequency
            self.video_save_frequency = TrainingParameters.video_save_frequency
            self.model_save_frequency = TrainingParameters.model_save_frequency
            self.discount = TrainingParameters.discount
            self.starting_episode = 0
            # Experience replay memory
            self.memory = deque(maxlen=TrainingParameters.Replay_memory.capacity)
            self.burn_memory(TrainingParameters.Replay_memory.burn_episodes)
            # Target Q network for better convergence
            self.target_Q_net = Net(
                self.state_dim,
                self.num_actions,
                NetworkParameters.QNet,
                "Target_QNet",
                duel=TrainingParameters.duel,
            )
            self.update_target_frequency = TrainingParameters.update_target_frequency
            self.double = TrainingParameters.double
        if parameters.model_path:
            self.QNet.load_weights(parameters.model_path)
            self.print_fn(
                f"Loading weights for {self.QNet.name} from {parameters.model_path}"
            )
            self.starting_episode = int(parameters.model_path[-5:])
            if not self.test_only:
                self.target_Q_net.load_weights(parameters.model_path)

    def burn_memory(self, num_burn_episodes):
        """ Burn replay memory with random action episodes

        Parameters
        ----------
        num_burn_episodes : int
            Number of random action episodes to be used
        """
        self.print_fn(f"Burning replay memory with {num_burn_episodes} episodes")
        for _ in range(num_burn_episodes):
            state = self.env.reset()
            done = False
            while not done:
                random_action = self.env.action_space.sample()
                next_state, reward, done, _ = self.env.step(random_action)
                self.memory.append((next_state, reward, state, random_action, ~done))
                state = next_state

    def epsilon_greedy_policy(self, q_values, epsilon=0.05):
        """ Returns action to be executed by agent as per epsilon-greedy policy

        Parameters
        ----------
        q_values : list
            Q values for all actions in the current state of agent
        epsilon : float
            Probability of random action (for exploration)

        Returns
        -------
        action : int
            Action as per epsilon-greedy policy in the current state of agent
        """
        if np.random.rand() < epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(q_values)

    def train(self):
        """ Trains agent with DQN / Double DQN / Dueling DQN algorithm

        TODOs
        -------
        - Model training step involves predicting Q-values and updating values for selected actions in batch as targets
        so that only those loss is computed only for those values. Later in train_step function Q-values are again predicted 
        from model to compute the loss. This can probably be refactored for better efficiency
        """
        tensorboard_file_writer = tf.summary.create_file_writer(os.path.join(Directories.output, "tensorboard"))
        if NetworkParameters.QNet.lr_scheduler.type is not None:
            scheduler_definition = getattr(tf.keras.optimizers.schedules, NetworkParameters.QNet.lr_scheduler.type)
            lr_schedule = scheduler_definition(float(NetworkParameters.QNet.learning_rate), **NetworkParameters.QNet.lr_scheduler.input_args)
        else:
            lr_schedule = float(NetworkParameters.QNet.learning_rate)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        mean_test_rewards, std_test_rewards = [], []
        self.print_fn(f"Training agent for {self.train_episodes} episodes in {self.env_name} environment")

        epsilon = TrainingParameters.Exploration.intial_epsilon
        anneal_eps = (epsilon - TrainingParameters.Exploration.final_epsilon) / TrainingParameters.Exploration.decay_steps

        for episode in tqdm(range(self.starting_episode, self.starting_episode + self.train_episodes + 1)):
            state = self.env.reset()
            done = False
            loss = 0.0
            while not done:
                # Perform an action in environment and add to replay memory
                q_values = self.QNet(tf.expand_dims(state, axis=0))
                action = int(self.epsilon_greedy_policy(q_values, epsilon))
                epsilon -= anneal_eps
                epsilon = max(epsilon, TrainingParameters.Exploration.final_epsilon)
                next_state, reward, done, _ = self.env.step(action)
                self.memory.append((next_state, reward, state, action, ~done))
                state = next_state

                # Sample batch from memory and train QNet model
                batch = random.sample(self.memory, TrainingParameters.batch_size)
                next_states, rewards, states, actions, if_not_terminals = map(np.array, zip(*batch))
                if self.double:
                    """
                    To avoid over-estimation of Q values, double DQN splits np.max in DQN into first step of
                    argmax over Q-values using Q_net and then use those indices to acquire Q-values of next state 
                    using target_Q_net
                    """
                    # Action is selected using QNet model instance (online weights)
                    q_next = self.QNet(next_states)
                    next_actions = np.argmax(q_next, axis=1)
                    next_actions_indices = np.vstack([np.arange(TrainingParameters.batch_size), next_actions]).T
                    # Value is estimated using target QNet model instance
                    target_q_next_all_actions = self.target_Q_net(next_states)
                    targets = rewards + if_not_terminals * self.discount * tf.gather_nd(target_q_next_all_actions, next_actions_indices)
                else:
                    # Use target_Q_net to generate targets for better stability
                    q_next = self.target_Q_net(next_states).numpy()
                    targets = rewards + if_not_terminals * self.discount * np.max(q_next, axis=1)
                # TODO: Optimize code by putting the target allocation in train_step
                q_values_for_targets = self.QNet(states).numpy()
                q_values_for_targets[np.arange(TrainingParameters.batch_size), actions] = targets
                loss += train_step(self.QNet, optimizer, states, q_values_for_targets)
################################################################################################################
            if loss > 1e10:
                print("Loss crossed over 1e10, exiting training loop")
                break
################################################################################################################
            with tensorboard_file_writer.as_default():
                tf.summary.scalar("Training_loss", loss, step=episode)
                tf.summary.scalar("Learning_rate", optimizer._decayed_lr(tf.float32).numpy(),step=episode)
                tf.summary.scalar("epsilon", epsilon, step=episode)
                # Test policy as per test frequency
                if episode % self.test_frequency == 0:
                    if episode % self.render_frequency == 0:
                        mean_test_reward, std_test_reward = self.execute_policy(
                            current_training_episode=episode,
                            test_episodes=self.test_episodes,
                            render=self.render_decision,
                            save_frequency=TrainingParameters.video_save_frequency,
                        )
                    else:
                        mean_test_reward, std_test_reward = self.execute_policy(test_episodes=self.test_episodes, render=False)
                    self.print_fn(f"After {episode} episodes, mean test reward is {mean_test_reward} with std of {std_test_reward} over {self.test_episodes} episodes")
                    mean_test_rewards.append(mean_test_reward)
                    std_test_rewards.append(std_test_reward)
                    tf.summary.scalar("Mean rewards over 100 episodes", mean_test_reward, step=episode)

            # Save model as per model_save_frequency
            if episode % self.model_save_frequency == 0:
                self.QNet.save_weights(os.path.join(Directories.output, f"saved_models/model_{episode:05}"))
            # Update target model as per update frequency
            if episode % self.update_target_frequency == 0:
                update_target_model(self.QNet, self.target_Q_net, smoothing=TrainingParameters.update_smoothing)

        self.QNet.save_weights(os.path.join(Directories.output, f"saved_models/model_{episode:05}"))
        plt.figure(figsize=(15, 10))
        plt.errorbar(x=range(self.starting_episode, episode + 1, self.test_frequency), y=mean_test_rewards, yerr=std_test_rewards)
        plt.xlabel("Number of episodes")
        plt.ylabel("Mean reward with std")
        plt.savefig(os.path.join(Directories.output, "Training_performance.png"))

    def execute_policy(
        self,
        current_training_episode=None,
        test_episodes=100,
        render=False,
        save_frequency=None,
    ):
        """ Run inference in environment using current policy and evaluate agent performance

        Parameters
        ----------
        current_training_episode : int, optional
            Current training episode number for saving progress video, by default None
        test_episodes : int, optional
            Number of test episodes to be simulated, by default 100
        render : bool, optional
            Render environment to stdout (or output window if render is implemented for environment), by default False
        save_frequency : int, optional
            specifies episode number out of test_episodes for which video is to be saved, if None, dont save any video,
            by default None

        Returns
        -------
        float
            Mean and std of rewards obtained by agent over number of test episodes specified
        """
        if render and save_frequency:
            def video_frequency(x):
                return x % save_frequency == 0
            if self.test_only:
                video_save_path = os.path.join(Directories.output, f"{self.env_name}_test_videos")
            else:
                video_save_path = os.path.join(Directories.output, f"{self.env_name}_Training_progress_videos/{current_training_episode}")
            self.env = gym.wrappers.Monitor(self.env, video_save_path, video_callable=video_frequency, force=True)
        rewards = np.zeros(test_episodes)
        for test_episode in range(test_episodes):
            curr_episode_reward, done = 0.0, False
            state = self.env.reset()
            while not done:
                if render:
                    self.env.render()
                # Reshape state to add batch dimension
                Q_values = self.QNet(tf.expand_dims(state, axis=0))
                action = int(np.argmax(Q_values))
                state, reward, done, _ = self.env.step(action)
                curr_episode_reward += reward
            rewards[test_episode] = curr_episode_reward
        return np.mean(rewards), np.std(rewards)


def parse_arguments():
    """ Returns command line arguments as per argparse

    Returns
    -------
    argparse arguments
        Parsed arguments from argparse object
    """
    parser = argparse.ArgumentParser(
        description="DQN algorithm implementation: This can be used to train an agent or test a saved model for policy"
    )
    parser.add_argument(
        "--t",
        dest="test_decision",
        action="store_true",
        help="Test the agent with saved model file",
    )
    parser.add_argument("--m", dest="model_path", default=None, help="Path to model to be used for test")
    parser.add_argument(
        "--e",
        dest="environment_name",
        default="CartPole-v0",
        type=str,
        help="Gym Environment",
    )
    parser.add_argument(
        "--r",
        dest="render_decision",
        action="store_true",
        help="Render the environment",
    )
    parser.add_argument(
        "--ep",
        dest="num_episodes",
        default=None,
        type=int,
        help="Number of episodes for training and if test flag is given, this can be used to specify number of test episodes",
    )
    parser.add_argument(
        "--c",
        dest="config",
        default="config.yml",
        type=str,
        help="Config file name",
    )
    return parser


def setup_logging(logger_file_path):
    """ Returns logging object which writes logs to a file and stdout

    Parameters
    ----------
    logger_file_path : str
        Path (with filename) to the log file

    Returns
    -------
    logging object
        Configured logging object for logging
    """
    # Create logging object
    logger = logging.getLogger(name=__name__)
    logger.setLevel(logging.DEBUG)
    logger.propagate = True
    # Save logs to file
    file_logging = logging.FileHandler(logger_file_path)
    fmt = logging.Formatter("%(asctime)s %(levelname)-8s: %(message)s")
    file_logging.setFormatter(fmt)
    file_logging.setLevel(logging.DEBUG)
    logger.addHandler(file_logging)
    # Print log to stdout
    stdout_logging = logging.StreamHandler()
    stdout_logging.setFormatter(fmt)
    logger.addHandler(stdout_logging)
    return logger


def run_agent(args, logger=None):
    """ Run agent training or inference

    Parameters
    ----------
    args : argparse.Namespace
        CLI arguments provided as input through argparse
    logger : logging.Logger, optional
        logging instance to be used for writing logs to 
        file and stdout, by default None
    """
    with KukaEnv(GymEnvironment) as env:
        agent = DQN(env, args, logger)
        if not args.test_decision:
            agent.train()
            test_episodes = agent.test_episodes
            mean_reward, std_reward = agent.execute_policy(
                test_episodes=test_episodes, render=args.render_decision
            )
        else:
            test_episodes = args.num_episodes
            mean_reward, std_reward = agent.execute_policy(
                test_episodes=test_episodes,
                render=args.render_decision,
                save_frequency=InferenceParameters.video_save_frequency,
            )
    agent.print_fn(f"The reward is {mean_reward} with standard deviation of {std_reward} over {test_episodes} episodes")

if __name__ == "__main__":

    TIMESTAMP = 'debug'#time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    args = parse_arguments().parse_args()

    # Set parameters from config file
    base_algo_dir = "/".join(__file__.split("/")[:-2])
    config_parameters = parse_config(os.path.join(base_algo_dir, "config", args.config))
    NetworkParameters = config_parameters.Network
    TrainingParameters = config_parameters.Training
    InferenceParameters = config_parameters.Inference
    GymEnvironment = config_parameters.environment
    for dir_ in config_parameters.Directories:
        config_parameters.Directories[dir_] = os.path.join(base_algo_dir, dir_)
    Directories = config_parameters.Directories

    if args.test_decision:
        if args.num_episodes is None:  # Set default test episodes to 100
            args.num_episodes = 100
        if args.model_path is None:
            raise FileNotFoundError("Model path missing for inference")
        # Use saved model path directory
        Directories.output = "/".join(args.model_path.split("/")[:-2])
        logger_file_path = os.path.join(Directories.output, "test.log")
    else:
        if args.num_episodes is None:  # Set default train episodes to 5000
            args.num_episodes = 5000
        if args.model_path:  # Continue training from pretrained model
            Directories.output = "/".join(args.model_path.split("/")[:-2])
        else:
            Directories.output = os.path.join(Directories.output, TIMESTAMP)
            os.makedirs(Directories.output, exist_ok=True)
        logger_file_path = os.path.join(Directories.output, "train.log")
    logger = setup_logging(logger_file_path)
    run_agent(args, logger)
