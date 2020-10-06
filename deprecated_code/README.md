# Reinforcement learning for Kuka Robotic arm manipulation task

Pybullet Kuka robotic arm manipulation task was analyzed for different reinforcement learning algorithms. Following approaches were considered for the task.

- DQN (discrete action space)
- A2C
- DDPG
- PPO

The PPO and DDPG implementation were augmented with Hindsight Experience Replay (HER). In addition to this, curiosity driven exploration was studied for DDPG implementation

## Results:

The simulated robotic arm was trained using DQN for discrete action space for the following result

![](https://github.com/kshitij-ingale/kuka_robotic_arm_RL/blob/master/video/final.gif)
