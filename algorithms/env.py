from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
from gym import spaces
import numpy as np
import pybullet as p
import os
import glob

class KukaEnv(KukaDiverseObjectEnv):
    def __init__(self, env_params):
        super().__init__(actionRepeat=env_params.actionRepeat, 
                        isEnableSelfCollision=env_params.isEnableSelfCollision, 
                        renders=env_params.renders, isDiscrete=env_params.isDiscrete, 
                        maxSteps=env_params.maxSteps, dv=env_params.dv, 
                        removeHeightHack=env_params.removeHeightHack, 
                        blockRandom=env_params.blockRandom, cameraRandom=env_params.cameraRandom, 
                        width=env_params.width, height=env_params.height, numObjects=env_params.numObjects)
        self.ImageAsState = env_params.ImageAsState
        if self.ImageAsState:
            self.observation_space = spaces.Box(low=0, high=255, shape=(env_params.height, env_params.width, 3), dtype=np.uint8)
        else:
            # This part obtained from kukaGymEnv.py
            self.reset()
            largeValObservation = 100
            observationDim = len(self.getExtendedObservation())
            observation_high = np.array([largeValObservation] * observationDim)
            self.observation_space = spaces.Box(-observation_high, observation_high)

    def getExtendedObservation(self):
        """ KukaGymEnv has hardcoded getExtendedObservation function to use `self.blockUid` to obtain
        position and orientation of object for feature vector. This function returns position and orientation 
        for all objects in the env (and gripper position & orientation)
        """
        if not hasattr(self, "_objectUids"):
            super().reset()
        feature_vec_for_all_blockUids = None
        for blockUid in self._objectUids:
            self.blockUid = blockUid
            feature_vec_for_blockUid = super().getExtendedObservation()
            if feature_vec_for_all_blockUids:
                # Only add relative x,y position and euler angle of block in gripper space to the feature vector
                feature_vec_for_all_blockUids.extend(feature_vec_for_blockUid[-3:])
            else:
                feature_vec_for_all_blockUids = feature_vec_for_blockUid
        return feature_vec_for_all_blockUids
    
    def reset(self):
        img = super().reset()
        if self.ImageAsState:
            return img
        else:
            return self.getExtendedObservation()

    def step(self, action):
        img_observation, reward, done, debug = super().step(action)
        if self.ImageAsState:
            return img_observation, reward, done, debug
        else:
            return self.getExtendedObservation(), reward, done, debug
    
    # def _reward(self):
    #     """Calculates the reward for the episode.

    #     The reward is 1 if one of the objects is above height .2 at the end of the
    #     episode.
    #     """
    #     reward = 0
    #     self._graspSuccess = 0
    #     for uid in self._objectUids:
    #         pos, _ = p.getBasePositionAndOrientation(uid)
    #         reward = 2+pos[2]
    #         # If any block is above height, provide reward.
    #         if pos[2] > 0.2:
    #             self._graspSuccess += 1
    #             reward += 10
    #             break
    #     return reward

    def _get_random_object(self, num_objects, test):
        """Randomly choose an object urdf from the random_urdfs directory.

        Args:
        num_objects:
            Number of graspable objects.

        Returns:
        A list of urdf filenames.
        """
        if test:
            urdf_pattern = os.path.join(self._urdfRoot, 'random_urdfs/*0/*.urdf')
        else:
            urdf_pattern = os.path.join(self._urdfRoot, 'random_urdfs/*[1-9]/*.urdf')
        found_object_directories = glob.glob(urdf_pattern)
        total_num_objects = len(found_object_directories)
        selected_objects = [10]#np.random.choice(np.arange(total_num_objects), num_objects)
        selected_objects_filenames = []
        for object_index in selected_objects:
            selected_objects_filenames += [found_object_directories[object_index]]
        return selected_objects_filenames


# class KukaVariedObjectEnv(KukaDiverseObjectEnv):
#     """Class for Kuka environment or 10703 project.
#     In each episode an object is chosen from a set of specific objects.
#     """

#     def __init__(self,
#                  urdfRoot2,
#                  urdfRoot=pybullet_data.getDataPath(),
#                  actionRepeat=80,
#                  isEnableSelfCollision=True,
#                  renders=False,
#                  isDiscrete=False,
#                  maxSteps=8,
#                  dv=0.06,
#                  removeHeightHack=False,
#                  blockRandom=0.3,
#                  cameraRandom=0,
#                  width=48,
#                  height=48):
#         """Initializes the KukaVariedObjectEnv.

#         Args:
#             urdfRoot2: The directory from which to load item URDFs.
#             urdfRoot: The directory from which to load environment URDFs.
#             actionRepeat: The number of simulation steps to apply for each action.
#             isEnableSelfCollision: If true, enable self-collision.
#             renders: If true, render the bullet GUI.
#             isDiscrete: If true, the action space is discrete. If False, the
#                 action space is continuous.
#             maxSteps: The maximum number of actions per episode.
#             dv: The velocity along each dimension for each action.
#             removeHeightHack: If false, there is a "height hack" where the gripper
#                 automatically moves down for each action. If true, the environment is
#                 harder and the policy chooses the height displacement.
#             blockRandom: A float between 0 and 1 indicated block randomness. 0 is
#                 deterministic.
#             cameraRandom: A float between 0 and 1 indicating camera placement
#                 randomness. 0 is deterministic.
#             width: The image width.
#             height: The observation image height.
#         """
#         super(KukaVariedObjectEnv, self).__init__(urdfRoot, actionRepeat, isEnableSelfCollision, renders, isDiscrete, maxSteps, dv, removeHeightHack, blockRandom, cameraRandom, width, height, 1, False)
#         self._urdfRoot2 = urdfRoot2
#         self.blockUid = None

#     def get_feature_vec_observation(self):
#         return self.getExtendedObservation() + [self.cur_file]

#     def _get_random_object(self, num_objects, test):
#         """Randomly choose an object urdf from the random_urdfs directory.

#         Args:
#             num_objects:
#                 Number of graspable objects (used in parent class; ignored here and 1 is used instead)

#         Returns:
#             A list of urdf filenames.
#         """
#         if test:
#             urdf_pattern = os.path.join(self._urdfRoot2, '*.urdf')
#         else:
#             urdf_pattern = os.path.join(self._urdfRoot2, '*.urdf')
#         found_object_directories = glob.glob(urdf_pattern)
#         total_num_objects = len(found_object_directories)
#         selected_object = np.random.choice(total_num_objects)
#         sof = found_object_directories[selected_object]

#         fname = os.path.split(sof)[1]
#         self.cur_file = int(fname[0 : fname.rfind('.')])
#         return [sof]

#     def _reset(self):
#         temp = super(KukaVariedObjectEnv, self).reset()
#         self.blockUid = self._objectUids[0]
#         return temp

#     def _step(self, action):
#         return super(KukaVariedObjectEnv, self).step(action)

#     def get_state(self):
#         n_object = 9
#         state = self.get_feature_vec_observation()
#         res = state[0:-1]
#         res.extend(to_categorical(state[-1], n_object))
#         return res

#     def get_block_in_gripper_pos(self, gripperPos, gripperOrn, blockPos, blockOrn):  # (x,x,x) (x,x,x,x)
#         invGripperPos,invGripperOrn = p.invertTransform(gripperPos, gripperOrn)

#         blockPosInGripper,blockOrnInGripper = p.multiplyTransforms(invGripperPos, invGripperOrn, blockPos, blockOrn)
#         blockEulerInGripper = p.getEulerFromQuaternion(blockOrnInGripper)

#         #we return the relative x,y position and euler angle of block in gripper space
#         blockInGripperPosXYEulZ =[blockPosInGripper[0], blockPosInGripper[1], blockEulerInGripper[2]]

#         return list(blockInGripperPosXYEulZ)

#     if parse_version(gym.__version__)>=parse_version('0.9.6'):

#         reset = _reset

#         step = _step
"""
```
position, orientation = p.getLinkState(env._kuka.kukaUid, env._kuka.kukaGripperIndex)
euler = p.getEulerFromQuaternion(orientation)
```
position: 3 floats indicating position of object/kuka gripper
orientation: 4 floats indicating orientation in Quaternion format (x,y,z,w)
euler: 3 floats indicating orientation in Euler format (Pitch, Roll, Yaw)
"""
if __name__ == '__main__':
    from algorithms.utils import parse_config

    base_algo_dir = "/".join(__file__.split("/")[:-1])
    config_params = parse_config(os.path.join(base_algo_dir, 'debug/env_config.yml'))
    env = KukaEnv(config_params.environment)

    # Allow user to use specify values with slides in UI
    user_params = []
    dv = 0.01
    for param in ['location_X', 'location_Y', 'location_Z', 'Yaw']:
        user_params.append(env._p.addUserDebugParameter(param,-dv,dv,0))
    user_params.append(env._p.addUserDebugParameter("Joint_angle",0,0.3,0.3))

    # Simulate robotic arm with actions from UI
    while True:
        state = env.reset()
        done = False
        while not done:
            action=[env._p.readUserDebugParameter(param) for param in user_params]
            state, reward, done, info = env.step(action)
