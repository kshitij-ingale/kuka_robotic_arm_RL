from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv
import os
import pybullet as p
import numpy as np
import pybullet_data
import glob
from pkg_resources import parse_version
import gym
from keras.utils import to_categorical
class KukaVariedObjectEnv(KukaDiverseObjectEnv):
    """Class for Kuka environment or 10703 project.
    In each episode an object is chosen from a set of specific objects.
    """

    def __init__(self,
                 urdfRoot2,
                 urdfRoot=pybullet_data.getDataPath(),
                 actionRepeat=80,
                 isEnableSelfCollision=True,
                 renders=False,
                 isDiscrete=False,
                 maxSteps=8,
                 dv=0.06,
                 removeHeightHack=False,
                 blockRandom=0.3,
                 cameraRandom=0,
                 width=48,
                 height=48):
        """Initializes the KukaVariedObjectEnv.

        Args:
            urdfRoot2: The directory from which to load item URDFs.
            urdfRoot: The directory from which to load environment URDFs.
            actionRepeat: The number of simulation steps to apply for each action.
            isEnableSelfCollision: If true, enable self-collision.
            renders: If true, render the bullet GUI.
            isDiscrete: If true, the action space is discrete. If False, the
                action space is continuous.
            maxSteps: The maximum number of actions per episode.
            dv: The velocity along each dimension for each action.
            removeHeightHack: If false, there is a "height hack" where the gripper
                automatically moves down for each action. If true, the environment is
                harder and the policy chooses the height displacement.
            blockRandom: A float between 0 and 1 indicated block randomness. 0 is
                deterministic.
            cameraRandom: A float between 0 and 1 indicating camera placement
                randomness. 0 is deterministic.
            width: The image width.
            height: The observation image height.
        """
        super(KukaVariedObjectEnv, self).__init__(urdfRoot, actionRepeat, isEnableSelfCollision, renders, isDiscrete, maxSteps, dv, removeHeightHack, blockRandom, cameraRandom, width, height, 1, False)
        self._urdfRoot2 = urdfRoot2
        self.blockUid = None

    def get_feature_vec_observation(self):
        return self.getExtendedObservation() + [self.cur_file]

    def _get_random_object(self, num_objects, test):
        """Randomly choose an object urdf from the random_urdfs directory.

        Args:
            num_objects:
                Number of graspable objects (used in parent class; ignored here and 1 is used instead)

        Returns:
            A list of urdf filenames.
        """
        if test:
            urdf_pattern = os.path.join(self._urdfRoot2, '*.urdf')
        else:
            urdf_pattern = os.path.join(self._urdfRoot2, '*.urdf')
        found_object_directories = glob.glob(urdf_pattern)
        total_num_objects = len(found_object_directories)
        selected_object = np.random.choice(total_num_objects)
        sof = found_object_directories[selected_object]

        fname = os.path.split(sof)[1]
        self.cur_file = int(fname[0 : fname.rfind('.')])
        return [sof]

    def _reset(self):
        temp = super(KukaVariedObjectEnv, self).reset()
        self.blockUid = self._objectUids[0]
        return temp

    def _step(self, action):
        return super(KukaVariedObjectEnv, self).step(action)

    def get_state(self):
        n_object = 9
        state = self.get_feature_vec_observation()
        res = state[0:-1]
        res.extend(to_categorical(state[-1], n_object))
        return res

    def get_block_in_gripper_pos(self, gripperPos, gripperOrn, blockPos, blockOrn):  # (x,x,x) (x,x,x,x)
        invGripperPos,invGripperOrn = p.invertTransform(gripperPos, gripperOrn)

        blockPosInGripper,blockOrnInGripper = p.multiplyTransforms(invGripperPos, invGripperOrn, blockPos, blockOrn)
        blockEulerInGripper = p.getEulerFromQuaternion(blockOrnInGripper)

        #we return the relative x,y position and euler angle of block in gripper space
        blockInGripperPosXYEulZ =[blockPosInGripper[0], blockPosInGripper[1], blockEulerInGripper[2]]

        return list(blockInGripperPosXYEulZ)

    if parse_version(gym.__version__)>=parse_version('0.9.6'):

        reset = _reset

        step = _step