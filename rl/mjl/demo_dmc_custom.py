import collections

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers
from dm_control import viewer
from dm_control import suite
from dm_control.utils import io as resources
import numpy as np
import os

_DEFAULT_TIME_LIMIT = 30
_CONTROL_TIMESTEP = 0.04

SUITE = containers.TaggedTasks()
THIS_DIR = os.path.dirname(__file__)
# MODEL_XML_PATH = '/home/jintian/mujoco/model/humanoid/humanoid.xml'
# MODEL_XML_PATH = os.path.join(THIS_DIR, './models/humanoid_h36m/humanoid_h36m_v5.xml') 
MODEL_XML_PATH = 'models/mug/mug.xml'

def get_model_and_assets():
    """Returns a tuple containing the model XML string and a dict of assets."""
    # return common.read_model(MODEL_XML_PATH), common.ASSETS
    # return resources.GetResource(MODEL_XML_PATH), common.ASSETS
    return resources.GetResource(MODEL_XML_PATH), {}


@SUITE.add("playing")
def my_env(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Test task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = Robot()
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics,
        task,
        time_limit=time_limit,
        control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs
    )


class Physics(mujoco.Physics):
    """Physics class"""


class Robot(base.Task):
    def __init__(self, random=None):
        """Initializes an instance of `Robot`."""
        super(Robot, self).__init__(random=random)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        pass

    def get_observation(self, physics):
        """Returns either the pure state or a set of egocentric features."""
        obs = collections.OrderedDict()
        return obs

    def get_reward(self, physics):
        """Returns a reward to the agent."""
        return 0


def simple_demo():
    env = my_env()
    action_spec = env.action_spec()
    print(action_spec)

    def random_policy(time_step):
        del time_step
        return np.random.uniform(
            low=action_spec.minimum, high=action_spec.maximum, size=action_spec.shape
        )

    viewer.launch(env, policy=random_policy)


if __name__ == '__main__':
    simple_demo()